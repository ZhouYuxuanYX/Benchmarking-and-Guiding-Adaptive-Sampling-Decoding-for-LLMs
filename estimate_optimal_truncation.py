import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# !!!!!!!!!!!!must set the environment variable before importing transformers, otherwise the default one will be used first and setting it will not work!!!!!!
# must include huggingface, otherwise it will not find the correct token under hub
# the program will be killed when i run the code locally from the login node when loading checkpoint shards
# of many models together, very likely due to the ram usage, loading many models at the same time consumes too much ram
# i could do it sequentially
# os.environ['HF_HOME'] = '/p/scratch/hai_llm_diversity/cache/huggingface'
# save time on haicore!!!
# os.environ["TRANSFORMERS_OFFLINE"] = "1"
"""
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
        - Avoid using `tokenizers` before the fork if possible
        - Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
"""
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from transformers import AutoModelForCausalLM, LlamaTokenizer, AutoTokenizer, GPT2LMHeadModel, GPT2TokenizerFast
from concurrent.futures import ProcessPoolExecutor
import matplotlib.pyplot as plt
import numpy as np
import json
# if without force=True, otherwise causing RuntimeError: context has already been set error
# or put under if __name__ == '__main__':
# see explanation of fork, spawn and forkserver, https://bnikolic.co.uk/blog/python/parallelism/2019/11/13/python-forkserver-preload.html
torch.multiprocessing.set_start_method("forkserver", force=True)
torch.multiprocessing.set_sharing_strategy('file_system')

##### calculate the frequencies of sub trajectories for the current node
def get_freq(example, depth):
    freq_dict = {}
    total_counts = 0
    if depth > -1:
        if isinstance(example[1], dict):
            for d in example[1].keys():
                # print(d)
                # skip the last "leaves-*" key
                if "leaves" in d:
                    break
                # print(list(example[1][d].keys())[-1])
                # we check the "leaves-depth+1" len for a depth tr in its sub dict

                # our collected data, the set(['.', ':', '"', '\'', '?']) is separated with space
                # but the model will predict these marks without space
                # so we have to change it back to not missing the match when checking good_ids
                if d[-1] in set(['.', ':', '"', '\'', '?', ',']) and d[-2] == " ":
                    d_ = d[:-2]
                    d_ += d[-1]
                else:
                    d_ = d
                if isinstance(example[1][d],dict):
                    # freq_dict[d_] = len(example[1][d][f"leaves-{depth + 1}"])
                    freq_dict[d_] = sum([int(c) for c in example[1][d][f"leaves-{depth + 1}"].values()])
                    # total_counts += len(example[1][d][f"leaves-{depth + 1}"])
                    total_counts += sum([int(c) for c in example[1][d][f"leaves-{depth + 1}"].values()])
                else:
                    # for sentence-based tree, it goes to the end when the sentence ends, so no freq_dict anymore
                    # we thus exclude this node
                    freq_dict[d_] = int(example[1][d])
        else:
            pass


    # for starting word as empty string ""
    else:
        for d in example.keys():
            # print(d)
            # skip the last "leaves-*" key
            if "leaves" in d:
                break
            # print(list(example[1][d].keys())[-1])
            # we check the "leaves-depth+1" len for a depth tr in its sub dict

            # our collected data, the set(['.', ':', '"', '\'', '?']) is separated with space
            # but the model will predict these marks without space
            # so we have to change it back to not missing the match when checking good_ids
            # if d[-1] in set(['.', ':', '"', '\'', '?', ',']) and d[-2] == " ":
            #     d_ = d[:-2]
            #     d_ += d[-1]
            # else:
            d_ = d

            # freq_dict[d_] = len(example[d][f"leaves-{depth + 1}"])
            freq_dict[d_] = sum([int(c) for c in example[d][f"leaves-{depth + 1}"].values()])
            # total_counts += len(example[d][f"leaves-{depth + 1}"])
            total_counts += sum([int(c) for c in example[d][f"leaves-{depth + 1}"].values()])
    return freq_dict, total_counts

def dict_count(prod, c=0):
    for mykey in prod:
        if mykey == "leveas-0":
            c += len(prod[mykey])
    return c

"""
https://github.com/pytorch/pytorch/issues/83973
Currently, when a user calls the functions torch.cuda.device_count or torch.cuda.is_available, 
PyTorch initializes some CUDA context that prevents us to fork processes. We get the well known error message "Cannot re-initialize CUDA in forked subprocess" as demonstrated in the code below:
"""
# this causes gpu out of memory even with llama-7b !!!!!
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# split llama into fractions to fit on multiple gpu !!!!!!!
"""https://discuss.huggingface.co/t/how-to-load-large-model-with-multiple-gpu-cards/18522/7
"""

def checking(inp):
    # for free account there are some limits using public server, try reducing number of works
    # but using local server even no multiprocessing is possible, thread error
    # with lang.LanguageToolPublicAPI('en-US') as tool:
        # current state not use grammarcheck
        # return grammarChecker(inp[4:], tool), inp
    return None, inp

def topk_func(k, inp, good_ids1, bad_ids1):

    topk, topp_index1 = torch.topk(inp, k, -1)
    topp_index1 = topp_index1.tolist()
    logits_1 = torch.softmax(inp, -1)
    inter_data1 = list(set(topp_index1) & set(good_ids1))

    # inter_bad1 = list(set(topp_index1) & set(bad_ids1))
    # inter_not_data1 = list(set(topp_index1) - set(inter_data1))
    logits_inter_data1 = logits_1[inter_data1] / logits_1[inter_data1].sum()
    entropy1 = - (logits_inter_data1 * torch.log(logits_inter_data1)).sum().item()

    good_mass_topp1 = logits_1[inter_data1].sum().item()
    # not_data_mass_topp1 = logits_1[inter_not_data1].sum().item() / logits_1[topp_index1].sum().item()

    # bad_mass_topp1 = logits_1[inter_bad1].sum().item() / logits_1[topp_index1].sum().item()

    return entropy1, good_mass_topp1, inter_data1

def identity(p):
    return p


def run_model(model1, tokenizer1, model_name1, start, freq_dict):
    # "<s>" is a token that signifies start of the string

    # convert the list of all possible next words to id, and assign the freqs to each id position to construct the gt prob vector
    # one problem is that some word might be two tokens
    # gt_encodings = tokenizer()
    # so the solution might be we let llm predict next token,
    # we convert each token from the full vocabulary of llm model to string and then check if
    # 1) it is a part of the real word, e.g, "ice" in "icecream" or it is equal to the real word, e.g., list==list
    # the discrepancy between token and word is also a important thing that perplexity doesn't consider

    # if input as a string, there will be problem, because some words might be divided into multiple tokens
    # then all the trajectories are mixed together
    # so better input as a list, where each word is a sample
    # ValueError: Asking to pad but the tokenizer does not have a padding token.
    # Please select a token to use as `pad_token` `(tokenizer.pad_token = tokenizer.eos_token e.g.)`
    # or add a new pad token via `tokenizer.add_special_tokens({'pad_token': '[PAD]'})`

    # for batchwise processing, add the pad, because some words may be composed of more than one tokens
    # tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    # gt_encodings = tokenizer(list(freq_dict.keys()), return_tensors="pt", padding=True, add_special_tokens=True)
    # print(gt_encodings.input_ids)
    #
    # # convert_ids_to_tokens only works with a sinlge string or a list of strings
    # gt_tokens = []
    # for s in gt_encodings.input_ids:
    #     gt_tokens.append(tokenizer.convert_ids_to_tokens(s))
    # print(gt_tokens)

    # we use for loop for each sample, making it easier without using padding
    print(start)
    gt_encodings1 = []

    gt_strings = []
    for tr in list(freq_dict.keys()):
        # since the empty string input "" is appended with bos, here we have to append the gt strings with bos too
        # print(tr)
        # exit()
        if "gpt2" in model_name1:
            tr = tokenizer1.bos_token + tr
        gt_strings.append(tr)
        gt_encodings1.append(tokenizer1(tr, return_tensors="pt"))

    if "gpt2" in model_name1:
        start = tokenizer1.bos_token + start

    encodings1 = tokenizer1(start, return_tensors="pt")
    input_tokens1 = tokenizer1.convert_ids_to_tokens(encodings1.input_ids.squeeze().tolist())
    print("################# input tokens #############")
    print(input_tokens1)
    print("################# input ids #############")
    print(encodings1.input_ids.squeeze().tolist())

    gt_ids1 = []
    input_token_list1 = encodings1.input_ids.squeeze().tolist()
    if not isinstance(input_token_list1, list):
        input_token_list1 = [input_token_list1]
    # since gpt does not have start of sentence, for empty string "" input, we have empty list []

    # print(input_token_list1)
    # exit()
    if len(input_token_list1):
        input_token_end_indx1 = input_token_list1.index(input_token_list1[-1])
    else:
        input_token_end_indx1 = -1
    # print(input_token_end_indx1)
    for s in gt_encodings1:
        # print(s)
        # start from the index for the last input token, instead of using index number, because there's a risk of extra tokens, e.g., bos token
        # it's tensor in original format and not compatible with convert to string with more than one ids
        # a single word might correspond to multiple tokens and thus multiple ids
        # print(s.input_ids)
        # exit()
        id = s.input_ids.squeeze(0)[input_token_end_indx1 + 1:].tolist()

        # for example, The first is a short scene, has no ids further and append [] everytime
        # ########### git_ids1 ###########
        # [[], [], []]
        # this can also be seen in the graph.png
        if not id:
            pass
        else:
            gt_ids1.append(id)

        if not gt_ids1:
            return None

    #### to construct the gt distribution in the form of the model vocabulary, we could simply assign the corresponding freq to
    ### a vector by gt_id index
    values = list(freq_dict.values())
    sum = np.array(values).sum()
    gt_prob1 = torch.zeros(model1.config.vocab_size)

    # print("##############")
    # print(f"{model_name1} vocab size: {len(gt_prob1)}")
    # print("##############")

    for num, id in enumerate(gt_ids1):
        gt_prob1[id[0]] = values[num] / sum

    # # # convert_ids_to_tokens only works with a sinlge string or a list of strings
    gt_tokens1 = []
    for s in gt_encodings1:
        gt_tokens1.append(tokenizer1.convert_ids_to_tokens(s.input_ids.squeeze().tolist()))
    # print("############ gt tokens last example ##########")
    # print(gt_tokens1[-1])
    # print("############ gt ids last example ##########")
    # print(s.input_ids.squeeze().tolist())

    # for visualization
    vocab_size1 = model1.config.vocab_size
    # input_ids (torch.LongTensor of shape (batch_size, sequence_length)) — Indices of input sequence tokens in the vocabulary.
    vocab_ids1 = torch.arange(vocab_size1).tolist()

    # until here it's fine
    # concate gt_ids as prefix first, to avoid the staring space being ignored using the single token alone at decoding
    vocab_tokens1 = tokenizer1.convert_ids_to_tokens(vocab_ids1)

    # for decoding correctly, concat first
    full_ids1 = []
    for i in range(len(vocab_ids1)):
        full_id = encodings1.input_ids.squeeze().tolist()
        # gpt-2 has no <s> or anything else at the sentence start
        if not isinstance(full_id, list):
            full_id = [full_id]
        full_id.append(vocab_ids1[i])
        full_ids1.append(full_id)

    # print(gt_ids1[-4:])
    # exit()

    full_tokens1 = []
    for full_id in full_ids1:
        full_tokens1.append(tokenizer1.convert_ids_to_tokens(full_id))

    # this will convert the list of tokens to a long string, which is not good
    # use for loop to store separately
    # there are 32000 tokens, let me visualize top-k
    vocabs1 = []
    for vocab_token in vocab_tokens1:
        # put single token into list as beginning token will make the staring space be ignored
        # e.g., "_list" will be decoded to "list" alone, if not put after some prefixes
        # for calculating probability, just check the ids to avoid such formatting difference
        vocabs1.append(tokenizer1.convert_tokens_to_string([vocab_token]))

    full_string1 = []
    for full_token in full_tokens1:
        # put single token into list as beginning token will make the staring space be ignored
        # e.g., "_list" will be decoded to "list" alone, if not put after some prefixes
        # for calculating probability, just check the ids to avoid such formatting difference
        full_string1.append(tokenizer1.convert_tokens_to_string(full_token))

    print(f"###### full string {model_name1}")

    import time
    bad_ids1 = []
    good_ids1 = []
    good_samples1 = []

    """
    huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
    To disable this warning, you can either:
            - Avoid using `tokenizers` before the fork if possible
            - Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
    """
    # os.environ["TOKENIZERS_PARALLELISM"] = "false"
    workers = 24

    torch.set_num_threads(1)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for num, out in enumerate(executor.map(checking, full_string1)):

            check, full = out

            if False:
                # because grammar check is not perfect, we can not guarantee the words without grammar check error is good!!!
                # so we need to only regard empricially presented words as good words
                pass

            elif any(vocab_tokens1[num] == g[input_token_end_indx1 + 1:][0] if len(
                    g) > input_token_end_indx1 + 1 else False for g in gt_tokens1):
                good_ids1.append(num)
                good_samples1.append(full)

    print("################")
    print(f"{num} tokens in the vocabulary checked")
    # if good_ids_list is empty, skip this node
    if not good_ids_list:
        return None

    # input_token_list1 = encodings1.input_ids.squeeze().tolist()
    # input_token_end_indx1 = input_token_list1.index(input_token_list1[-1])
    #
    ############### back translate to double-check #################
    # print(encodings1)
    # print(len(encodings1[0]))
    # print(encodings1.input_ids)
    # print(input_tokens1)
    # output = tokenizer1.convert_tokens_to_string(input_tokens1)
    # print(input_tokens1)
    # print("back translate")
    # print(output)

    # print(tokenizer2.convert_ids_to_tokens(encodings2.input_ids.squeeze().tolist()))
    # print(encodings2)
    # print(len(encodings2[0]))
    # exit()

    # use n_positions instead of max_length
    # DEPRECATED. Use logits_processor or stopping_criteria directly to cap the number of generated tokens. The maximum length of the sequence to be generated
    # default is only 50
    # print("model.config.max_length", model.config.max_length)

    ####### evaluate on tree data ########
    # device = "cuda"
    max_length1 = model1.config.max_position_embeddings

    stride = 512
    seq_len1 = encodings1.input_ids.size(1)
    device = "cpu"
    # put the input on cuda seems to work

    # on the cluster
    # input_ids1 = encodings1.input_ids.cuda()

    # on the login node, it is much faster (almost 10 times) than using gpu on haicore!
    # on the login node
    input_ids1 = encodings1.input_ids.to(device)

    # don't forget to use this!!!!! otherwise there will out of memory issue
    with torch.no_grad():
        outputs1 = model1(input_ids1, labels=None, output_hidden_states=True)

    # raw scores before softmax
    logits1 = outputs1.logits.to(device)

    # can't decode top-5 together at once
    top_probs1, top_inds1 = logits1.topk(1, -1)

    tokens1 = tokenizer1.convert_ids_to_tokens(top_inds1.squeeze().tolist())

    # llm only predicts next word for the whole sequence, i.e., for each word of the given input, it predicts the next word only
    # so have to use for loop for longer prediction

    logits1_ = logits1.squeeze(0)[-1]
    # print(logits1_.shape)
    logits_1 = torch.nn.functional.softmax(logits1_, -1)


    plt.figure(figsize=(10, 15))
    # # will raise error if not list type, only single element tensor can be used as index
    # # also convert vocabs to numpy array, so that list can be used as indices
    # # torch tensor requires real value, not str
    # vocabs1 = np.array(vocabs1)
    # there will be a start sign after decoding in the beginning, e.g., "<s> The following is"
    name = model_name1.split("/")[-1]

    # mask1 = [indice not in bad_ids1 for indice in top_indices1]
    # mask2 = [indice in bad_ids1 for indice in top_indices1]

    # first_grammar_k = bad_ids1[0]
    # first_grammar_p = logits_1[:first_grammar_k+1].sum().item()

    # plt.title(f"{start}")
    # profit_color = [{indice not in bad_ids1: 'blue', indice in bad_ids1: 'red'}[True] for indice in top_indices1]
    # ax = plt.barh(np.arange(len(top_indices1)), top_logits1.tolist(),
    #          tick_label=np.array([s[4:] for s in np.array(full_string1)[top_indices1.tolist()].tolist()]), color=profit_color)
    #
    # plt.savefig(f"figures/hist_llm_{name}_{start}.png")

    # plt.figure(figsize=(10, 15))
    # # print(gt_prob1)
    # # print(len(gt_prob1))
    # plt.title(f"{start}")
    # ax = plt.barh(list(range(len(top_indices1))), gt_prob1[top_indices1.tolist()],
    #          tick_label=[s[4:] for s in np.array(full_string1)[top_indices1.tolist()].tolist()], color='b')
    # # # plt.xscale("log")
    # # this just changes the bounds of axis
    # # ax[0].set_bounds(6, 1, 1, 1)
    # plt.savefig(f"figures/hist_empirical_{name}_{start}.png")

    # # 1. we don't want to care about frequencies, because we don't want llm output to represent the most frequent words, but
    # # as long as it's used by human beings, it's fine
    # # 2. maybe we should use sentences as basic samples, instead of paragraphs, because anyway after a few prefix words, only one
    # # paragraph has the same prefix, so most of the paragraph content is not ultilized for our evaluation
    # # or we go for larger dataset, e.g., openwebtext, might be better
    # # 3. or we only evaluate at nodes where enough sub-branches exist, but we don't check the whole sub-tree branches,
    # # for evaluation, we should check the next-level sub-branch number !

    if not good_ids1:
        return None

    critical = max([torch.sort(logits1_, dim=-1, descending=True)[1].tolist().index(id) for id in good_ids1])


    return good_ids1, logits_1, logits1_, critical, len(vocab_ids1)


if __name__ == '__main__':

    with open('sorted_trajects_test.json', 'r') as fp:
        sorted_trajects = json.load(fp)

    graph_data = sorted_trajects

    # the following recursive function has a risk of getting memory error on the login node
    # try set the two lists as global variable
    def retrieve_trajects(example, current_depth, final_depth,
                          # , starts=[], freq_dicts=[]
                          ):
        if current_depth == final_depth:
            # return starts, freq_dicts
            return
        current_depth += 1
        # at each depth we have a span of 2
        examples = list(example[1].items())
        for m in range(min(span, len(examples))):
            # get the key-value pairs of values subdictionary
            # it is key-value tuple!!!!!!!!!!
            if not isinstance(examples[m], tuple):
                # print(examples[m])
                continue

            freq, counts = get_freq(examples[m], current_depth)
            if freq is not None:
                freq_dicts.append(freq)
                start = examples[m][0]
                print(start)
                starts.append(start)
            # if it reaches the end of a sentence, then no subbranches are available
            else:
                continue

            retrieve_trajects(examples[m], current_depth, final_depth)
        return

    ############# calculate critical values with full data coverage #################
    stats = {}
    # # to resume
    # file = "full_test"
    # with open(file, "r") as fp:
    #     stats = json.load(fp)

    # currently use a fixed span of 2 for each depth
    span = 2

    for n_tree in range(0, 10):
        starts = []
        freq_dicts = []

        current_depth = 0
        final_depth = 5
        # get key-value pairs of the subroot, i.e., first word of a sentence, e.g., "The"
        example = list(sorted_trajects.items())[n_tree]
        # get the key
        start = example[0]
        starts.append(start)

        freq_dicts.append(get_freq(example, current_depth)[0])
        # print(freq_dicts)
        retrieve_trajects(example, current_depth, final_depth)

        model_list = [
             "openai-community/gpt2-xl",
             # 'meta-llama/Llama-2-7b-hf',
             # "meta-llama/Meta-Llama-3-8B",
             # "meta-llama/Meta-Llama-3-70B", # the model is 140G, each parameter is around 2 bytes
             # "meta-llama/Llama-2-70b-hf", # these 70b models[ as well as mistral moe models, e.g., 8x7b are downloaded manually,
             # "mistralai/Mixtral-8x7B-v0.1",
             # "mistralai/Mistral-7B-v0.1",
            ]

        ################ run #############
        good_ids_list = [[] for _ in range(len(model_list))]
        top_95_inter_data_ids_list = [[] for _ in range(len(model_list))]
        inter_datas_topp_list = [[] for _ in range(len(model_list))]
        datas_normalized_topp_temp1_list = [[] for _ in range(len(model_list))]
        bad_topp_list = [[] for _ in range(len(model_list))]
        critical_values = [[] for _ in range(len(model_list))]
        critical_values_top_95 = [[] for _ in range(len(model_list))]
        coverage_list = [[] for _ in range(len(model_list))]
        logits_list = [[] for _ in range(len(model_list))]
        raw_logits_list = [[] for _ in range(len(model_list))]
        vocab_sizes = [[] for _ in range(len(model_list))]

        for num, model_name in enumerate(model_list):
            # please wait a few minutes every time after printing "full string", it is normal that there's a wait time before multiprocessing loop starts
            print("########## model name ##############")
            print(model_name)
            # home directory disk quota exceeded, change hugging face cache directory
            # device_map='auto' already knows to split the model into pieces to fit into gpu memory!!!!!
            model1 = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map='auto',
            )
            model1.eval()
            if "gpt2" in model_name:
                # gpt2 doesn't have bos token
                tokenizer1 = AutoTokenizer.from_pretrained(model_name,
                                                           bos_token="<|startoftext|>",
                                                           # eos_token="<|endoftext|>",
                                                           # pad_token="<|pad|>",
                                                           # sep_token="<|sep|>"
                                                           )
                model1.resize_token_embeddings(len(tokenizer1))

            else:
                tokenizer1 = AutoTokenizer.from_pretrained(model_name)

            for n in range(0, len(starts)):
                print(f"at {n}th node")

                # download the model first before submitting to cluster, because there's no internet connection there
                # using this way will be killed when loading llama-70B model, download manually instead
                # must set the environment variable to download to scratch, otherwise disk quota exceeded
                # only for llama-3, need "--include "original/*"
                # llama-2 has hf version explicity, don't use "--include "original/*", otherwise nothing will be downloaded
                # and even with this it lacks config file, so run the tokenizer1 = ... code on the login node first, such that config can be downloaded
                # seems that the model will be downloaded in local-dir
                #  HF_HOME='/p/scratch/hai_recmax/cache/huggingface' huggingface-cli download meta-llama/Meta-Llama-3-70B --include "original/*" --local-dir Meta-Llama-3-70B
                # mistralai/Mixtral-8x22B-v0.1
                # tokenizer1 = AutoTokenizer.from_pretrained(model_name)
                # model = AutoModelForCausalLM.from_pretrained(
                #     model_name,
                #     device_map='auto',
                # )
                # exit()
                # pass
                outs = run_model(model1, tokenizer1, model_name, starts[n], freq_dicts[n])
                print("######## run model finished##########")
                # exit()

                if outs is not None:
                    # do nothing, go on with the rest of the code
                    pass
                else:
                    # skip this for loop
                    continue
                good_ids, logits_1, logits1_, critical, vocab_size = outs
                good_ids_list[num].append(good_ids)
                critical_values[num].append(critical)
                # json dump can't handle tensor
                logits_list[num].append(logits_1.tolist())
                raw_logits_list[num].append(logits1_.tolist())
                vocab_sizes[num] = vocab_size

        file = "full_test"
        for m in range(len(model_list)):
            mn = model_list[m].split("/")[1]
            seg = f"{mn}"

            if not stats:
                stats[seg] = {"starts": starts, "model": model_list[m], "good_ids_list": good_ids_list[m],
                              # "inter_data_list": inter_datas_topp_list[m],
                              "critical_values": critical_values[m],
                              "critical_values_top_95": critical_values_top_95[m], "logits_list": logits_list[m],
                              "raw_logits_list": raw_logits_list[m], "vocab_sizes": vocab_sizes[m], "top_95_inter_data_ids_list": top_95_inter_data_ids_list[m]}
                # with open(file, "w") as fp:
                #     json.dump(stats, fp)
            else:
                # with open(file, "r") as fp:
                #     stats = json.load(fp)
                if seg in stats:
                    stats[seg]["starts"].extend(starts)
                    stats[seg]["model"] = model_list[m]
                    stats[seg]["good_ids_list"].extend(good_ids_list[m])
                    stats[seg]["critical_values"].extend(critical_values[m])
                    stats[seg]["logits_list"].extend(logits_list[m])
                    stats[seg]["raw_logits_list"].extend(raw_logits_list[m])
                else:
                    stats[seg] = {"starts": starts, "model": model_list[m], "good_ids_list": good_ids_list[m],
                                  # "inter_data_list": inter_datas_topp_list[m],
                                  "critical_values": critical_values[m],
                                  "logits_list": logits_list[m],
                                  "raw_logits_list": raw_logits_list[m], "vocab_sizes": vocab_sizes[m],
                                  }
        with open(file, "w") as fp:
            json.dump(stats, fp)