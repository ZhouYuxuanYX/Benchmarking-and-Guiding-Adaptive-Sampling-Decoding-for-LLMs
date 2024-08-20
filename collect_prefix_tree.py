# if without force=True, otherwise causing RuntimeError: context has already been set error
# or put under if __name__ == '__main__':

import torch.nn.functional as F
# from gramformer import Gramformer
import torch

# see explanation of fork, spawn and forkserver, https://bnikolic.co.uk/blog/python/parallelism/2019/11/13/python-forkserver-preload.html
torch.multiprocessing.set_start_method("forkserver", force=True)
torch.multiprocessing.set_sharing_strategy('file_system')
import networkx as nx
import os
# must set the environment variable before importing transformers, otherwise it won't work!!!!!!!!
# as the default one will be used first and setting it will not work!!!!!!
# must include huggingface, otherwise it will not find the correct token under hub
# os.environ['HF_HOME'] = '/p/scratch/hai_recmax/cache/huggingface'
# os.environ["TRANSFORMERS_OFFLINE"] = "1"
"""
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
        - Avoid using `tokenizers` before the fork if possible
        - Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
"""
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from concurrent.futures import ProcessPoolExecutor
from datasets import load_dataset
from nltk import word_tokenize
from nltk import tokenize

import nltk
nltk.download('punkt_tab')
from functools import partial

import json
# language-tool-python needs to install languagetool-standalone in ~/.cache and thus causes disk quota exceeded error when installing
# manually specify the download path when installing, LTP_PATH=/p/project/hai_recmax pip install language_tool_python
# it also needs to select from us-en and uk-en, use auto
import language_tool_python as lang
import os
# from gramformer import Gramformer


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

def grammarChecker(text, tool):
    result = tool.check(text)
    # print(result)
    return result


# handle paragraphs end with abreviations with their own '.'
# https://stackoverflow.com/questions/42101027/keep-trailing-punctuation-in-python-nltk-word-tokenize
def trailing(test_str):
    # only if test_str is empty '', then the index -2 will cause error
    toks = word_tokenize(test_str + " .")

    if len(toks) > 1 and len(toks[-2]) > 1 and toks[-2].endswith("."):
        pass  # Keep the added period
    else:
        toks = toks[:-1]
    return toks

def opt(inps):
    # split by paragraph
    # inps = inps.split('\n')
    # split by sentences
    inps = tokenize.sent_tokenize(inps)

    docs_ = []
    # use d.split(" ") will not handle the trailing punctuations well, because there's no space between period and the last word for example
    for d in inps:
        # avoid empty string and only spaces
        if d.strip() != "":
            # print("before", d)
            d = trailing(d)
            # print("after", d)
            # print(d)
            if d[-1] in set(['.', ':', '"', '\'', '?']) and len(d) > 3:
                docs_.append(d)
    return docs_

def text_filter(inputs, outputs, depth=0):
    if depth > 6 or len(inputs) < 2:
        # print("reaches bottom depth")
        return
    for idx, parag in enumerate(inputs):
        # if idx % 10000 == 0:
        #     print(f"{idx}th paragraph")
        # print(parag)
        if type(parag) is not list:

            parag = parag.split(" ")

        # since all the paragraphs of wikitext starts with " ", string.split(" ") will return '' as the first item, also ''.split(" ") will retrun '' as well
        if "" in parag:
            parag.remove("")
        if '\n' in parag:
            parag.remove('\n')

        # if parag contains non-common words, ignore parag
        ignore = False
        # if use for loop at every depth, it will be super slow, and it suffices to only check at the first depth, because the unqualified sentences will be ignored
        if depth == 0:
            for s in parag:
                # should exclude '\n', which appear at the end of every paragraph
                # no paragraph in wikitext-103 is qualified, because it contains a lot of foreign words and strange stuffs, change a dataset
                # print(s)
                # to handle the problem of the first letter of a beginning word being captalized and thus no in words.txt
                # this can also avoid specific names, which of course are not expected to be predicted by llms
                if s.lower() not in words1:
                    ignore = True
                    break
        # print("ignore:", ignore)
        if ignore:
            # should use continue instead of pass, pass will just pass the if condition
            continue

        # print(parag)
        # exclude empty list
        if parag and len(parag) > depth + 1:
            # print("updating")
            # check the length, otherwise :depth+1 will always retieve the full short text, e.g., the single word sentence "end" will appear at every depth
            if " ".join(parag[:depth + 1]) not in outputs:
                outputs.update({" ".join(parag[:depth + 1]): {f"leaves-{depth}": {" ".join(parag): 1}}})
                # print(f"leaves-{depth}")
            else:
                # use .items to get the counts
                # use len() to get the branches
                # here we don't exclude the repeated occurence of the same sentences, for counting the frequencies
                # remeber here the basic unit is paragraph, so it makes no sense to count the same paragraphs!!!!!
                if " ".join(parag) in outputs[" ".join(parag[:depth + 1])][f"leaves-{depth}"]:
                    outputs[" ".join(parag[:depth + 1])][f"leaves-{depth}"][" ".join(parag)] += 1
                else:
                    outputs[" ".join(parag[:depth + 1])][f"leaves-{depth}"].update({" ".join(parag): 1})
            # don't use return here, otherwise the for loop will stop at the first parag for the upper level
            # this will waste computation if the dict will be updated in the later loops, then it will recompute again and again

    for prefix in outputs:
        # print(outputs[prefix])
        # for cases where the nth word is not in the words.txt, there will be no leaves-depth
        if prefix != f"leaves-{depth}" and f"leaves-{depth}" in outputs[prefix]:
            text_filter(outputs[prefix][f"leaves-{depth}"].keys(), outputs[prefix], depth + 1)
        else:
            pass

def dict_count(prod, c=0):
    for mykey in prod:
        if mykey == "leveas-0":
            c += len(prod[mykey])
    return c


def checking(inp):
    # for free account there are some limits using public server, try reducing number of works
    # but using local server even no multiprocessing is possible, thread error
    # with lang.LanguageToolPublicAPI('en-US') as tool:
        # current state not use grammarcheck
        # return grammarChecker(inp[4:], tool), inp
    return None, inp


def identity(p):
    return p

"""insert an if __name__ == '__main__': guard in the main module to avoid creating subprocesses recursively."""
if __name__ == '__main__':
    os.environ["LTP_PATH"] = "/p/project/hai_recmax"
    ### aloutgh using python_language_tool not perfect
    # any words after "a list of" will not be considered as grammar mistake by this software except for spelling mistakes
    # for example, no error will be detected for "A list of reads"
    ### gramformer can not detect this problem either!!!! it just adds a "." to "A list of reads"
    # so it always judge unfinished sentence as grammarly wrong
    # in comparison python_language_tool is better
    mytext = "A list of books"
    # mytext = 'The following is#'
    # print(mytext[4:])

    # corrected_sentences = gf.correct(mytext, max_candidates=1)
    # print("[Input] ", mytext)
    # for corrected_sentence in corrected_sentences:
    #       print("[Edits] ", gf.highlight(mytext, corrected_sentence))
    #
    # exit()

    # 1) we could have a common vocabulary list to avoid strange sentences, e.g.,
    # there are many sentences in wikitext-103 start with ['SM', 'UB', '@-@', ...]

    '''
    A couple jumping off points to help you define "regular" for your project are:

    a frequency metric (does this word appear at least XX% of the time in your corpus)
    an agreement between sources (words that appear in all three of your word lists)
    '''
    '''
    1) For example, https://github.com/first20hours/google-10000-english
    This repo contains a list of the 10,000 most common English words in order of frequency, as determined by n-gram frequency analysis of the Google's Trillion Word Corpus.

    2) or https://gist.github.com/h3xx/1976236
    Wictionary top 100,000 most frequently-used English words
    3) or https://github.com/dwyl/english-words?tab=readme-ov-file
    this one is good, because it also contains abrevations e.g., 3D, 3rd, etc
    but it maybe contains too many words, which is not good for us to select sentences which are reasonable and with common expressions.
    words.txt contains all words.
    I checked that only lower case letters are contained in words_alpha.txt
    words_alpha.txt contains only [[:alpha:]] words (words that only have letters, no numbers or symbols). If you want a quick solution choose this.
    words_dictionary.json contains all the words from words_alpha.txt as json format. If you are using Python, you can easily load this file and use it as a dictionary for faster performance. All the words are assigned with 1 in the dictionary.
    '''
    # punctuations must be included besides common words
    punctuations = [',', '.', ':', '(', ')', '\'', '\"', ';', '?', '!']
    # numbers are to be considered as well
    numbers = [str(n) for n in range(10)]

    # for english words only
    # only lower case letters are contained in words_alpha.txt, use words.txt directly is better
    with open("words.txt") as f:
        words1 = f.readlines()
    for i, word in enumerate(words1):
        # to avoid upper and lower cases complexity
        # for example, it has "Center" but no "center", and many such problems
        words1[i] = word.replace('\n', '').lower()

    # converting list to set will save a lot of time!! because it uses hashing, much faster
    # or using dictionary is even faster!
    # 100+ times speedup
    words1 = set(words1 + punctuations)

    check = '.' in words1
    print("check: ", check)
    # exit()

    test = load_dataset("wikipedia", "20220301.en", split='train')

    # wikitext is not good, because it contains mulitple languages
    # test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    # load error
    # test = load_dataset("blog_authorship_corpus", split="test")
    # https://huggingface.co/datasets/wikipedia
    # from datasets import load_dataset

    # for english wikipedia, each paragraph is just separated by "\n"s, so i have to preprocess it

    # nyt tgz file not available
    # docs = load_dataset('irds/nyt', 'docs')
    # text = []
    # for doc in docs:
    #    text.append(doc["body"])

    # print(test[0])
    # features:"text"
    # print(test)

    ###### it is super slow when using the full dataset #########
    ###### create trajectory dictionary for each segment and merge by dict.update() function may help ###############
    # using 10w segment is very slow, comparing to using 1w segment
    #
    ############## step 1 #################
    # process will be killed if the whole dataset is fed at once, very likely due to out of memory issue
    # probably due to recursion, so we iteratively update the dictionary
    # takes 1 hour for the first 10w samples on haicore, 2 hours for the first 28w samples
    # 24 hours on haicore for 303w samples, and another 22 hours for the rest 300w samples

    ######## for starting from the beginning ##########
    start = 0
    segment = 10000
    its = 0
    trajects = {}
    workers = 16
    length = len(test["text"])

    print("####### total samples #########")
    # print(length)

    ########## second run reached 605
    # ######## for resume from previous compute ##############
    # # # because it exceeds 24 hours limit, we resume from 303w paragraphs
    # start = 3030000
    # # segment must be small, otherwise reading from data will be super slow
    # # 48 processes will handle a segment together at a time !!!!!
    # segment = 10000
    # its = 303
    # workers = 96
    # length = len(test["text"])

    # # for resume
    # with open('trajects_test.json', 'r') as fp:
    #     trajects = json.load(fp)

    while True:
        # resume by checking its, just take the rest starting from last its*segment
        print(its)

        if start >= length:
            break
        # empty evaluate to False
        if trajects:
            with open('trajects_test.json', 'r') as fp:
                trajects = json.load(fp)
        text = []
        # 10min for 10w paragraphs, 3 times faster with 16 workers; 7min 4.5 times faster with 32 workers; 5min 6 times faster with 48 workers
        with ProcessPoolExecutor(max_workers=workers) as executor:
            print(f"multiprocessing: {its}")
            for r in executor.map(opt, test["text"][start:start+segment]):
               # maybe this is the problem!!!!!!!!
               # has to double-check if each r are the same or not !!!!!!!!!!
               text.extend(r)
               # double checked, correct! they are not the same
               # print(r[0])
        start += segment
        its += 1

        # because each while loop the text will be checked from beginning, there are many replicates of the same paragrahs
        text_filter(text, trajects)

        if its % 10 == 0:
            print(f"total items: {dict_count(trajects)}")


        #text_filter(text, trajects)
        #print(trajects)
        with open('trajects_test.json', 'w') as fp:
            json.dump(trajects, fp)

