import numpy as np
import torch
torch.multiprocessing.set_start_method("forkserver", force=True)
torch.multiprocessing.set_sharing_strategy('file_system')
import json
from functools import partial
import time
from concurrent.futures import ProcessPoolExecutor
import math

def opt_topk(i):
    return i

def opt_topp(i, cumsum, index):
    # sorting is done outside to save computation
    # it's ok to divide p into the same number of intervals as k, but consumes time !!!
    p_thres = i/2000
    top_p_ind = index[cumsum<p_thres]
    top_p_ind = [ind.item() for ind in top_p_ind]
    return len(top_p_ind)

def opt_delta_conf(i, delta_conf, index):
    # sorting is done outside to save computation
    vs = [i / 1e7 for i in range(1000)] + [1e-4 + i / 1e6 for i in range(0, 1000)] + [
        1e-3 + 1e-4 + i * (1 - 1e-3 - 1e-4) / 2000 for i
        in range(0, 2000)]
    p_thres = vs[i]
    # till the last index with the condition delta_conf > threshold satisfied

    thres = index[delta_conf>p_thres]
    if len(thres)>0:
        thres = thres[-1]
        ind = index.tolist().index(thres)
        allowed = index[:ind+1]
        allowed = [allow.item() for allow in allowed]
    else:
        allowed = []

    return len(allowed)

def opt_eta(i, sort, index):
    # sorting is done outside to save computation
    # the small epsilon values near 0 is the critical low-risk region, so use denser grid for this region
    vs = [i / 1e7 for i in range(1000)] + [1e-4 + i / 1e6 for i in range(0, 1000)] + [i / 1000 for i in range(1000)]
    epsilon = vs[i]
    eta = torch.minimum(torch.tensor(epsilon),
                        torch.sqrt(torch.tensor(epsilon)) * torch.exp(
                            (sort * torch.log(sort)).sum()))

    allowed = index[sort>eta]
    allowed = [allow.item() for allow in allowed]

    return len(allowed)

# https://github.com/basusourya/mirostat/blob/master/mirostat.py
def estimate_s(prob):
  result = 0
  num = 0
  den = 0
  for i in range(100):
    b = prob[i]/prob[i+1]
    t = (i+2)/(i+1)
    num += math.log(b)*math.log(t)
    den += math.log(t)**2
  return num/den

# https://github.com/basusourya/mirostat/blob/master/mirostat.py
def compute_k(n,s,tau):
    eps = s-1
    k = ((eps*(2**(tau)))/(1-n**(-eps)))**(1/s)
    k = round(k)
    return k

def opt_mirostat(i, logits, n):
    # sorting is done outside to save computation
    # it's ok to divide p into the same number of intervals as k, but consumes time !!!
    logits = torch.tensor(logits)

    tau = 10/4000 * i
    max_surprise = 2*tau

    sorted_logits, sorted_indices = torch.sort(logits, descending=True)

    prob_original = sorted_logits.tolist()

    # Estimate s
    s = estimate_s(prob_original)
    # Compute k
    k = compute_k(n, s, max_surprise) + 1

    # this k is not indexed, because it means the numbering starts from 1
    allowed = sorted_indices[0:k]

    allowed = [allow.item() for allow in allowed]

    return len(allowed)




if __name__ =="__main__":

    model_list = [
        'meta-llama/Llama-2-7b-hf',
        "meta-llama/Llama-2-70b-hf",
        "meta-llama/Meta-Llama-3-8B",
        "meta-llama/Meta-Llama-3-70B", # the model is 140G, each parameter is around 2 bytes
        "mistralai/Mistral-7B-v0.1"
        "mistralai/Mixtral-8x7B-v0.1",
        "openai-community/gpt2-xl",
    ]

    file = "full_test"
    with open(file, "r") as fp:
        stats = json.load(fp)

    good_ids_list = [[] for _ in range(len(model_list))]
    critical_values = [[] for _ in range(len(model_list))]
    logits_list = [[] for _ in range(len(model_list))]
    raw_logits_list = [[] for _ in range(len(model_list))]
    vocab_sizes = [[] for _ in range(len(model_list))]

    for m in range(len(model_list)):
        mn = model_list[m].split("/")[1]
        seg = f"{mn}"
        starts = stats[seg]["starts"]
        model_list[m] = stats[seg]["model"]
        good_ids_list[m] = stats[seg]["good_ids_list"]
        critical_values[m] = stats[seg]["critical_values"]
        logits_list[m] = stats[seg]["logits_list"]
        raw_logits_list[m] = stats[seg]["raw_logits_list"]
        vocab_sizes[m] = stats[seg]["vocab_sizes"]

    workers = 24
    border = [[[] for _ in range(len(good_ids_list[id]))] for id in range(len(model_list))]

    for id in range(len(model_list)):
        for n in range(len(good_ids_list[id])):
            print(f"Evaluating {n}th node")
            s = time.time()
            func = partial(opt_topk)
            with ProcessPoolExecutor(max_workers=workers) as executor:
                # small k values are within critical low-risk region
                # to save computation, we use a sparser grid for large k values
                for i, out in enumerate(executor.map(func, list(range(0, 1000, 1))+list(range(1000, vocab_sizes[id], (vocab_sizes[id]-1000)//1000)))):
                    border[id][n].append(out)
            e = time.time()
            print(f"each node consumes {s-e} time")

    for id in range(len(model_list)):
        m = model_list[id].split("/")[-1]
        border_ = np.array(border[id])
        np.save(f"test/border_top_k_{m}", border_)


    # top_p takes 307 s/node using 10000 interval, 6 times of top_k,
    # reduce the density of evaluation points to 2000 interval, then 61s/node
    workers = 24
    border = [[[] for _ in range(len(good_ids_list[0]))] for id in range(len(model_list))]

    for id in range(len(model_list)):
        for n in range(len(good_ids_list[0])):
            print(f"Evaluating {n}th node")
            s = time.time()
            sort = torch.sort(torch.tensor(logits_list[id][n]), -1, descending=True)
            func = partial(opt_topp, cumsum=torch.cumsum(sort[0], -1), index = sort[1])
            with ProcessPoolExecutor(max_workers=workers) as executor:
                # print(f"multiprocessing: {its}")
                for i, out in enumerate(executor.map(func, list(range(2000)))):
                    # print(out)
                    border[id][n].append(out)
            e = time.time()
            print(f"each node consumes {s-e} time")

    for id in range(len(model_list)):
        m = model_list[id].split("/")[-1]
        border_ = np.array(border[id])
        np.save(f"test/border_top_p_{m}", border_)


    workers = 24
    delta_conf = [[[] for _ in range(len(good_ids_list[id]))] for id in range(len(model_list))]
    border = [[[] for _ in range(len(good_ids_list[id]))] for id in range(len(model_list))]
    for id in range(len(model_list)):
        for n in range(len(good_ids_list[id])):
            print(f"Evaluating {n}th node")
            s = time.time()
            # print(logits_list[id][n])
            # exit()
            index = torch.sort(torch.tensor(logits_list[id][n]), -1, descending=True)[1][1:-1]
            sort = torch.sort(torch.tensor(logits_list[id][n]), -1, descending=True)[0][1:-1]

            delta_conf[id] = [(1 / torch.log(torch.tensor(vocab_sizes[id])) * (
                    torch.sort(torch.tensor(logits_list[id][n]), -1, descending=True)[0][1:-1] * torch.log(
                torch.sort(torch.tensor(logits_list[id][n]), -1, descending=True)[0][1:-1]) + (
                            1 - torch.cumsum(torch.sort(torch.tensor(logits_list[id][n]), -1, descending=True)[0],
                                             dim=-1)[1:-1]) * torch.log(
                (1 - torch.cumsum(torch.sort(torch.tensor(logits_list[id][n]), -1, descending=True)[0], dim=-1)[
                     1:-1]) / (
                        vocab_sizes[id] - torch.arange(1, vocab_sizes[id] - 1))) - (
                            1 - torch.cumsum(torch.sort(torch.tensor(logits_list[id][n]), -1, descending=True)[0],
                                             dim=-1)[:-2]) * torch.log(
                (1 - torch.cumsum(torch.sort(torch.tensor(logits_list[id][n]), -1, descending=True)[0], dim=-1)[
                     :-2]) / (
                        vocab_sizes[id] - torch.arange(vocab_sizes[id] - 2))))).nan_to_num(0)

                                                 for
                                                 n in range(0, len(logits_list[id]))]
            func = partial(opt_delta_conf, delta_conf=delta_conf[id][n], index=index)
            with ProcessPoolExecutor(max_workers=workers) as executor:
                # print(f"multiprocessing: {its}")
                for i, out in enumerate(executor.map(func, list(range(4000)))):
                    # print(out)
                    border[id][n].append(out)
            e = time.time()
            print(f"each node consumes {s-e} time")

    for id in range(len(model_list)):
        m = model_list[id].split("/")[-1]
        border_ = np.array(border[id])
        np.save(f"test/border_delta_conf_{m}", border_)


    workers = 24
    border = [[[] for _ in range(len(good_ids_list[id]))] for id in range(len(model_list))]
    for id in range(len(model_list)):
        for n in range(len(good_ids_list[id])):
            print(f"Evaluating {n}th node")
            s = time.time()
            # print(logits_list[id][n])
            # exit()
            sort, index = torch.sort(torch.tensor(logits_list[id][n]), -1, descending=True)

            func = partial(opt_eta, sort=sort, index=index)
            with ProcessPoolExecutor(max_workers=workers) as executor:
                # print(f"multiprocessing: {its}")
                for i, out in enumerate(executor.map(func, list(range(3000)))):
                    # print(out)
                    border[id][n].append(out[1])
            e = time.time()
            print(f"each node consumes {s-e} time")

    for id in range(len(model_list)):
        m = model_list[id].split("/")[-1]
        border_ = np.array(border[id])
        np.save(f"test/border_eta_{m}", border_)

    workers = 24
    border = [[[] for _ in range(len(good_ids_list[id]))] for id in range(len(model_list))]
    for id in range(len(model_list)):
        for n in range(len(good_ids_list[id])):
            print(f"Evaluating {n}th node")
            s = time.time()
            # print(logits_list[id][n])
            # exit()
            sort, index = torch.sort(torch.tensor(logits_list[id][n]), -1, descending=True)

            func = partial(opt_mirostat, logits=logits_list[id][n], n=vocab_sizes[id])
            with ProcessPoolExecutor(max_workers=workers) as executor:
                # print(f"multiprocessing: {its}")
                for i, out in enumerate(executor.map(func, list(range(4000)))):
                    # print(out)
                    border[id][n].append(out[1])
            e = time.time()
            print(f"each node consumes {s-e} time")
    for id in range(len(model_list)):
        m = model_list[id].split("/")[-1]
        border_ = np.array(border[id])
        np.save(f"test/border_mirostat_{m}", border_)