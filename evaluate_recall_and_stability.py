import numpy as np
import matplotlib.pyplot as plt
import torch
import json

if __name__ =="__main__":
    model_list = [
        'meta-llama/Llama-2-7b-hf',
        "meta-llama/Llama-2-70b-hf",
        "meta-llama/Meta-Llama-3-8B",
        "meta-llama/Meta-Llama-3-70B",  # the model is 140G, each parameter is around 2 bytes
        "mistralai/Mixtral-8x7B-v0.1",
        "mistralai/Mistral-7B-v0.1",
        "mistralai/Mixtral-8x22B-v0.1",
        "openai-community/gpt2-xl"
    ]
    file = "full_test"
    with open(file, "r") as fp:
        stats = json.load(fp)

    good_ids_list = [[] for _ in range(len(model_list))]
    critical_values = [[] for _ in range(len(model_list))]
    logits_list = [[] for _ in range(len(model_list))]
    vocab_sizes = [[] for _ in range(len(model_list))]

    for m in range(len(model_list)):
        mn = model_list[m].split("/")[1]
        seg = f"{mn}"
        good_ids_list[m] = stats[seg]["good_ids_list"]
        critical_values[m] = stats[seg]["critical_values"]
        logits_list[m] = stats[seg]["logits_list"]
        vocab_sizes[m] = stats[seg]["vocab_sizes"]

        border_top_k = np.load(f"files_new/border_top_k_{mn}.npy")
        border_top_k[border_top_k == 0] = 1

        border_top_p = np.load(f"files_new/border_top_p_{mn}.npy")
        border_top_p[border_top_p == 0] = 1

        border_delta_conf = np.load(f"files_new/border_delta_conf_{mn}.npy")
        border_delta_conf[border_delta_conf == 0] = 1

        border_eta = np.load(f"files_new/border_eta_{mn}.npy")
        border_eta[border_eta == 0] = 1

        border_mirostat = np.load(f"files_new/border_mirostat_{mn}.npy")
        border_mirostat[border_mirostat == 0] = 1

        labels = []
        recalls = []
        risks = []

        risks.extend([
            np.maximum(0, border_top_k / np.expand_dims(np.array(critical_values[m]) + 1,
                                                        -1) - 1)[:,
            torch.min(torch.abs(
                torch.tensor(
                    np.maximum(0, border_top_k / np.expand_dims(np.array(critical_values[m]) + 1,
                                                                -1) - 1).mean(
                        0) - 15)), 0)[1]],
            np.maximum(0, border_top_p / np.expand_dims(np.array(critical_values[m]) + 1,
                                                                       -1) - 1)[:,
                   torch.min(torch.abs(
                       torch.tensor(
                           np.maximum(0, border_top_p / np.expand_dims(np.array(critical_values[m]) + 1,
                                                                       -1) - 1).mean(
                               0) - 15)), 0)[1]],
            np.maximum(0, border_delta_conf / np.expand_dims(np.array(critical_values[m]) + 1,
                                                        -1) - 1)[:,
            torch.min(torch.abs(
                torch.tensor(
                    np.maximum(0, border_delta_conf / np.expand_dims(np.array(critical_values[m]) + 1,
                                                                -1) - 1).mean(
                        0) - 15)), 0)[1]],
            np.maximum(0, border_eta / np.expand_dims(np.array(critical_values[m]) + 1,
                                                        -1) - 1)[:,
            torch.min(torch.abs(
                torch.tensor(
                    np.maximum(0, border_eta / np.expand_dims(np.array(critical_values[m]) + 1,
                                                                -1) - 1).mean(
                        0) - 15)), 0)[1]],
            np.maximum(0, border_mirostat / np.expand_dims(np.array(critical_values[m]) + 1,
                                                        -1) - 1)[:,
            torch.min(torch.abs(
                torch.tensor(
                    np.maximum(0, border_mirostat / np.expand_dims(np.array(critical_values[m]) + 1,
                                                                -1) - 1).mean(
                        0) - 15)), 0)[1]],
        ])
        recalls.extend([
            np.minimum(1, border_top_k / np.expand_dims(np.array(critical_values[m]) + 1, -1))[:,
                          torch.min(torch.abs(torch.tensor(np.maximum(0, border_top_k / np.expand_dims(
                              np.array(critical_values[m]) + 1, -1) - 1).mean(0) - 15)), 0)[1]],
                   np.minimum(1, border_top_p / np.expand_dims(np.array(critical_values[m]) + 1, -1))[:,
                          torch.min(torch.abs(
                              torch.tensor(
                                  np.maximum(0, border_top_p / np.expand_dims(np.array(critical_values[m]) + 1,
                                                                              -1) - 1).mean(
                                      0) - 15)), 0)[1]],
                   np.minimum(1, border_delta_conf / np.expand_dims(np.array(critical_values[m]) + 1, -1))[:,
                          torch.min(torch.abs(
                              torch.tensor(
                                  np.maximum(0, border_delta_conf / np.expand_dims(np.array(critical_values[m]) + 1,
                                                                                   -1) - 1).mean(
                                      0) - 15)), 0)[1]],
                   np.minimum(1, border_eta / np.expand_dims(np.array(critical_values[m]) + 1, -1))[:,
                          torch.min(torch.abs(
                              torch.tensor(
                                  np.maximum(0, border_eta / np.expand_dims(np.array(critical_values[m]) + 1,
                                                                            -1) - 1).mean(
                                      0) - 15)), 0)[1]],
                   np.minimum(1, border_mirostat / np.expand_dims(np.array(critical_values[m]) + 1, -1))[:,
                          torch.min(torch.abs(
                              torch.tensor(
                                  np.maximum(0, border_mirostat / np.expand_dims(np.array(critical_values[m]) + 1,
                                                                                 -1) - 1).mean(
                                      0) - 15)), 0)[1]]
                   ])

        a = torch.tensor(np.maximum(0, border_top_k / np.expand_dims(np.array(critical_values[m]) + 1,
                                                                     -1) - 1).mean(
            0))[torch.min(torch.abs(torch.tensor(np.maximum(0, border_top_k / np.expand_dims(
            np.array(critical_values[m]) + 1, -1) - 1).mean(0) - 15)), 0)[1]]


        b = torch.tensor(np.maximum(0, border_top_p / np.expand_dims(np.array(critical_values[m]) + 1,
                                                                     -1) - 1).mean(
            0))[torch.min(torch.abs(
            torch.tensor(
                np.maximum(0, border_top_p / np.expand_dims(np.array(critical_values[m]) + 1,
                                                            -1) - 1).mean(
                    0) - 15)), 0)[1]]

        c = torch.tensor(np.maximum(0, border_delta_conf / np.expand_dims(np.array(critical_values[m]) + 1,
                                                                          -1) - 1).mean(
            0))[torch.min(torch.abs(
            torch.tensor(
                np.maximum(0, border_delta_conf / np.expand_dims(np.array(critical_values[m]) + 1,
                                                                 -1) - 1).mean(
                    0) - 15)), 0)[1]]


        d = torch.tensor(np.maximum(0, border_eta / np.expand_dims(np.array(critical_values[m]) + 1,
                                                                   -1) - 1).mean(
            0))[torch.min(torch.abs(
            torch.tensor(
                np.maximum(0, border_eta / np.expand_dims(np.array(critical_values[m]) + 1,
                                                          -1) - 1).mean(
                    0) - 15)), 0)[1]]
        e = torch.tensor(np.maximum(0, border_mirostat / np.expand_dims(np.array(critical_values[m]) + 1,
                                                                        -1) - 1).mean(
            0))[torch.min(torch.abs(
            torch.tensor(
                np.maximum(0, border_mirostat / np.expand_dims(np.array(critical_values[m]) + 1,
                                                               -1) - 1).mean(
                    0) - 15)), 0)[1]]

        print("############check risk 15 parameters############")

        ks = np.array(list(range(0, 1000, 1)) + list(range(1000, vocab_sizes[m], (vocab_sizes[m] - 1000) // 1000)))
        print("k")
        print(ks[torch.min(torch.abs(torch.tensor(np.maximum(0, border_top_k / np.expand_dims(
            np.array(critical_values[m]) + 1, -1) - 1).mean(0) - 15)), 0)[1]])

        ps = np.array([i / 2000 for i in range(2000)])
        print("p")
        print(ps[torch.min(torch.abs(torch.tensor(np.maximum(0, border_top_p / np.expand_dims(
            np.array(critical_values[m]) + 1, -1) - 1).mean(0) - 15)), 0)[1]])

        eps = np.array([i / 1e7 for i in range(1000)] + [1e-4 + i / 1e6 for i in range(0, 1000)] + [1e-3 + 1e-4 + i * (1 - 1e-3 - 1e-4) / 2000 for i
                                                                                     in range(0, 2000)])
        print("epsilon adaptive sampling")
        print(eps[torch.min(torch.abs(torch.tensor(np.maximum(0, border_delta_conf / np.expand_dims(
            np.array(critical_values[m]) + 1, -1) - 1).mean(0) - 15)), 0)[1]])

        es = np.array(
            [i / 1e7 for i in range(1000)] + [1e-4 + i / 1e6 for i in range(0, 1000)] + [i / 1000 for i in range(1000)])
        print("epsilon eta-sampling")
        print(es[torch.min(torch.abs(torch.tensor(np.maximum(0, border_eta / np.expand_dims(
            np.array(critical_values[m]) + 1, -1) - 1).mean(0) - 15)), 0)[1]])

        ms = np.array([10 / 4000 * i for i in range(4000)])
        print("tau mirostat")
        print(ms[torch.min(torch.abs(torch.tensor(np.maximum(0, border_mirostat / np.expand_dims(
            np.array(critical_values[m]) + 1, -1) - 1).mean(0) - 15)), 0)[1]])

        print("############check risk 15 stats############")
        names = ["top-k", "top-p", "adaptive", "eta", "mirostat"]
        parameters = [ks, ps, eps, es, ms]
        for n in range(0, 5):
            print(names[n])
            print("risk mean, risk standard error, recall mean, recall standard error")
            print(f"{parameters[n]} & {risks[n].mean():10.3f} ({risks[n].std()/np.sqrt(len(risks[n])):10.3f}) & {recalls[n].mean():10.3f} ({recalls[n].std()/np.sqrt(len(recalls[n])):10.3f}"
                                )



        #################################


        labels = []
        recalls = []
        risks = []

        risks.extend([
            np.maximum(0, border_top_k / np.expand_dims(np.array(critical_values[m]) + 1,
                                                        -1) - 1)[:,
            torch.min(torch.abs(
                torch.tensor(
                    np.maximum(0, border_top_k / np.expand_dims(np.array(critical_values[m]) + 1,
                                                                -1) - 1).mean(
                        0) - 5)), 0)[1]],
            np.maximum(0, border_top_p / np.expand_dims(np.array(critical_values[m]) + 1,
                                                                       -1) - 1)[:,
                   torch.min(torch.abs(
                       torch.tensor(
                           np.maximum(0, border_top_p / np.expand_dims(np.array(critical_values[m]) + 1,
                                                                       -1) - 1).mean(
                               0) - 5)), 0)[1]],
            np.maximum(0, border_delta_conf / np.expand_dims(np.array(critical_values[m]) + 1,
                                                        -1) - 1)[:,
            torch.min(torch.abs(
                torch.tensor(
                    np.maximum(0, border_delta_conf / np.expand_dims(np.array(critical_values[m]) + 1,
                                                                -1) - 1).mean(
                        0) - 5)), 0)[1]],
            np.maximum(0, border_eta / np.expand_dims(np.array(critical_values[m]) + 1,
                                                        -1) - 1)[:,
            torch.min(torch.abs(
                torch.tensor(
                    np.maximum(0, border_eta / np.expand_dims(np.array(critical_values[m]) + 1,
                                                                -1) - 1).mean(
                        0) - 5)), 0)[1]],
            np.maximum(0, border_mirostat / np.expand_dims(np.array(critical_values[m]) + 1,
                                                        -1) - 1)[:,
            torch.min(torch.abs(
                torch.tensor(
                    np.maximum(0, border_mirostat / np.expand_dims(np.array(critical_values[m]) + 1,
                                                                -1) - 1).mean(
                        0) - 5)), 0)[1]],
        ])
        recalls.extend([
            np.minimum(1, border_top_k / np.expand_dims(np.array(critical_values[m]) + 1, -1))[:,
                          torch.min(torch.abs(torch.tensor(np.maximum(0, border_top_k / np.expand_dims(
                              np.array(critical_values[m]) + 1, -1) - 1).mean(0) - 5)), 0)[1]],
                   np.minimum(1, border_top_p / np.expand_dims(np.array(critical_values[m]) + 1, -1))[:,
                          torch.min(torch.abs(
                              torch.tensor(
                                  np.maximum(0, border_top_p / np.expand_dims(np.array(critical_values[m]) + 1,
                                                                              -1) - 1).mean(
                                      0) - 5)), 0)[1]],
                   np.minimum(1, border_delta_conf / np.expand_dims(np.array(critical_values[m]) + 1, -1))[:,
                          torch.min(torch.abs(
                              torch.tensor(
                                  np.maximum(0, border_delta_conf / np.expand_dims(np.array(critical_values[m]) + 1,
                                                                                   -1) - 1).mean(
                                      0) - 5)), 0)[1]],
                   np.minimum(1, border_eta / np.expand_dims(np.array(critical_values[m]) + 1, -1))[:,
                          torch.min(torch.abs(
                              torch.tensor(
                                  np.maximum(0, border_eta / np.expand_dims(np.array(critical_values[m]) + 1,
                                                                            -1) - 1).mean(
                                      0) - 5)), 0)[1]],
                   np.minimum(1, border_mirostat / np.expand_dims(np.array(critical_values[m]) + 1, -1))[:,
                          torch.min(torch.abs(
                              torch.tensor(
                                  np.maximum(0, border_mirostat / np.expand_dims(np.array(critical_values[m]) + 1,
                                                                                 -1) - 1).mean(
                                      0) - 5)), 0)[1]]
                   ])

        a = torch.tensor(np.maximum(0, border_top_k / np.expand_dims(np.array(critical_values[m]) + 1,
                                                                     -1) - 1).mean(
            0))[torch.min(torch.abs(torch.tensor(np.maximum(0, border_top_k / np.expand_dims(
            np.array(critical_values[m]) + 1, -1) - 1).mean(0) - 5)), 0)[1]]


        b = torch.tensor(np.maximum(0, border_top_p / np.expand_dims(np.array(critical_values[m]) + 1,
                                                                     -1) - 1).mean(
            0))[torch.min(torch.abs(
            torch.tensor(
                np.maximum(0, border_top_p / np.expand_dims(np.array(critical_values[m]) + 1,
                                                            -1) - 1).mean(
                    0) - 5)), 0)[1]]

        c = torch.tensor(np.maximum(0, border_delta_conf / np.expand_dims(np.array(critical_values[m]) + 1,
                                                                          -1) - 1).mean(
            0))[torch.min(torch.abs(
            torch.tensor(
                np.maximum(0, border_delta_conf / np.expand_dims(np.array(critical_values[m]) + 1,
                                                                 -1) - 1).mean(
                    0) - 5)), 0)[1]]

        d = torch.tensor(np.maximum(0, border_eta / np.expand_dims(np.array(critical_values[m]) + 1,
                                                                   -1) - 1).mean(
            0))[torch.min(torch.abs(
            torch.tensor(
                np.maximum(0, border_eta / np.expand_dims(np.array(critical_values[m]) + 1,
                                                          -1) - 1).mean(
                    0) - 5)), 0)[1]]
        e = torch.tensor(np.maximum(0, border_mirostat / np.expand_dims(np.array(critical_values[m]) + 1,
                                                                        -1) - 1).mean(
            0))[torch.min(torch.abs(
            torch.tensor(
                np.maximum(0, border_mirostat / np.expand_dims(np.array(critical_values[m]) + 1,
                                                               -1) - 1).mean(
                    0) - 5)), 0)[1]]

        print("###########check risk 5 parameters#############")
        ks = np.array(list(range(0, 1000, 1)) + list(range(1000, vocab_sizes[m], (vocab_sizes[m] - 1000) // 1000)))
        print("k")
        print(ks[torch.min(torch.abs(torch.tensor(np.maximum(0, border_top_k / np.expand_dims(
            np.array(critical_values[m]) + 1, -1) - 1).mean(0) - 5)), 0)[1]])

        ps = np.array([i / 2000 for i in range(2000)])
        print("p")
        print(ps[torch.min(torch.abs(torch.tensor(np.maximum(0, border_top_p / np.expand_dims(
            np.array(critical_values[m]) + 1, -1) - 1).mean(0) - 5)), 0)[1]])

        eps = np.array([i / 1e7 for i in range(1000)] + [1e-4 + i / 1e6 for i in range(0, 1000)] + [1e-3 + 1e-4 + i * (1 - 1e-3 - 1e-4) / 2000 for i
                                                                                     in range(0, 2000)])
        print("epsilon adaptive sampling")
        print(eps[torch.min(torch.abs(torch.tensor(np.maximum(0, border_delta_conf / np.expand_dims(
            np.array(critical_values[m]) + 1, -1) - 1).mean(0) - 5)), 0)[1]])

        es = np.array(
            [i / 1e7 for i in range(1000)] + [1e-4 + i / 1e6 for i in range(0, 1000)] + [i / 1000 for i in range(1000)])
        print("epsilon eta-sampling")
        print(es[torch.min(torch.abs(torch.tensor(np.maximum(0, border_eta / np.expand_dims(
            np.array(critical_values[m]) + 1, -1) - 1).mean(0) - 5)), 0)[1]])

        ms = np.array([10 / 4000 * i for i in range(4000)])
        print("tau")
        print(ms[torch.min(torch.abs(torch.tensor(np.maximum(0, border_mirostat / np.expand_dims(
            np.array(critical_values[m]) + 1, -1) - 1).mean(0) - 5)), 0)[1]])

        print("############check risk 5 stats############")
        names = ["top-k", "top-p", "adaptive", "eta", "mirostat"]
        parameters = [ks, ps, eps, es, ms]
        for n in range(0, 5):
            print(names[n])
            print("risk mean, risk standard error, recall mean, recall standard error")
            print(f"{parameters[n]} & {risks[n].mean():10.3f} ({risks[n].std()/np.sqrt(len(risks[n])):10.3f}) & {recalls[n].mean():10.3f} ({recalls[n].std()/np.sqrt(len(recalls[n])):10.3f}"
                                )


        ################

        labels = []
        recalls = []
        risks = []

        risks.extend([
            np.maximum(0, border_top_k / np.expand_dims(np.array(critical_values[m]) + 1,
                                                        -1) - 1)[:,
            torch.min(torch.abs(
                torch.tensor(
                    np.maximum(0, border_top_k / np.expand_dims(np.array(critical_values[m]) + 1,
                                                                -1) - 1).mean(
                        0) - 1)), 0)[1]],
            np.maximum(0, border_top_p / np.expand_dims(np.array(critical_values[m]) + 1,
                                                        -1) - 1)[:,
            torch.min(torch.abs(
                torch.tensor(
                    np.maximum(0, border_top_p / np.expand_dims(np.array(critical_values[m]) + 1,
                                                                -1) - 1).mean(
                        0) - 1)), 0)[1]],
            np.maximum(0, border_delta_conf / np.expand_dims(np.array(critical_values[m]) + 1,
                                                             -1) - 1)[:,
            torch.min(torch.abs(
                torch.tensor(
                    np.maximum(0, border_delta_conf / np.expand_dims(np.array(critical_values[m]) + 1,
                                                                     -1) - 1).mean(
                        0) - 1)), 0)[1]],
            np.maximum(0, border_eta / np.expand_dims(np.array(critical_values[m]) + 1,
                                                      -1) - 1)[:,
            torch.min(torch.abs(
                torch.tensor(
                    np.maximum(0, border_eta / np.expand_dims(np.array(critical_values[m]) + 1,
                                                              -1) - 1).mean(
                        0) - 1)), 0)[1]],
            np.maximum(0, border_mirostat / np.expand_dims(np.array(critical_values[m]) + 1,
                                                           -1) - 1)[:,
            torch.min(torch.abs(
                torch.tensor(
                    np.maximum(0, border_mirostat / np.expand_dims(np.array(critical_values[m]) + 1,
                                                                   -1) - 1).mean(
                        0) - 1)), 0)[1]],
        ])
        recalls.extend([
            np.minimum(1, border_top_k / np.expand_dims(np.array(critical_values[m]) + 1, -1))[:,
            torch.min(torch.abs(torch.tensor(np.maximum(0, border_top_k / np.expand_dims(
                np.array(critical_values[m]) + 1, -1) - 1).mean(0) - 1)), 0)[1]],
            np.minimum(1, border_top_p / np.expand_dims(np.array(critical_values[m]) + 1, -1))[:,
            torch.min(torch.abs(
                torch.tensor(
                    np.maximum(0, border_top_p / np.expand_dims(np.array(critical_values[m]) + 1,
                                                                -1) - 1).mean(
                        0) - 1)), 0)[1]],
            np.minimum(1, border_delta_conf / np.expand_dims(np.array(critical_values[m]) + 1, -1))[:,
            torch.min(torch.abs(
                torch.tensor(
                    np.maximum(0, border_delta_conf / np.expand_dims(np.array(critical_values[m]) + 1,
                                                                     -1) - 1).mean(
                        0) - 1)), 0)[1]],
            np.minimum(1, border_eta / np.expand_dims(np.array(critical_values[m]) + 1, -1))[:,
            torch.min(torch.abs(
                torch.tensor(
                    np.maximum(0, border_eta / np.expand_dims(np.array(critical_values[m]) + 1,
                                                              -1) - 1).mean(
                        0) - 1)), 0)[1]],
            np.minimum(1, border_mirostat / np.expand_dims(np.array(critical_values[m]) + 1, -1))[:,
            torch.min(torch.abs(
                torch.tensor(
                    np.maximum(0, border_mirostat / np.expand_dims(np.array(critical_values[m]) + 1,
                                                                   -1) - 1).mean(
                        0) - 1)), 0)[1]]
        ])

        a = torch.tensor(np.maximum(0, border_top_k / np.expand_dims(np.array(critical_values[m]) + 1,
                                                                     -1) - 1).mean(
            0))[torch.min(torch.abs(torch.tensor(np.maximum(0, border_top_k / np.expand_dims(
            np.array(critical_values[m]) + 1, -1) - 1).mean(0) - 1)), 0)[1]]

        b = torch.tensor(np.maximum(0, border_top_p / np.expand_dims(np.array(critical_values[m]) + 1,
                                                                     -1) - 1).mean(
            0))[torch.min(torch.abs(
            torch.tensor(
                np.maximum(0, border_top_p / np.expand_dims(np.array(critical_values[m]) + 1,
                                                            -1) - 1).mean(
                    0) - 1)), 0)[1]]

        c = torch.tensor(np.maximum(0, border_delta_conf / np.expand_dims(np.array(critical_values[m]) + 1,
                                                                          -1) - 1).mean(
            0))[torch.min(torch.abs(
            torch.tensor(
                np.maximum(0, border_delta_conf / np.expand_dims(np.array(critical_values[m]) + 1,
                                                                 -1) - 1).mean(
                    0) - 1)), 0)[1]]

        d = torch.tensor(np.maximum(0, border_eta / np.expand_dims(np.array(critical_values[m]) + 1,
                                                                   -1) - 1).mean(
            0))[torch.min(torch.abs(
            torch.tensor(
                np.maximum(0, border_eta / np.expand_dims(np.array(critical_values[m]) + 1,
                                                          -1) - 1).mean(
                    0) - 1)), 0)[1]]
        e = torch.tensor(np.maximum(0, border_mirostat / np.expand_dims(np.array(critical_values[m]) + 1,
                                                                        -1) - 1).mean(
            0))[torch.min(torch.abs(
            torch.tensor(
                np.maximum(0, border_mirostat / np.expand_dims(np.array(critical_values[m]) + 1,
                                                               -1) - 1).mean(
                    0) - 1)), 0)[1]]

        print("###########check risk 1 parameters##########")
        ks = np.array(list(range(0, 1000, 1)) + list(range(1000, vocab_sizes[m], (vocab_sizes[m] - 1000) // 1000)))
        print("k")
        print(ks[torch.min(torch.abs(torch.tensor(np.maximum(0, border_top_k / np.expand_dims(
            np.array(critical_values[m]) + 1, -1) - 1).mean(0) - 1)), 0)[1]])

        ps = np.array([i / 2000 for i in range(2000)])
        print("p")
        print(ps[torch.min(torch.abs(torch.tensor(np.maximum(0, border_top_p / np.expand_dims(
            np.array(critical_values[m]) + 1, -1) - 1).mean(0) - 1)), 0)[1]])

        eps = np.array([i / 1e7 for i in range(1000)] + [1e-4 + i / 1e6 for i in range(0, 1000)] + [
            1e-3 + 1e-4 + i * (1 - 1e-3 - 1e-4) / 2000 for i
            in range(0, 2000)])
        print("epsilon adpative sampling")
        print(eps[torch.min(torch.abs(torch.tensor(np.maximum(0, border_delta_conf / np.expand_dims(
            np.array(critical_values[m]) + 1, -1) - 1).mean(0) - 1)), 0)[1]])

        es = np.array(
            [i / 1e7 for i in range(1000)] + [1e-4 + i / 1e6 for i in range(0, 1000)] + [i / 1000 for i in range(1000)])
        print("epsilon eta-sampling")
        print(es[torch.min(torch.abs(torch.tensor(np.maximum(0, border_eta / np.expand_dims(
            np.array(critical_values[m]) + 1, -1) - 1).mean(0) - 1)), 0)[1]])

        ms = np.array([10 / 4000 * i for i in range(4000)])
        print("tau")
        print(ms[torch.min(torch.abs(torch.tensor(np.maximum(0, border_mirostat / np.expand_dims(
            np.array(critical_values[m]) + 1, -1) - 1).mean(0) - 1)), 0)[1]])

        print("############check risk 1 stats############")
        names = ["top-k", "top-p", "adaptive", "eta", "mirostat"]
        parameters = [ks, ps, eps, es, ms]
        for n in range(0, 5):
            print(names[n])
            print("risk mean, risk standard error, recall mean, recall standard error")
            print(f"{parameters[n]} & {risks[n].mean():10.3f} ({risks[n].std()/np.sqrt(len(risks[n])):10.3f}) & {recalls[n].mean():10.3f} ({recalls[n].std()/np.sqrt(len(recalls[n])):10.3f}"
                                )



