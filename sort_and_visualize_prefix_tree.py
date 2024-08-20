import torch
torch.multiprocessing.set_start_method("forkserver", force=True)
torch.multiprocessing.set_sharing_strategy('file_system')
from collections.abc import Mapping
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from functools import partial
import json
import os
import matplotlib.collections as mcoll
import collections
import itertools
import time

##### calculate the frequencies of sub trajectories for the current node
def get_freq(example, depth):
    freq_dict = {}
    total_counts = 0
    if depth > -1:
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

            freq_dict[d_] = len(example[1][d][f"leaves-{depth + 1}"])
            total_counts += len(example[1][d][f"leaves-{depth + 1}"])
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

            freq_dict[d_] = len(example[d][f"leaves-{depth + 1}"])
            total_counts += len(example[d][f"leaves-{depth + 1}"])
    return freq_dict

def dict_count(prod, c=0):
    for mykey in prod:
        if mykey == "leveas-0":
            c += len(prod[mykey])
    return c


### try this to solve the hanging problem when using executor for topp varying p
"""
https://stackoverflow.com/questions/74633896/processpoolexecutor-using-map-hang-on-large-load
"""
def colorline(
    x, y, z=None, cmap=plt.get_cmap('seismic'), norm=plt.Normalize(vmin=0, vmax=1),
        linewidth=3, alpha=1.0, label=None):
    """
    http://nbviewer.ipython.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb
    http://matplotlib.org/examples/pylab_examples/multicolored_line.html
    Plot a colored line with coordinates x and y
    Optionally specify colors in the array z
    Optionally specify a colormap, a norm function and a line width
    """

    # Default colors equally spaced on [-1,1]:
    # color range is -1,1 not 0,1
    # the norm here is not for z
    if z is None:
        z = np.linspace(-0.8, 1, len(x))

    # Special
    # case if a single number:
    if not hasattr(z, "__iter__"):  # to check for numerical input -- this is a hack
        z = np.array([z])

    z = np.asarray(z)

    segments = make_segments(x, y)
    if label is None:
        lc = mcoll.LineCollection(segments, array=z, cmap=cmap, norm=norm,
                              linewidth=linewidth, alpha=alpha)
    else:
        lc = mcoll.LineCollection(segments, array=z, cmap=cmap, norm=norm,
                                  linewidth=linewidth, alpha=alpha, label=label)
    ax = plt.gca()
    ax.add_collection(lc)

    return lc

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

# this will cause dead lock
# mp.set_start_method('spawn')
def SeLU(x, ranges, ts):
    # print("before")
    # print(x)
    x = x + ts[0]*torch.relu(ranges[0] - x) + ts[1]*torch.relu(x - ranges[1]) \
         + ts[2]*torch.relu(ranges[2] - x) **2 + ts[3]*torch.relu(x - ranges[3])**2
    # print("after")

    # print(x)
    # exit()
    return x
# # vit
# ranges_ = np.load(f'ranges_vit.npz')
# print(len(ranges_["arr_0"]))
# ts_ = np.load(f'ts_vit.npz')


# colors_multimax = plt.cm.cool(np.linspace(0, 1, 12))


# exit()
def make_segments(x, y):

    """
    Create list of line segments from x and y coordinates, in the correct format
    for LineCollection: an array of the form numlines x (points per line) x 2 (x
    and y) array
    """

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    return segments

# this considers repeated sentences as well!!!
# sorted_trajects = dict(sorted(trajects.items(), key=lambda item: sum(item[1].get("leaves-0", {-1:"leaves-" not in list(item[1].keys())[0]}).values()), reverse=True))
# print(list(list(sorted_trajects.values())[1].keys())[0])
# values of nested dictionary are again dictionaries
def sorting(value_dict_item, level):
    if f"leaves-{level}" in value_dict_item[1]:
        # print(value_dict_item)
        # exit()
        return len(value_dict_item[1][f"leaves-{level}"])

    # never happens for leaves-0
    else:
        # this is leaves-* sub dictionary
        if "leaves-" not in list(value_dict_item[1].keys())[0]:
            # print(list(value_dict_item[1].keys())[0])
            # print
            # print(level)
            # print("if")
            # exit()
            return -1
        else:
            # print("else")
            # exit()
            return 0

def sum_key1(vs):
    # print(vs)
    # print([v.values() for v in vs])
    return sum([list(v.values())[0] for v in vs])


def sum_key2(vs):
    # print(vs)
    return sum(vs)


def hierarchy_pos(G, root=None, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5):
    '''
    From Joel's answer at https://stackoverflow.com/a/29597209/2966723.
    Licensed under Creative Commons Attribution-Share Alike

    If the graph is a tree this will return the positions to plot this in a
    hierarchical layout.

    G: the graph (must be a tree)

    root: the root node of current branch
    - if the tree is directed and this is not given,
      the root will be found and used
    - if the tree is directed and this is given, then
      the positions will be just for the descendants of this node.
    - if the tree is undirected and not given,
      then a random choice will be used.

    width: horizontal space allocated for this branch - avoids overlap with other branches

    vert_gap: gap between levels of hierarchy

    vert_loc: vertical location of root

    xcenter: horizontal location of root
    '''
    if not nx.is_tree(G):
        raise TypeError('cannot use hierarchy_pos on a graph that is not a tree')

    if root is None:
        if isinstance(G, nx.DiGraph):
            # this does not change the order
            root = next(iter(nx.topological_sort(G)))  # allows back compatibility with nx version 1.11
            # pass
        else:
            exit()
            root = random.choice(list(G.nodes))

    def _hierarchy_pos(G, root, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5, pos=None, parent=None):
        '''
        see hierarchy_pos docstring for most arguments

        pos: a dict saying where all nodes go if they have been assigned
        parent: parent of this branch. - only affects it if non-directed

        '''

        if pos is None:
            pos = {root: (xcenter, vert_loc)}
        else:
            pos[root] = (xcenter, vert_loc)
        children = list(G.neighbors(root))
        if not isinstance(G, nx.DiGraph) and parent is not None:
            children.remove(parent)
        if len(children) != 0:
            dx = width / len(children)
            nextx = xcenter - width / 2 - dx / 2
            for child in children:
                nextx += dx
                pos = _hierarchy_pos(G, child, width=dx, vert_gap=vert_gap,
                                     vert_loc=vert_loc - vert_gap, xcenter=nextx,
                                     pos=pos, parent=root)
        return pos

    return _hierarchy_pos(G, root, width, vert_gap, vert_loc, xcenter)


def hierarchy_pos_horizontal(G, root=None, width=1., vert_gap=0.2, vert_loc=1, xcenter=0):
    '''
    From Joel's answer at https://stackoverflow.com/a/29597209/2966723.
    Licensed under Creative Commons Attribution-Share Alike

    If the graph is a tree this will return the positions to plot this in a
    hierarchical layout.

    G: the graph (must be a tree)

    root: the root node of current branch
    - if the tree is directed and this is not given,
      the root will be found and used
    - if the tree is directed and this is given, then
      the positions will be just for the descendants of this node.
    - if the tree is undirected and not given,
      then a random choice will be used.

    width: horizontal space allocated for this branch - avoids overlap with other branches

    vert_gap: gap between levels of hierarchy

    vert_loc: vertical location of root

    xcenter: horizontal location of root
    '''
    if not nx.is_tree(G):
        raise TypeError('cannot use hierarchy_pos on a graph that is not a tree')

    if root is None:
        if isinstance(G, nx.DiGraph):
            root = next(iter(nx.topological_sort(G)))  # allows back compatibility with nx version 1.11
        else:
            root = random.choice(list(G.nodes))

    def _hierarchy_pos_horizontal(G, root, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5, pos=None, parent=None):
        '''
        see hierarchy_pos docstring for most arguments

        pos: a dict saying where all nodes go if they have been assigned
        parent: parent of this branch. - only affects it if non-directed

        '''

        if pos is None:
            pos = {root: (xcenter, vert_loc)}
        else:
            pos[root] = (xcenter, vert_loc)
        children = list(G.neighbors(root))
        if not isinstance(G, nx.DiGraph) and parent is not None:
            children.remove(parent)
        if len(children) != 0:
            dx = width / len(children)
            next_vert = vert_loc - width / 2 - dx / 2
            for child in children:
                next_vert += dx
                pos = _hierarchy_pos_horizontal(G, child, width=dx, vert_gap=vert_gap,
                                                vert_loc=next_vert, xcenter=xcenter + vert_gap,
                                                pos=pos, parent=root)
        return pos

    return _hierarchy_pos_horizontal(G, root, width, vert_gap, vert_loc, xcenter)

'''
Pool/ProcessPoolExecutor both must serialize everything before sending them to the workers. 
Serializing (also sometimes called pickling) actually is the process in which the name of a function is saved, to only be imported again once Pool wants to have access to it. 
For this process to work, the function has to be defined at the top level since nested functions are not importable by the child which is the reason for the following error to show up
'''

def checking(inp):
    # for free account there are some limits using public server, try reducing number of works
    # but using local server even no multiprocessing is possible, thread error
    # with lang.LanguageToolPublicAPI('en-US') as tool:
        # current state not use grammarcheck
        # return grammarChecker(inp[4:], tool), inp
    return None, inp

def identity(p):
    return p

# this solves the problem!!!!!!!!! key to solve multiprocessing task hanging after first model_run problem
"""insert an if __name__ == '__main__': guard in the main module to avoid creating subprocesses recursively."""
if __name__ == '__main__':

    os.environ["LTP_PATH"] = "/p/project/hai_recmax"

    colors = plt.cm.cool(np.linspace(0, 1, 5))

    ############# load and sort the trajects ###############
    with open('trajects_test.json', 'r') as fp:
        trajects = json.load(fp)

    # using the len function, for ending word and leaves-* keys (sub), they both don't have leaves-*+1 in their value dictionary
    # and using len function will result in the same len=1
    sort0 = partial(sorting, level=0)
    # sort0 is sorting the first level tr, where no leaves are included
    # leaves-0 is actually sorted at sort1, which is the second level
    sorted_trajects = dict(sorted(trajects.items(), key=sort0, reverse=True))


    # hard-coded for simplicity
    for tr in sorted_trajects:
        # "leaves-depth" is used to store the list of all leaves
        if "leaves-0" not in trajects[tr]:
            pass
        else:
            # here a little bit tricky to count the repeated times of a sentence when it goes to the final token
            # for the final token, it will not have "leaves-" key
            # so use -1 as a temporay placeholder
            # "leaves-" not in item[1] will assign 0 to "leaves-*" and 1 to ending word, so that "leaves-*" key is put in the last position
            sort1 = partial(sorting, level=1)
            sorted_trajects[tr] = dict(sorted(sorted_trajects[tr].items(), key=sort1, reverse=True))
            for tr1 in sorted_trajects[tr]:
                if tr1 == "leaves-0" or "leaves-1" not in trajects[tr][tr1]:
                    pass
                else:
                    # sorted_trajects[tr][tr1] = dict(sorted(sorted_trajects[tr][tr1].items(), key=lambda item: sum(item[1].get("leaves-2", {-1:0}).values()), reverse=True))
                    sort2 = partial(sorting, level=2)
                    sorted_trajects[tr][tr1] = dict(sorted(sorted_trajects[tr][tr1].items(), key=sort2, reverse=True))

                    for tr2 in sorted_trajects[tr][tr1]:
                        # leaves-1 key or end of a paragraph
                        if tr2 == "leaves-1" or "leaves-2" not in trajects[tr][tr1][tr2]:
                            pass
                        else:
                            sort3 = partial(sorting, level=3)
                            sorted_trajects[tr][tr1][tr2] = dict(
                                sorted(sorted_trajects[tr][tr1][tr2].items(), key=sort3, reverse=True))

                            for tr3 in sorted_trajects[tr][tr1][tr2]:
                                # leaves-1 key or end of a paragraph
                                if tr3 == "leaves-2" or "leaves-3" not in trajects[tr][tr1][tr2][tr3]:
                                    pass
                                else:
                                    sort4 = partial(sorting, level=4)
                                    sorted_trajects[tr][tr1][tr2][tr3] = dict(
                                        sorted(sorted_trajects[tr][tr1][tr2][tr3].items(), key=sort4,
                                               reverse=True))

                                    for tr4 in sorted_trajects[tr][tr1][tr2][tr3]:
                                        # leaves-1 key or end of a paragraph
                                        if tr4 == "leaves-3" or "leaves-4" not in trajects[tr][tr1][tr2][tr3][tr4]:
                                            pass
                                        else:
                                            sort5 = partial(sorting, level=5)
                                            sorted_trajects[tr][tr1][tr2][tr3][tr4] = dict(
                                                sorted(sorted_trajects[tr][tr1][tr2][tr3][tr4].items(), key=sort5,
                                                       reverse=True))

                                            for tr5 in sorted_trajects[tr][tr1][tr2][tr3][tr4]:
                                                # leaves-1 key or end of a paragraph
                                                if tr5 == "leaves-4" or "leaves-5" not in \
                                                        trajects[tr][tr1][tr2][tr3][tr4][
                                                            tr5]:
                                                    pass
                                                else:
                                                    # since "leaves-*" key will not have "leaves-*+1" as its value, similar to the sentence ending words,
                                                    # it will be assigned -1
                                                    sort6 = partial(sorting, level=6)
                                                    sorted_trajects[tr][tr1][tr2][tr3][tr4][tr5] = dict(
                                                        sorted(sorted_trajects[tr][tr1][tr2][tr3][tr4][tr5].items(),
                                                               key=sort6,
                                                               reverse=True))

                                                    for tr6 in sorted_trajects[tr][tr1][tr2][tr3][tr4][tr5]:
                                                        # leaves-1 key or end of a paragraph
                                                        if tr6 == "leaves-5" or "leaves-6" not in \
                                                                trajects[tr][tr1][tr2][tr3][tr4][tr5][tr6]:
                                                            pass
                                                        else:
                                                            sort7 = partial(sorting, level=7)
                                                            sorted_trajects[tr][tr1][tr2][tr3][tr4][tr5][tr6] = dict(
                                                                sorted(sorted_trajects[tr][tr1][tr2][tr3][tr4][tr5][
                                                                           tr6].items(),
                                                                       key=sort7,
                                                                       reverse=True))

    with open('sorted_trajects_test.json', 'w') as fp:
        json.dump(sorted_trajects, fp)

    graph_data = sorted_trajects

    # Empty directed graph
    G = nx.DiGraph()

    # print(list(graph_data.items())[0:1])
    # Iterate through the layers
    q = list(graph_data.items())[0:1]
    start = q[0][0]
    print("####### start #######")
    print(start)
    print(graph_data[start])

    root = None
    level = 0
    # here q is "In" for visualization
    while q:
        # pop last in first out, that's why it's inversed order
        # use pop(0) instead
        # v, d = q.pop()
        v, d = q.pop(0)
        # print(v)
        # print(type(v))
        # print(v)
        # exit()
        # if root is None:
        #     root = v

        counts = 0

        # start from index 1 to remove leaves
        for nv, nd in d.items():
            # add the second condition to avoid ending word wich doesn't contain "leaves-"
            if "leaves-" not in nv:
                counts += 1
                if counts < 5:

                    # in the first level, there's no leaves
                    # check if the key string contains "leaves-", if not, then it should be skipped, because we are drawing real trajectories
                    if "leaves-" in list(nd.keys())[0]:
                        pass
                    # consider paragraph ending word, no values anymore
                    # this is just leaves-*+1 sub dictionary
                    elif type(list(list(nd.values())[-1].values())[0]) != dict:
                        # two numbers: first-total number of branches at the whole subtree, second-number of branches at the subnode only
                        G.add_edge(
                            v + f"- {len(list(d.keys())) - 1}/{len(d[f'leaves-{len(list(d.keys())[0].split()) - 2}'])}",
                            nv + f"- {len(list(nd.keys())) - 1}/{len(nd[f'leaves-{len(list(nd.keys())[0].split()) - 2}'])}")
                        if isinstance(nd, Mapping):
                            q.append((nv, nd))
                        # print(list(nd.values())[-1])
                    # first level doesn't have leaves-
                    else:
                        pass
                else:
                    break
            else:
                pass
        # not update every while loop, because each while loop just pop one q out
        # level += 1
    # the order is correct
    # print(G.nodes)
    # t()

    plt.figure(figsize=(40, 80))
    np.random.seed(8)

    # graphviz throws error
    # pos = graphviz_layout(G, prog="dot")

    pos = hierarchy_pos_horizontal(G, root)
    nx.draw(G, pos, with_labels=True)
    plt.tight_layout()
    # the plot code is somehow in invfersed order, wasted a whole day to solve, so use the following trick
    plt.gca().invert_yaxis()
    plt.savefig("Graph.png", format="PNG")
    plt.close()