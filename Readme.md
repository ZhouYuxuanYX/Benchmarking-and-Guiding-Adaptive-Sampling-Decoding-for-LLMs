# An Evaluation Benchmark and User Guidline of Adaptive Sampling Decoding for LLMs in Open-Ended Text Generation

# Data preparation
Step 1: Build a context-preserving prefix tree from any existing dataset
```
python collect_prefix_tree.py
```
Step 2: Sort the context-preserving tree w.r.t. number of leaves of the current subtree
```
python sort_and_visualize_prefix_tree.py
```

Main features:
  - **Efficient Multi-Processing**. For example, given 2 sockets (24 cores per socket) of AMD EPYC 7402 processor, the script can process roughly 140k articles in 1 hour to build a context-preserving tree with sentence-level context. Then the total 6,458,670 articles of English Wikipedia dataset will be transformed into a context-preserving tree in around 2 days.
  - **Filtering**. For example, to avoid invalid words or rare proper names, we exclude the sentences containing such words by checking their presence in the WORD LIST dataset, which is available on the [word-list dataset homepage](https://web.archive.org/web/20131118073324/https://www.infochimps.com/datasets/word-list-350000-simple-english-words-excel-readable). It contains a total amount of 354986 words and explicitly excludes proper names and compound words. Section titles are also excluded, because they are often incomplete sentences with poor contextual information.
  - **Hashing-based Retrieval** The extracted prefix tree is implemented as a nested dictionary and saved in a single JSON file. Since each lookup at any depth has constant complexity, the retrieval from our dataset is highly efficient. Moreover, the dictionary is easily extendable if extra data are needed for a more accurate estimation of the full data support.

Illustration of a context-preserving prefix tree:
<p align="center">
   <img src="tree.jpg" alt="drawing" width="450"/>
</p>
<p align="center">
   <b>Figure 1:</b> A fragment of the EnWiki CP-Trie. For brevity, only two child nodes are shown at each depth. The number at the left side of the slash symbol refers to the branching factor at the current node, and the number at the right side refers to the total number of leaves of the sub-tree with the current node as the root node.
</p>

# Evaluate truncation sampling methods
Step 1: Estimate the Optimal Truncations 
```
python estimate_optimal_truncation.py
```
This computes the estimation of the optimal truncation positions given different prefixes. Briefly speaking, truncation sampling is mainly based on the fact that most of the out-of-distribution tokens reside in the tail of the predicted distribution by LLMs. Therefore, the quality of the generated text is expected to be improved by excluding a proper fraction of tail from the sampling procedure. For more details such as the rationale of our estimation method, please refer to the paper.

# To be updated!
