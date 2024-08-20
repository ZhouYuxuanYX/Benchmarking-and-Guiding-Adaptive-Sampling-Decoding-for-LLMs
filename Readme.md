# An Evluation Benchmark for Truncation Sampling Decoding Strategies of LLMs

# Data preparation
### Step 1: Build a context-preserving prefix tree from any existing dataset
```
python collect_prefix_tree.py
```
Main features:
  - **Efficient Multi-Processing**. For example, given 2 sockets (24 cores per socket) of AMD EPYC 7402 processor, the script can process roughly 120k articles in 1 hour to build a context-preserving tree with sentence-level context. Then the total 6,458,670 articles of English Wikipedia dataset will be transformed into a context-preserving tree in around 2 days.
