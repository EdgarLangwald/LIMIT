## Overview

The goal of this project is to understand the LIMIT paper in action. Currently I want to find out whether an embedding model can even embed a list of attributes meaningfully, i.e. if models fail due to geometric constraints or due to bad experiment design.

## Current state of project

Hypothesis: Such long lists of items (loi) are unfeasable to embed for any model.
    Question: Do document embeddings ie loi cluster more then text chunks from realistic pdf's?
    Question: Is information about every item preserved in lois embedding? I.e. can every item be retrieved?
Result: Clustering increases and recall steadily drops with increasing loi length (Recall@1 from 100% to 20% at n=100 people and m=1, ..., 18 items). Models can't embed a loi in a way that preserves each items meaning reliably. But they are still able to with m = 3.



## Other

URL to LIMIT repo: https://github.com/google-deepmind/limit
This Claude Code session is being run from windows powershell
**IMPORTANT** Run all scripts from the venv of the parent folder (e.g. .venv or Machine_Learning_venv). Also pip install into there

## Paper description for quick context

**"On the Theoretical Limitations of Embedding-Based Retrieval"** (ICLR 2026)
Weller et al., Google DeepMind & Johns Hopkins

**Core claim:** Single-vector embedding models have a hard geometric ceiling. For a given embedding dimension *d*, there exist top-k subsets of documents that *no* query vector can retrieve — not because of bad training, but because the geometry doesn't allow it. The number of representable top-k combinations is bounded by *d*.

**Theoretical basis:** Connects results from linear algebra / high-dimensional geometry (Papadimitriou & Sipser) to dense retrieval. Lower-bounds the embedding dimension needed to represent a given set of query-document relevance combinations.

**Empirical validation:** They bypass training entirely by *directly optimizing free parameter embeddings* on the test set.

**The LIMIT dataset:** A deliberately simple synthetic-to-natural dataset. Documents are short person profiles ("Jon likes Apples and Rabbits."), queries are attribute lookups ("Who likes Apples?").