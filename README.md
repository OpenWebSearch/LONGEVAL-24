# Team OpenWebSearch (OWS) at LongEval 2024

The repository for team OWS at [LongEval 2024](https://clef-longeval.github.io/) at [CLEF 2024](https://clef2024.imag.fr/)

## Idea 1: Features from WOWS

- Feature-based Learning to Rank (Gijs)
  - Use the WOWS submissions to create around 50 features
    - Query Features: Query Intent + Query Performance Prediction
    - Document Features: Web Page Genre + Corpus Grap + Readibility 
    - Query-Document Features: BM25 + MonoT5 + ColBERT
    - ToDo: Look for more interesting components
  - Apply LambdaMART


## Idea 2: Archive Lookup / Learning from History / History Repeats Itself / Exploit Overlap / Zipfs Law

I already know which topics will be submitted. I already know "similar" documents

Archive lookup: I look up what was good a few months ago, now we try to transfer this via two strategies and combinations thereof: (1) query reformulation, (2) document reformulation.

- Query Reformulation (Daria):
  - Idea:
    - Queries overlap over the different time slots
    - I.e. for some query, we know which documents were clicked a few months ago
    - We insert the clicked documents into the current corpus and reformulate the query with RM3 until the known relevant documents from a few months ago are at the top positions
      - Use Explain Like I am BM25 for better reformulating
    - Remove the old doc ids from the ranking
    - Combine it with the Corpus Graph idea
      - What to do if a query is new

- Document Reformulation/Oracle Indexing/Index Partitioning (Maik):
  - Idea:
    - Reformulate all documents so that they are only retrieved for the queries for which they are relevant and not be retrieved for queries for which they are not relevant
    - Because queries are overlapping
    - Learn optimal sequence to sequence translation of documents to ideal documents on the training data
    - Use this on the test data that is X months in the future and uses the same topics (information needs) but (slightly) updated documents
  - What do we need:
    - Sequence to sequence training dataset: Document -> perfect document
    - How do we construct this?
      - document => all queries for which the document is relevant (DeepCT / splade training objective)
      - Reverse bipartite graph between documents and its relevant queries
      - Include corpus graph idea?
      - Goal: ndcg above 0.8
  - Technnologies to look into:
    - Naive Bayes: P(query|document text) or P(partition|document text)
    - Transformer: sequence to sequence
    - Splade
    - RM3 reversed
