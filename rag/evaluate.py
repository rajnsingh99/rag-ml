"""A small retrieval test set and hit-rate / MRR metrics.

Each question comes with one or more phrases that only appear in the passage
that answers it (checked against the book text). A retrieved chunk counts as
a hit if it contains one of those phrases.
"""

import re

from .store import VectorStore

EVAL_SET = [
    {"question": "How do you encode a categorical feature like workclass so a linear model can use it?",
     "phrases": ["one-hot-encoding", "dummy variables"]},
    {"question": "How does k-fold cross-validation split the data?",
     "phrases": ["called folds"]},
    {"question": "What is the difference between L1 and L2 regularization in Lasso and Ridge?",
     "phrases": ["l1 regularization"]},
    {"question": "What does the parameter C control in logistic regression and linear SVMs?",
     "phrases": ["strength of the regularization"]},
    {"question": "What are core samples in DBSCAN?",
     "phrases": ["core samples"]},
    {"question": "Why should scaling be done inside a pipeline when using cross-validation?",
     "phrases": ["information leakage", "leak"]},
    {"question": "What is nested cross-validation?",
     "phrases": ["nested cross-validation"]},
    {"question": "How does tf-idf rescale word counts?",
     "phrases": ["tf–idf", "tf-idf"]},
    {"question": "What is the kernel trick in support vector machines?",
     "phrases": ["kernel trick"]},
    {"question": "Why is accuracy a bad metric for imbalanced datasets?",
     "phrases": ["imbalanced"]},
    {"question": "What does the area under the ROC curve measure?",
     "phrases": ["area under the curve"]},
    {"question": "How can t-SNE be used to visualize the digits dataset?",
     "phrases": ["t-sne"]},
    {"question": "What is the silhouette coefficient used for in clustering?",
     "phrases": ["silhouette"]},
    {"question": "What is pre-pruning in decision trees?",
     "phrases": ["pre-pruning"]},
    {"question": "What does the learning_rate parameter do in gradient boosting?",
     "phrases": ["learning_rate"]},
    {"question": "What are n-grams in text processing?",
     "phrases": ["n-grams"]},
]


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).lower()


def is_relevant(chunk_text: str, phrases: list[str]) -> bool:
    t = _norm(chunk_text)
    return any(_norm(p) in t for p in phrases)


def evaluate_retrieval(store: VectorStore, ks=(3, 5, 7), eval_set=EVAL_SET) -> dict:
    """Hit rate @k (did any of the top-k chunks contain the answer?) and MRR (how high was the first hit?)."""
    max_k = max(ks)
    hits_at = {k: 0 for k in ks}
    reciprocal_ranks = []
    per_question = []
    for item in eval_set:
        results = store.search(item["question"], max_k)
        first = next((h.rank for h in results if is_relevant(h.text, item["phrases"])), None)
        for k in ks:
            hits_at[k] += int(first is not None and first <= k)
        reciprocal_ranks.append(1 / first if first else 0.0)
        per_question.append({"question": item["question"], "first_relevant_rank": first})
    n = len(eval_set)
    return {
        "hit_rate": {k: round(v / n, 3) for k, v in hits_at.items()},
        "mrr": round(sum(reciprocal_ranks) / n, 3),
        "per_question": per_question,
    }
