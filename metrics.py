from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_score, recall_score
from rouge_score import rouge_scorer
from time import time
import numpy as np

smoothie = lambda x: 1  # Replace this line with a custom smoothing function if needed

# Precision at K
def precision_at_k(relevant, retrieved, k):
    retrieved_k = retrieved[:k]
    relevant_set = set(relevant)
    return len([x for x in retrieved_k if x in relevant_set]) / k

# Recall at K
def recall_at_k(relevant, retrieved, k):
    retrieved_k = retrieved[:k]
    relevant_set = set(relevant)
    return len([x for x in retrieved_k if x in relevant_set]) / len(relevant_set)

# Mean Average Precision
def mean_average_precision(relevant, retrieved):
    score = 0.0
    hits = 0
    for i, r in enumerate(retrieved):
        if r in relevant:
            hits += 1
            score += hits / (i + 1)
    return score / len(relevant) if relevant else 0.0

# Cosine Similarity
def cosine_sim(vec1, vec2):
    return cosine_similarity([vec1], [vec2])[0][0]

# BLEU Score (without nltk)
def bleu_score(reference, hypothesis):
    def tokenize(text):
        return text.lower().split()  # Simple whitespace tokenization
    
    ref_tokens = tokenize(reference)
    hyp_tokens = tokenize(hypothesis)
    
    # Calculate BLEU score
    overlap = sum(1 for token in hyp_tokens if token in ref_tokens)
    if len(hyp_tokens) == 0:
        return 0.0
    return overlap / len(hyp_tokens)  # Simplified BLEU calculation

# ROUGE Score
def rouge_score(reference, hypothesis):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    score = scorer.score(reference, hypothesis)
    return score['rougeL'].fmeasure

# Response Time
def response_time(fn, *args, **kwargs):
    start = time()
    result = fn(*args, **kwargs)
    latency = time() - start
    return result, latency

# Placeholder for human judgment
def human_eval(answer, reference):
    # In reality this would be replaced with a manual review or GUI input
    print(f"Answer: {answer}\nReference: {reference}")
    return input("Is this answer correct? (yes/no): ").strip().lower() == "yes"
