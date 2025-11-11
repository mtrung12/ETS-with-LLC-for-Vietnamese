import networkx as nx
from underthesea import sent_tokenize, word_tokenize

def textrank_summary(text, topk=3):
    sentences = sent_tokenize(text)
    G = nx.Graph()
    for i, s1 in enumerate(sentences):
        for j, s2 in enumerate(sentences):
            if i != j:
                sim = len(set(word_tokenize(s1)) & set(word_tokenize(s2))) / (len(word_tokenize(s1)) + len(word_tokenize(s2)))
                G.add_edge(i, j, weight=sim)
    ranks = nx.pagerank(G)
    top_idx = sorted(ranks, key=ranks.get, reverse=True)[:topk]
    return ' '.join([sentences[i] for i in sorted(top_idx)])
