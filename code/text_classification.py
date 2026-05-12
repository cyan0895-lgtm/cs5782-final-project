"""
text_classification.py
Reproducing Wu et al. (2019) Table 4 — SGC vs GCN on text classification.

Requires sgc.py, gcn.py (and optionally fastgcn.py) in the same directory.

Requires datasets in PROJECT_ROOT/data/text:
- r8.clean : raw text corpus (one doc per line, cleaned)
- r8       : doc splits and labels (one line per doc: <doc_id>

Usage:
    python text_classification.py --K 2 --dataset R8 --svd_dim 800

svd_dim SVD_DIM
    TruncatedSVD output dimension: reduces node feature matrix X from (num_docs+num_words) x vocab_size down 
    to this many dimensions before graph propagation. Lower = faster but less expressive.
    This dimensionality reduction is mainly introduced to reduce memory usage and avoid out-of-memory (OOM)
K
    SGC propagation steps (K=2 is best for R8 according to paper)
"""

import argparse
import datetime
import json
import math
import os
import random
import sys
import time
import warnings

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from collections import Counter, defaultdict
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Import sibling model files (sgc.py / gcn.py must be in the same directory)
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(SCRIPT_DIR, "..")
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from sgc import SGC
from gcn import GCN


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="SGC text classification (Wu et al. 2019 Table 4)")
    parser.add_argument("--dataset",     default="R8",
                        choices=["R8"],
                        help="Text dataset name")
    parser.add_argument("--K",           type=int,   default=2,
                        help="SGC propagation steps")
    parser.add_argument("--window",      type=int,   default=20,
                        help="PMI sliding window size")
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--svd_dim",     type=int,   default=800,
                        help="TruncatedSVD output dimension")
    parser.add_argument("--gcn_hidden",  type=int,   default=200)
    parser.add_argument("--gcn_epochs",  type=int,   default=400)
    parser.add_argument("--gcn_lr",      type=float, default=0.02)
    parser.add_argument("--gcn_wd",      type=float, default=5e-4)
    parser.add_argument("--gcn_dropout", type=float, default=0.5)
    parser.add_argument("--data_dir",    default=os.path.join(PROJECT_ROOT, "data", "text"),
                        help="Directory containing <dataset>.clean and <dataset>")
    parser.add_argument("--results_dir", default=os.path.join(PROJECT_ROOT, "results", "text_classification"),
                        help="Directory for JSON / TXT result files")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_corpus(data_dir: str, dataset: str):
    dataset_lower = dataset.lower()
    corpus_file = os.path.join(data_dir, f"{dataset_lower}.clean")
    label_file  = os.path.join(data_dir, f"{dataset_lower}")

    with open(corpus_file) as f:
        docs_raw = [line.strip() for line in f]

    splits, labels_str = [], []
    with open(label_file) as f:
        for line in f:
            parts = line.strip().split()
            splits.append(parts[1])
            labels_str.append(parts[2])

    label_set  = sorted(set(labels_str))
    label2id   = {l: i for i, l in enumerate(label_set)}
    labels_int = [label2id[l] for l in labels_str]

    train_idx = [i for i, s in enumerate(splits) if s == "train"]
    test_idx  = [i for i, s in enumerate(splits) if s == "test"]

    print(f"Docs: {len(docs_raw)} | Train: {len(train_idx)} | Test: {len(test_idx)}")
    print(f"Classes ({len(label_set)}): {label_set}")
    return docs_raw, labels_int, label_set, train_idx, test_idx


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------
def build_vocabulary(docs_raw):
    tokenized = [doc.split() for doc in docs_raw]
    doc_freq  = Counter()
    for tokens in tokenized:
        doc_freq.update(set(tokens))
    vocab   = sorted(doc_freq.keys())
    word2id = {w: i for i, w in enumerate(vocab)}
    return tokenized, vocab, word2id


def build_graph(tokenized, word2id, num_docs: int, window: int):
    """Build word-document TF-IDF edges and word-word PMI edges."""
    num_words = len(word2id)

    # --- TF matrix ---
    rows_tf, cols_tf, vals_tf = [], [], []
    for doc_i, tokens in enumerate(tokenized):
        cnt = Counter(tokens)
        for word, freq in cnt.items():
            if word in word2id:
                rows_tf.append(doc_i)
                cols_tf.append(word2id[word])
                vals_tf.append(float(freq))

    tf_matrix    = sp.csr_matrix((vals_tf, (rows_tf, cols_tf)),
                                  shape=(num_docs, num_words), dtype=np.float32)
    tfidf_matrix = TfidfTransformer().fit_transform(tf_matrix)

    # --- PMI ---
    pair_count  = defaultdict(float)
    word_count  = defaultdict(float)
    num_windows = 0
    for tokens in tokenized:
        for start in range(len(tokens)):
            win = [w for w in tokens[start:start + window] if w in word2id]
            if len(win) < 2:
                continue
            num_windows += 1
            for w in win:
                word_count[w] += 1
            for i in range(len(win)):
                for j in range(i + 1, len(win)):
                    pair_count[(win[i], win[j])] += 1
                    pair_count[(win[j], win[i])] += 1

    edge_rows, edge_cols, edge_vals = [], [], []
    seen = set()
    for (wi, wj), cnt in pair_count.items():
        if (wj, wi) in seen:
            continue
        seen.add((wi, wj))
        pmi = math.log2(
            (cnt / num_windows) /
            (word_count[wi] / num_windows * word_count[wj] / num_windows)
        )
        if pmi <= 0:
            continue
        ni = num_docs + word2id[wi]
        nj = num_docs + word2id[wj]
        edge_rows += [ni, nj]
        edge_cols += [nj, ni]
        edge_vals += [pmi, pmi]

    # --- TF-IDF doc-word edges ---
    cx = tfidf_matrix.tocoo()
    for di, wi, v in zip(cx.row, cx.col, cx.data):
        ni, nj = di, num_docs + wi
        edge_rows += [ni, nj]
        edge_cols += [nj, ni]
        edge_vals += [float(v), float(v)]

    edge_index  = torch.tensor([edge_rows, edge_cols], dtype=torch.long)
    edge_weight = torch.tensor(edge_vals, dtype=torch.float32)
    print(f"Edges: {edge_index.size(1)}")
    return tf_matrix, edge_index, edge_weight


# ---------------------------------------------------------------------------
# Feature matrix
# ---------------------------------------------------------------------------
def build_features(tf_matrix, num_words: int, svd_dim: int, seed: int):
    num_docs  = tf_matrix.shape[0]
    bow_dense = np.array(tf_matrix.todense(), dtype=np.float32)
    row_sums  = bow_dense.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    doc_feat  = torch.tensor(bow_dense / row_sums)
    word_feat = torch.eye(num_words)
    X = torch.cat([doc_feat, word_feat], dim=0)
    print(f"Pre-SVD feature matrix: {X.shape}")

    svd = TruncatedSVD(n_components=svd_dim, random_state=seed)
    X   = torch.tensor(svd.fit_transform(X.numpy()), dtype=torch.float32)
    print(f"Post-SVD feature matrix: {X.shape}")
    return X


# ---------------------------------------------------------------------------
# SGC
# ---------------------------------------------------------------------------
def run_sgc(X, edge_index, edge_weight, labels_int, train_idx, test_idx,
            num_classes, num_docs, svd_dim, K, seed):
    device = torch.device("cpu")

    X_gpu  = X.to(device)
    ei_gpu = edge_index.to(device)
    ew_gpu = edge_weight.to(device)

    sgc_model = SGC(
        in_channels=svd_dim,
        out_channels=num_classes,
        K=K,
        cached=True,
        dropout=0.0,
    ).to(device)

    t0 = time.time()
    sgc_model.precompute(X_gpu, ei_gpu, edge_weight=ew_gpu)
    precompute_time = time.time() - t0

    X_bar     = sgc_model._cached_x

    # Slice only the doc nodes (first num_docs rows); word nodes follow after
    X_doc_raw = X_bar[:num_docs].cpu().numpy()  
    scaler    = StandardScaler()
    X_doc     = scaler.fit_transform(X_doc_raw)
    print(f"Precompute time: {precompute_time:.2f}s | Doc features: {X_doc.shape}")

    X_train  = X_doc[train_idx]
    y_train  = np.array(labels_int)[train_idx]
    X_test   = X_doc[test_idx]
    y_test   = np.array(labels_int)[test_idx]

    best_C, best_val = 1.0, 0.0
    for C in [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]:
        clf = LogisticRegression(C=C, solver="lbfgs", max_iter=1000,
                                 multi_class="multinomial", random_state=seed)
        sc  = cross_val_score(clf, X_train, y_train, cv=5).mean()
        if sc > best_val:
            best_val, best_C = sc, C
    print(f"Best C={best_C}  val_acc={best_val:.4f}")

    t0 = time.time()
    clf_sgc = LogisticRegression(C=best_C, solver="lbfgs", max_iter=1000,
                                 multi_class="multinomial", random_state=seed)
    clf_sgc.fit(X_train, y_train)
    train_time_sgc = time.time() - t0

    sgc_acc = clf_sgc.score(X_test, y_test)
    print(f"SGC Test Accuracy       : {sgc_acc * 100:.2f}%")
    print(f"Train time (LR only)    : {train_time_sgc:.3f}s")
    print(f"Total (precompute + LR) : {precompute_time + train_time_sgc:.3f}s")
    return sgc_acc, best_C, precompute_time, train_time_sgc


# ---------------------------------------------------------------------------
# GCN
# ---------------------------------------------------------------------------
def run_gcn(X, edge_index, labels_int, train_idx, test_idx,
            num_nodes, num_classes, svd_dim,
            hidden, epochs, lr, wd, dropout):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"GCN device: {device}")

    X_gpu  = X.to(device)
    ei_gpu = edge_index.to(device)

    Y_graph = torch.full((num_nodes,), -1, dtype=torch.long)
    for i in train_idx + test_idx:
        Y_graph[i] = labels_int[i]

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask  = torch.zeros(num_nodes, dtype=torch.bool)
    for i in train_idx:
        train_mask[i] = True
    for i in test_idx:
        test_mask[i]  = True

    data = Data(
        x=X_gpu,
        edge_index=ei_gpu,
        y=Y_graph.to(device),
        train_mask=train_mask.to(device),
        test_mask=test_mask.to(device),
    )

    gcn_model = GCN(
        in_channels=svd_dim,
        hidden_channels=hidden,
        out_channels=num_classes,
        dropout=dropout,
    ).to(device)

    optimizer = torch.optim.Adam(gcn_model.parameters(), lr=lr, weight_decay=wd)

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        gcn_model.train()
        optimizer.zero_grad()
        out  = gcn_model(data.x, data.edge_index)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        if epoch % 50 == 0:
            print(f"  Epoch {epoch:3d} | loss {loss.item():.4f}")
    train_time_gcn = time.time() - t0

    gcn_model.eval()
    with torch.no_grad():
        pred = gcn_model(data.x, data.edge_index).argmax(1)
    gcn_acc = (pred[data.test_mask] == data.y[data.test_mask]).float().mean().item()
    print(f"GCN Test Accuracy : {gcn_acc * 100:.2f}%")
    print(f"Train time        : {train_time_gcn:.1f}s")
    return gcn_acc, train_time_gcn


# ---------------------------------------------------------------------------
# Summary & save
# ---------------------------------------------------------------------------
PAPER_RESULTS = {
    "r8":      (97.2, 97.0),
    "r52":     (94.0, 93.8),
    "ohsumed": (68.5, 68.2),
    "20ng":    (88.5, 87.9),
    "mr":      (75.9, 76.3),
}


def print_summary(dataset, K, sgc_acc, gcn_acc, precompute_time,
                  train_time_sgc, train_time_gcn):
    p_sgc, p_gcn = PAPER_RESULTS.get(dataset.lower(), ("?", "?"))
    speedup = train_time_gcn / (precompute_time + train_time_sgc)
    sep = "=" * 52
    print(sep)
    print(f"  Dataset: {dataset}   K={K}")
    print(sep)
    print(f"  {'':26s} {'Ours':>7s}   {'Paper':>7s}")
    print(f"  {'SGC (precompute + LR)':26s} {sgc_acc*100:6.2f}%   {p_sgc}%")
    print(f"  {'GCN':26s} {gcn_acc*100:6.2f}%   {p_gcn}%")
    print(f"  SGC speedup over GCN  : {speedup:.1f}x")
    print(sep)
    return speedup, p_sgc, p_gcn


def save_results(results_dir, dataset, K, sgc_acc, gcn_acc, best_C,
                 precompute_time, train_time_sgc, train_time_gcn,
                 speedup, p_sgc, p_gcn, seed):
    os.makedirs(results_dir, exist_ok=True)
    results = {
        "dataset": dataset,
        "K":       K,
        "seed":    seed,
        "sgc": {
            "test_acc":          round(sgc_acc * 100, 2),
            "best_C":            best_C,
            "precompute_time_s": round(precompute_time, 3),
            "train_time_s":      round(train_time_sgc, 3),
            "total_time_s":      round(precompute_time + train_time_sgc, 3),
            "paper_acc":         p_sgc,
        },
        "gcn": {
            "test_acc":     round(gcn_acc * 100, 2),
            "train_time_s": round(train_time_gcn, 1),
            "paper_acc":    p_gcn,
        },
        "sgc_speedup_over_gcn": round(speedup, 1),
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    json_path = os.path.join(results_dir, f"text_cls_{dataset}_K{K}.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved JSON : {json_path}")

    txt_path = os.path.join(results_dir, f"text_cls_{dataset}_K{K}.txt")
    with open(txt_path, "w") as f:
        f.write(f"Dataset : {dataset}\n")
        f.write(f"K       : {K}\n")
        f.write(f"Seed    : {seed}\n")
        f.write(f"Time    : {results['timestamp']}\n\n")
        f.write(f"{'':26s} {'Ours':>8s}   {'Paper':>8s}\n")
        f.write(f"{'SGC (precompute + LR)':26s} {sgc_acc*100:7.2f}%   {p_sgc}%\n")
        f.write(f"{'GCN':26s} {gcn_acc*100:7.2f}%   {p_gcn}%\n")
        f.write(f"\nSGC speedup over GCN : {speedup:.1f}x\n")
        f.write(f"SGC total time       : {precompute_time + train_time_sgc:.3f}s\n")
        f.write(f"GCN train time       : {train_time_gcn:.1f}s\n")
    print(f"Saved TXT  : {txt_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    set_seed(args.seed)

    print(f"\n{'='*52}")
    print(f"  Dataset={args.dataset}  K={args.K}  seed={args.seed}")
    print(f"{'='*52}\n")

    # 1. Load corpus
    docs_raw, labels_int, label_set, train_idx, test_idx = load_corpus(
        args.data_dir, args.dataset
    )
    num_classes = len(label_set)

    # 2. Vocabulary
    tokenized, vocab, word2id = build_vocabulary(docs_raw)
    num_docs  = len(docs_raw)
    num_words = len(vocab)
    num_nodes = num_docs + num_words
    print(f"Vocab: {num_words} | Total nodes: {num_nodes}")

    # 3. Graph
    tf_matrix, edge_index, edge_weight = build_graph(
        tokenized, word2id, num_docs, args.window
    )

    # 4. Features
    X = build_features(tf_matrix, num_words, args.svd_dim, args.seed)

    # 5. SGC
    print("\n--- SGC ---")
    sgc_acc, best_C, precompute_time, train_time_sgc = run_sgc(
        X, edge_index, edge_weight, labels_int, train_idx, test_idx,
        num_classes, num_docs, args.svd_dim, args.K, args.seed,
    )

    # 6. GCN
    print("\n--- GCN ---")
    gcn_acc, train_time_gcn = run_gcn(
        X, edge_index, labels_int, train_idx, test_idx,
        num_nodes, num_classes, args.svd_dim,
        args.gcn_hidden, args.gcn_epochs, args.gcn_lr, args.gcn_wd, args.gcn_dropout,
    )

    # 7. Summary
    print()
    speedup, p_sgc, p_gcn = print_summary(
        args.dataset, args.K, sgc_acc, gcn_acc,
        precompute_time, train_time_sgc, train_time_gcn,
    )

    # 8. Save
    save_results(
        args.results_dir, args.dataset, args.K,
        sgc_acc, gcn_acc, best_C,
        precompute_time, train_time_sgc, train_time_gcn,
        speedup, p_sgc, p_gcn, args.seed,
    )


if __name__ == "__main__":
    main()
