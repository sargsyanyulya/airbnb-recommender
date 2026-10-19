import numpy as np
import pandas as pd
from scipy import sparse

METRIC_COLS = ['Precision@5', 'Recall@5', 'MAP@5', 'NDCG@5', 'Coverage', 'Novelty']


def build_split(reviews_df, listings_df, min_interactions=3, n_eval_users=10000, seed=42):
    """
    Sort each user's reviews by date:
      last review          -> TEST
      second-to-last review -> VAL (for hyperparameter tuning)
      the rest             -> TRAIN
    Item index = row position in listings_df.
    """
    id_col = 'listing_id'
    item_ids = listings_df[id_col].values
    item_index = pd.Series(np.arange(len(item_ids)), index=item_ids)
    n_items = len(item_ids)

    df = reviews_df[['reviewer_id', id_col, 'date']].copy()
    df = df[df[id_col].isin(item_index.index)]
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date']).drop_duplicates(['reviewer_id', id_col])

    cnt = df.groupby('reviewer_id')[id_col].transform('size')
    df = df[cnt >= min_interactions].sort_values(['reviewer_id', 'date', id_col])
    df['rank_from_end'] = df.groupby('reviewer_id').cumcount(ascending=False)

    user_ids = np.sort(df['reviewer_id'].unique())
    user_index = pd.Series(np.arange(len(user_ids)), index=user_ids)
    n_users = len(user_ids)
    df['u'] = df['reviewer_id'].map(user_index).values
    df['i'] = df[id_col].map(item_index).values

    def to_csr(d):
        return sparse.csr_matrix(
            (np.ones(len(d), dtype=np.float32), (d['u'].values, d['i'].values)),
            shape=(n_users, n_items))

    train = to_csr(df[df['rank_from_end'] >= 2])      # for tuning
    trainval = to_csr(df[df['rank_from_end'] >= 1])   # for final test

    rng = np.random.default_rng(seed)
    eval_users = np.sort(rng.choice(n_users, size=min(n_eval_users, n_users), replace=False))
    eval_set = set(eval_users.tolist())

    def truth(rank):
        d = df[(df['rank_from_end'] == rank) & (df['u'].isin(eval_set))]
        return {int(u): {int(i)} for u, i in zip(d['u'], d['i'])}

    return {
        'train': train, 'trainval': trainval,
        'val_truth': truth(1), 'test_truth': truth(0),
        'eval_users': eval_users,
        'item_ids': item_ids, 'user_ids': user_ids,
        'n_users': n_users, 'n_items': n_items,
        'pop_train': np.asarray(train.sum(axis=0)).ravel(),
        'pop_trainval': np.asarray(trainval.sum(axis=0)).ravel(),
    }


def evaluate(recs, truth, item_pop, n_users, n_items, k=5):
    """
    recs  : {user: [item, ...]} sorted recommendations
    truth : {user: {item, ...}} ground truth test items
    Coverage = what fraction of the catalog was recommended at least once
    Novelty  = average -log2(popularity), popularity = item interactions / n_users
    """
    disc = 1.0 / np.log2(np.arange(2, k + 2))
    P, R, AP, ND, nov, shown = [], [], [], [], [], set()

    for u, true_items in truth.items():
        rec = list(recs.get(u, []))[:k]
        shown.update(rec)
        rel = np.array([1.0 if i in true_items else 0.0 for i in rec])
        hits = rel.sum()
        P.append(hits / k)
        R.append(hits / len(true_items))
        if hits > 0:
            prec_at_i = np.cumsum(rel) / np.arange(1, len(rel) + 1)
            AP.append((prec_at_i * rel).sum() / min(len(true_items), k))
        else:
            AP.append(0.0)
        dcg = (rel * disc[:len(rel)]).sum()
        idcg = disc[:min(len(true_items), k)].sum()
        ND.append(dcg / idcg)
        if rec:
            nov.extend(-np.log2(np.maximum(item_pop[rec], 1) / n_users))

    return pd.Series({
        'Precision@5': np.mean(P), 'Recall@5': np.mean(R),
        'MAP@5': np.mean(AP), 'NDCG@5': np.mean(ND),
        'Coverage': len(shown) / n_items,
        'Novelty': np.mean(nov) if nov else 0.0,
    })


def _seen(csr, u):
    return set(csr.indices[csr.indptr[u]:csr.indptr[u + 1]].tolist())


def popularity_recs(users, item_pop, seen_csr, k=5):
    """To everyone: the top k most popular homes (excluding already seen ones)."""
    top = np.argsort(-item_pop)[:k + 500]
    recs = {}
    for u in users:
        s = _seen(seen_csr, u)
        recs[int(u)] = [int(i) for i in top if i not in s][:k]
    return recs


def random_recs(users, n_items, seen_csr, k=5, seed=42):
    rng = np.random.default_rng(seed)
    recs = {}
    for u in users:
        s = _seen(seen_csr, u)
        cand = [int(i) for i in rng.choice(n_items, size=k + len(s) + 5, replace=False) if i not in s]
        recs[int(u)] = cand[:k]
    return recs


def implicit_recs(model, users, user_items_csr, k=5, batch=2000):
    """Recommendations from the implicit library (KNN/ALS) model."""
    recs = {}
    users = np.asarray(users)
    for s in range(0, len(users), batch):
        ub = users[s:s + batch]
        ids, _ = model.recommend(ub, user_items_csr[ub], N=k, filter_already_liked_items=True)
        for u, row in zip(ub, ids):
            recs[int(u)] = [int(i) for i in row]
    return recs


def fix_text(s):
    """'prÃÂ¨s' -> 'près' (recovery of doubly encoded text)"""
    if not isinstance(s, str):
        return s
    for _ in range(3):
        try:
            fixed = s.encode('latin1').decode('utf-8')
        except UnicodeError:
            try:
                fixed = s.encode('cp1252').decode('utf-8')
            except UnicodeError:
                break
        if fixed == s:
            break
        s = fixed
    return s


def save_als_factors(model, path):
    def to_np(x):
        return x.to_numpy() if hasattr(x, 'to_numpy') else np.asarray(x)
    np.savez_compressed(path,
                        user_factors=to_np(model.user_factors).astype(np.float32),
                        item_factors=to_np(model.item_factors).astype(np.float32))


def als_user_recs(row, user_factors, item_factors, seen_csr, k=5):
    scores = item_factors.astype(np.float32) @ user_factors[row].astype(np.float32)
    seen = seen_csr.indices[seen_csr.indptr[row]:seen_csr.indptr[row + 1]]
    scores[seen] = -np.inf
    top = np.argpartition(-scores, k)[:k]
    top = top[np.argsort(-scores[top])]
    return top, scores[top]