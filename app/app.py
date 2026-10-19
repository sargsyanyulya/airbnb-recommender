import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path

DATA = Path(__file__).parent / "app_data"
SHOW = ["listing_id", "name", "city", "neighbourhood", "room_type", "price"]

st.set_page_config(page_title="Airbnb Recommender", page_icon="🏠", layout="wide")


@st.cache_resource(show_spinner="Loading data...")
def load():
    listings = pd.read_csv(DATA / "listings_slim.csv.gz")
    a = np.load(DATA / "als_demo.npz")
    als = {k: a[k] for k in a.files}
    als["item_factors"] = als["item_factors"].astype(np.float32)
    als["user_factors"] = als["user_factors"].astype(np.float32)
    pos = np.full(len(listings), -1, dtype=np.int64)
    pos[als["item_idx"]] = np.arange(len(als["item_idx"]))
    X = np.load(DATA / "item_features.npz")["X"].astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-9
    city_codes = listings["city"].astype("category").cat.codes.values
    return listings, als, pos, X, city_codes


listings, als, pos, X, city_codes = load()


def history_of(u):
    return als["hist_idx"][als["hist_ptr"][u]:als["hist_ptr"][u + 1]]


def als_top(u, k):
    seen = pos[history_of(u)]
    scores = als["item_factors"] @ als["user_factors"][u]
    scores[seen[seen >= 0]] = -np.inf
    top = np.argpartition(-scores, k)[:k]
    top = top[np.argsort(-scores[top])]
    return als["item_idx"][top], scores[top]


def cold_top(hist, k):
    city = np.bincount(city_codes[hist]).argmax()
    cand = np.where((pos < 0) & (city_codes == city))[0]
    p = X[hist].sum(0)
    p /= np.linalg.norm(p) + 1e-9
    s = X[cand] @ p
    top = np.argsort(-s)[:k]
    return cand[top], s[top]


def hybrid(u, k, n_cold):
    rec, sc = als_top(u, k)
    if n_cold == 0:
        return rec, sc, np.zeros(len(rec), bool)
    c_idx, c_sc = cold_top(history_of(u), n_cold)
    keep, scores = list(rec[:k - n_cold]), list(sc[:k - n_cold])
    is_cold = [False] * len(keep)
    for i, s in zip(c_idx, c_sc):
        if i not in keep:
            keep.append(i); scores.append(s); is_cold.append(True)
    return np.array(keep), np.array(scores), np.array(is_cold)


st.title("🏠 Airbnb Recommendation System")
st.caption("ALS (Matrix Factorization) + Content-Based cold-start · 279,599 homes, 10 cities")
tab1, tab2, tab3 = st.tabs(["👤 User Recommendations", "🆕 Similar Homes (cold-start)", "📊 Results"])

# ------------------------------------------------ TAB 1
with tab1:
    c1, c2, c3 = st.columns([2, 1, 1])
    n_users = len(als["demo_user_ids"])
    if "u" not in st.session_state:
        st.session_state.u = 0
    if c1.button("🎲 Random User"):
        st.session_state.u = int(np.random.randint(n_users))
    u = c1.number_input("User ID (0 … %d)" % (n_users - 1), 0, n_users - 1, key="u")
    k = c2.slider("Number of recommendations", 3, 10, 5)
    n_cold = c3.slider("Cold-start slots (hybrid)", 0, 3, 0)
    st.write("reviewer_id:", int(als["demo_user_ids"][u]))

    hist = history_of(u)
    rec, sc, is_cold = hybrid(u, k, n_cold)

    st.subheader("Previous History")
    st.dataframe(listings.iloc[hist][SHOW], use_container_width=True, hide_index=True)

    st.subheader("Recommended Homes")
    out = listings.iloc[rec][SHOW].copy()
    out["score"] = np.round(sc, 4)
    out["type"] = np.where(is_cold, "🆕 content-based", "ALS")
    st.dataframe(out, use_container_width=True, hide_index=True)

    true_item = int(als["test_item"][u])
    st.subheader("Actual Next Home (hidden test)")
    st.dataframe(listings.iloc[[true_item]][SHOW], use_container_width=True, hide_index=True)
    if true_item in set(rec.tolist()):
        st.success("✅ Model guessed it in top-%d" % k)
    else:
        st.info("Did not guess for this user (overall Recall@5 ≈ 1.3%)")

    m = listings.iloc[rec][["latitude", "longitude"]].dropna()
    if len(m):
        st.map(m, latitude="latitude", longitude="longitude")

# ------------------------------------------------ TAB 2
with tab2:
    st.write("Select a home to find similar homes in the same city (cosine similarity, 56 features).")
    only_cold = st.checkbox("Only homes without reviews (cold-start)", value=True)
    city = st.selectbox("City", sorted(listings["city"].unique()))
    mask = listings["city"] == city
    if only_cold:
        mask &= listings["number_of_reviews"] == 0
    pool = listings[mask]
    if pool.empty:
      st.warning("There are no houses without reviews in this city, so all houses are displayed.")
      pool = listings[listings["city"] == city]
    if st.session_state.get("pick_key") != (city, only_cold):
        st.session_state.pick = int(pool.index[0])
        st.session_state.pick_key = (city, only_cold)
    if st.button("🎲 Random Home"):
        st.session_state.pick = int(pool.sample(1).index[0])
    typed = st.text_input("or enter listing_id", "")
    if typed.strip().isdigit():
        m = listings.index[listings["listing_id"] == int(typed)]
        if len(m):
            st.session_state.pick = int(m[0])
        else:
            st.warning("Listing ID does not exist")
    p = st.session_state.pick

    st.subheader("Selected Home")
    st.dataframe(listings.iloc[[p]][SHOW + ["bedrooms", "number_of_reviews"]],
                 hide_index=True, use_container_width=True)
    same = np.where(city_codes == city_codes[p])[0]
    s = X[same] @ X[p]
    s[same == p] = -1
    top = np.argsort(-s)[:10]
    res = listings.iloc[same[top]][SHOW + ["bedrooms"]].copy()
    res["similarity"] = np.round(s[top], 4)
    st.subheader("Top 10 Similar Homes")
    st.dataframe(res, hide_index=True, use_container_width=True)

# ------------------------------------------------ TAB 3
with tab3:
    R = DATA / "results"
    if (R / "summary_table.csv").exists():
        df = pd.read_csv(R / "summary_table.csv")
        st.subheader("Summary Table (10,000 test users, @5)")
        st.dataframe(df.round(5), hide_index=True, use_container_width=True)
        st.bar_chart(df.set_index("Model")[["Recall@5", "NDCG@5"]])
    for name, title in [("cold_start_table", "Cold-start test (1,359 cases)"),
                        ("hybrid_table", "Hybrid: ALS + content-based slots"),
                        ("ci_table", "Bootstrap 95% CI"),
                        ("als_tuning", "ALS hyperparameter tuning"),
                        ("knn_tuning", "KNN hyperparameter tuning")]:
        if (R / f"{name}.csv").exists():
            st.subheader(title)
            st.dataframe(pd.read_csv(R / f"{name}.csv").round(5), hide_index=True, use_container_width=True)