# Airbnb Recommendation System

ALS (Matrix Factorization) + content-based cold-start hybrid.
279,599 listings, 10 cities, 140,786 users.

**Live demo:** (link will be added here)

## Results (10,000 test users, @5)

| Model | Recall@5 | NDCG@5 |
|---|---|---|
| ALS | 0.013 | 0.00868 |
| Content-Based | 0.0064 | 0.00395 |
| KNN (cosine, K=50) | 0.0049 | 0.00329 |
| Popularity | 0.0005 | 0.00028 |

## Project structure

- `notebooks/` — 01 EDA → 06 export for app
- `app/` — Streamlit application and its data (`app/app_data`)

## Reproduce

1. Download the dataset from Kaggle: https://www.kaggle.com/datasets/mysarahmadbhat/airbnb-listings-reviews and put it into `data/`
2. `pip install -r requirements-notebooks.txt`
3. Run notebooks 01 → 06 in order
4. `cd app && streamlit run app.py`

`data/` and `models/` are not in the repo because of size.