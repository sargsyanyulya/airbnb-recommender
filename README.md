# Airbnb Recommendation System

ALS (Matrix Factorization) + content-based cold-start hybrid.
279,599 listings, 10 cities, 140,786 users.

**Live demo:** [PASTE STREAMLIT LINK HERE]

## Results (10,000 test users, @5)

| Model | Recall@5 | NDCG@5 |
|---|---|---|
| ALS | 0.013 | 0.00868 |
| Content-Based | 0.0064 | 0.00395 |
| KNN (cosine, K=50) | 0.0049 | 0.00329 |
| Popularity | 0.0005 | 0.00028 |

On cold-start items (no training interactions, 1,359 test cases) only Content-Based is non-zero (Recall@5 = 0.0044); the hybrid (ALS + content-based slots) trades some overall accuracy for exposure of new listings.

## Project structure

- `notebooks/` — 01 EDA → 06 export for app (includes `recsys_utils.py`)
- `app/` — Streamlit application and its data (`app/app_data`)

## Run the app

```bash
pip install -r requirements.txt
cd app
streamlit run app.py
```

## Reproduce

1. Download the dataset from Kaggle: https://www.kaggle.com/datasets/mysarahmadbhat/airbnb-listings-reviews and put the CSV files (`Listings .csv`, `Reviews.csv` and the two data dictionaries) into `data/`
2. `pip install -r requirements-notebooks.txt`
3. Run notebooks 01 → 06 in order
4. `cd app && streamlit run app.py`

`data/` and `models/` are not in the repo because of size.

## References

- Hu, Koren, Volinsky (2008). *Collaborative Filtering for Implicit Feedback Datasets.* ICDM.
- `implicit` library: https://github.com/benfred/implicit