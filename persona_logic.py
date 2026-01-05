import pandas as pd
import numpy as np

def build_features(customers: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Aggregate customer events into persona-ready features (90-day window)."""
    ev = events.copy()
    ev["event_dt"] = pd.to_datetime(ev["event_dt"])
    max_dt = ev["event_dt"].max()

    cutoff_90 = max_dt - pd.Timedelta(days=90)
    ev90 = ev[ev["event_dt"] >= cutoff_90]

    p = ev[ev["event_type"]=="purchase"].copy()
    p90 = ev90[ev90["event_type"]=="purchase"].copy()
    r = ev[ev["event_type"]=="review"].copy()

    last_purchase = p.groupby("customer_id")["event_dt"].max()
    purchase_count_90 = p90.groupby("customer_id").size()
    spend_90 = p90.groupby("customer_id")["net_price"].sum()
    aov_90 = p90.groupby("customer_id")["net_price"].mean()
    cat_div_90 = p90.groupby("customer_id")["category"].nunique()
    discount_share_90 = (p90["discount_pct"]>0).groupby(p90["customer_id"]).mean() if len(p90) else pd.Series(dtype=float)

    premium_threshold = p90["list_price"].quantile(0.75) if len(p90) else 0
    premium_share_90 = (p90["list_price"] >= premium_threshold).groupby(p90["customer_id"]).mean() if len(p90) else pd.Series(dtype=float)

    p90m = p90.merge(customers[["customer_id","label_affinity"]], on="customer_id", how="left")
    label_match_90 = (p90m["label"].fillna("") == p90m["label_affinity"]).groupby(p90m["customer_id"]).mean() if len(p90m) else pd.Series(dtype=float)

    reviews_ct = r.groupby("customer_id").size()
    orders_ct = p.groupby("customer_id")["order_id"].nunique()
    review_rate = (reviews_ct / orders_ct).fillna(0)
    avg_rating = r.groupby("customer_id")["rating_value"].mean().fillna(0) if len(r) else pd.Series(dtype=float)
    avg_polarity = r.groupby("customer_id")["polarity_score"].mean().fillna(0) if len(r) else pd.Series(dtype=float)

    def top_brand_share(df):
        c = df["brand"].value_counts(normalize=True)
        return float(c.iloc[0]) if len(c) else 0.0
    brand_share_90 = p90.groupby("customer_id").apply(top_brand_share) if len(p90) else pd.Series(dtype=float)

    feat = pd.DataFrame({
        "customer_id": customers["customer_id"].values,
        "recency_days": (max_dt - customers["customer_id"].map(last_purchase)).dt.days,
        "tenure_days": (max_dt - customers["join_date"]).dt.days,
        "purchase_count_90": customers["customer_id"].map(purchase_count_90).fillna(0).astype(int),
        "spend_90": customers["customer_id"].map(spend_90).fillna(0.0),
        "aov_90": customers["customer_id"].map(aov_90).fillna(0.0),
        "category_diversity_90": customers["customer_id"].map(cat_div_90).fillna(0).astype(int),
        "discount_share_90": customers["customer_id"].map(discount_share_90).fillna(0.0),
        "premium_share_90": customers["customer_id"].map(premium_share_90).fillna(0.0),
        "label_match_90": customers["customer_id"].map(label_match_90).fillna(0.0),
        "review_rate": customers["customer_id"].map(review_rate).fillna(0.0),
        "avg_rating": customers["customer_id"].map(avg_rating).fillna(0.0),
        "avg_polarity": customers["customer_id"].map(avg_polarity).fillna(0.0),
        "top_brand_share_90": customers["customer_id"].map(brand_share_90).fillna(0.0),
    })

    feat["recency_days"] = feat["recency_days"].fillna(999).astype(int)
    feat["tenure_days"] = feat["tenure_days"].fillna(999).astype(int)
    return feat

def assign_personas(features: pd.DataFrame, customers: pd.DataFrame) -> pd.DataFrame:
    f = features.copy()

    spend_hi = f["spend_90"].quantile(0.80)
    freq_hi = f["purchase_count_90"].quantile(0.80)
    disc_hi = f["discount_share_90"].quantile(0.80)
    prem_hi = f["premium_share_90"].quantile(0.80)
    loyal_hi = 0.55
    div_hi = f["category_diversity_90"].quantile(0.75)
    review_hi = f["review_rate"].quantile(0.80)

    if "label_affinity" not in f.columns and "label_affinity" in customers.columns:
        f = f.merge(customers[["customer_id","label_affinity"]], on="customer_id", how="left")

    def _persona(row):
        if row["purchase_count_90"] == 0 and row["recency_days"] > 120:
            return "Lapsed / Winback"
        if row["spend_90"] >= spend_hi and row["purchase_count_90"] >= freq_hi and row["premium_share_90"] >= prem_hi:
            return "Premium Power Shopper"
        if row["top_brand_share_90"] >= loyal_hi and row["purchase_count_90"] >= 3:
            return "Brand Loyalist"
        if row.get("label_affinity","none") != "none" and row["label_match_90"] >= 0.45 and row["purchase_count_90"] >= 2:
            return "Ingredient-Conscious"
        if row["discount_share_90"] >= disc_hi and row["purchase_count_90"] >= 2:
            return "Deal Hunter"
        if row["category_diversity_90"] >= div_hi and row["purchase_count_90"] >= 3:
            return "Category Explorer"
        if row["review_rate"] >= review_hi and row["purchase_count_90"] >= 2:
            return "Reviewer / Researcher"
        if row["purchase_count_90"] >= 2 and row["recency_days"] <= 45:
            return "Routine Replenisher"
        return "Casual Browser"

    f["persona"] = f.apply(_persona, axis=1)
    return f[["customer_id","persona"]]
