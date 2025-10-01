# app.py — Vertical disease-by-disease view with risk per disease
import re
import json
import numpy as np
import pandas as pd
import streamlit as st
from typing import Dict, List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Ayur Assist — Disease-by-Disease Risk", page_icon="🩺", layout="wide")

# =========================
# Fixed Risk Tables (/10)
# =========================
GROUP_RISK_MAP: Dict[str, float] = {
    "Urinary tract infections":                 4,
    "Muscular disorders":                       5,
    "Cardiomyopathies":                         7,
    "Cardiovascular diseases":                  9,
    "Ear diseases":                             3,
    "Eye diseases":                             4,
    "Hematological diseases":                   6,
    "Liver disease":                            7,
    "Mental health / Psychiatric disorders":    6,
    "Nutritional Deficiency Diseases":          4,
    "Reproductive system diseases":             5,
    "Tropical diseases":                        6,
    "Endocrine and Metabolic Diseases":         7,
    "Cancer and neoplasms":                     9,
    "Zoonotic diseases":                        6,
}

DOSHA_RISK_MAP: Dict[str, float] = {
    "vata":         7.5,
    "pitta":        8.0,
    "kapha":        6.5,
    "vata|pitta":   8.5,
    "vata|kapha":   7.0,
    "pitta|kapha":  8.0,
    "tridosha":     9.5,
    # If missing/none comes through, we’ll default later.
}

# =========================
# Utilities
# =========================
def clean_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.lower()
    s = re.sub(r"[^a-z\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()

ORDER = ["vata", "pitta", "kapha"]
ORDER_SET = set(ORDER)

def normalize_dosha(x: str):
    """Normalize various dosha encodings into:
       {'vata','pitta','kapha','vata|pitta','vata|kapha','pitta|kapha','tridosha'}
    """
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "none"
    s = str(x).strip().lower()

    # Hard catch any tri-dosha wording
    if "tridosha" in s or re.search(r"\btri\s*dosha\b", s):
        return "tridosha"

    # unify separators
    s = re.sub(r"[;,+/]+", "|", s)
    s = s.replace(" ", "")
    s = re.sub(r"[^a-z|]", "", s)

    parts = [p for p in s.split("|") if p]
    parts = [p for p in parts if p in ORDER_SET]
    if not parts:
        return "none"

    present = [d for d in ORDER if d in parts]
    if len(present) >= 3:
        return "tridosha"
    if len(present) == 2:
        return "|".join(present)
    return present[0] if present else "none"

def risk_bucket_0_10(score: float) -> str:
    if score >= 7.0:
        return "High"
    if score >= 4.0:
        return "Medium"
    return "Low"

def risk_for_pair(group_name: str, dosha_combo: str) -> Tuple[float, float, float, str, str]:
    """Returns: (group_risk, dosha_risk, final_risk, level, equation_str)"""
    g = GROUP_RISK_MAP.get(group_name, 5.0)  # mild default
    if dosha_combo in DOSHA_RISK_MAP:
        d = DOSHA_RISK_MAP[dosha_combo]
    else:
        # Simple fallback for 'none' or unexpected combos
        if dosha_combo == "none":
            d = 6.5
        else:
            # try to parse again just in case
            d = 7.0
    final = round(0.6 * g + 0.4 * d, 2)
    level = risk_bucket_0_10(final)
    eq = f"Final Risk (/10) = 0.6×{g:.1f} + 0.4×{d:.1f} = {final:.2f}"
    return g, d, final, level, eq

# =========================
# Load data and build NN index
# =========================
@st.cache_resource
def load_data_and_vectorizer():
    df = pd.read_csv("disease.csv")
    df.rename(columns=lambda c: str(c).strip(), inplace=True)

    # Clean symptoms
    if "Symptoms_clean" not in df.columns:
        df["Symptoms_clean"] = df["Symptoms"].fillna("").apply(clean_text)

    # Normalize dosha column to a standard field
    if "Dosha_Clean" not in df.columns:
        df["Dosha_Clean"] = df["Dosha Types"].apply(normalize_dosha)
    else:
        df["Dosha_Clean"] = df["Dosha_Clean"].apply(normalize_dosha)

    # TF-IDF for nearest-neighbour search
    tfidf = TfidfVectorizer(max_features=30000, ngram_range=(1,2))
    X = tfidf.fit_transform(df["Symptoms_clean"])
    return df, tfidf, X

def nearest_diseases(text: str, df: pd.DataFrame, tfidf: TfidfVectorizer, X, top_k=5) -> pd.DataFrame:
    s = clean_text(text)
    q = tfidf.transform([s])
    sims = cosine_similarity(q, X).ravel()
    idx = np.argsort(sims)[::-1][:top_k]
    rows = []
    for i in idx:
        rows.append({
            "Disease": df.iloc[i]["Disease"],
            "Common disease group": df.iloc[i]["Common disease group"],
            "Disease Group": df.iloc[i]["Disease Group"],
            "Dosha Type (normalized)": df.iloc[i]["Dosha_Clean"],
            "Similarity": float(sims[i])
        })
    return pd.DataFrame(rows)

# =========================
# UI
# =========================
st.title("🩺 Ayur Assist — Disease-by-Disease Prediction & Risk")

with st.sidebar:
    st.header("Settings")
    top_k = st.slider("How many diseases to list", 1, 15, 5)

st.markdown("Enter symptoms (any format). Then click **Predict**.")
default_text = "Burning urination; frequency; urgency; pelvic pain; painful ejaculation; recurrent infections; possible fever; weak urine stream"
symptoms_text = st.text_area("Symptoms", value=default_text, height=140, placeholder="Type symptoms here…")

# Load once
try:
    df, tfidf, X = load_data_and_vectorizer()
except Exception as e:
    st.error(f"Failed to load dataset: {e}")
    st.stop()

if st.button("Predict", type="primary", use_container_width=True):
    if not symptoms_text.strip():
        st.warning("Please enter symptoms first.")
        st.stop()

    with st.spinner("Finding nearest diseases and computing risks…"):
        cands = nearest_diseases(symptoms_text, df, tfidf, X, top_k=top_k)

    if cands.empty:
        st.info("No similar diseases found.")
        st.stop()

    st.success(f"Found {len(cands)} candidate diseases. See details below.")

    # Show disease-by-disease, vertically
    for idx, row in cands.iterrows():
        st.markdown("---")
        st.subheader(f"🧪 Disease: **{row['Disease']}**  (Similarity: {row['Similarity']:.3f})")

        # labels
        group_common = row["Common disease group"]
        group_fine   = row["Disease Group"]
        dosha_combo  = normalize_dosha(row["Dosha Type (normalized)"])

        colA, colB = st.columns([1.3, 1])
        with colA:
            st.markdown("**Details**")
            st.write(f"- **Common disease group**: {group_common}")
            st.write(f"- **Disease group**: {group_fine}")
            st.write(f"- **Dosha type**: `{dosha_combo}`")

        # risk per disease (using the disease's own labels)
        g_risk, d_risk, final_risk, level, equation = risk_for_pair(group_common, dosha_combo)

        with colB:
            st.markdown("**Risk**")
            st.metric("Group Risk (/10)", g_risk)
            st.metric("Dosha Risk (/10)", d_risk)
            st.metric("Final Risk (/10)", final_risk)
            st.metric("Risk Level", level)

        st.caption(f"Equation: {equation}")

    st.markdown("---")
    # Optional: download all candidate results + risks as JSON
    export_rows = []
    for _, row in cands.iterrows():
        group_common = row["Common disease group"]
        group_fine   = row["Disease Group"]
        dosha_combo  = normalize_dosha(row["Dosha Type (normalized)"])
        g_risk, d_risk, final_risk, level, equation = risk_for_pair(group_common, dosha_combo)
        export_rows.append({
            "Disease": row["Disease"],
            "Common disease group": group_common,
            "Disease Group": group_fine,
            "Dosha": dosha_combo,
            "Similarity": float(row["Similarity"]),
            "Group Risk (/10)": g_risk,
            "Dosha Risk (/10)": d_risk,
            "Final Risk (/10)": final_risk,
            "Risk Level": level,
            "Equation": equation
        })
    st.download_button(
        "Download all results (JSON)",
        data=json.dumps(export_rows, indent=2),
        file_name="disease_results_with_risk.json",
        mime="application/json",
        use_container_width=True
    )
else:
    st.info("Enter symptoms and click **Predict** to see disease-by-disease details.")
