"""Shared text helpers used by every component."""
import re

import pandas as pd

COLUMNS = ["id", "title", "text", "category"]


def load_corpus(path):
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"corpus is missing columns: {missing}")

    df = df.dropna(subset=["id", "text"]).copy()
    df["id"] = df["id"].astype(int)
    for col in ["title", "text", "category"]:
        df[col] = df[col].fillna("").astype(str).str.strip()
    if df["id"].duplicated().any():
        raise ValueError("duplicated ids in corpus")

    df["full_text"] = df["title"] + ". " + df["text"]
    df["clean_text"] = df["full_text"].apply(clean_text)
    return df.reset_index(drop=True)


def clean_text(text):
    # lowercase, no punctuation, no numbers, single spaces
    text = str(text).lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"[\d_]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def split_sentences(text):
    parts = re.split(r"(?<=[.!?])\s+", str(text).strip())
    return [p.strip() for p in parts if len(p.strip()) > 3]
