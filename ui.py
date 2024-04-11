"""
Streamlit (https://streamlit.io/) app to use SecretoGen without coding.

Spin up the ui locally: 
- pip install -r requirements-ui.txt
- streamlit run ui.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import torch

from src.utils.dataset import PredictionSeq2SeqDataset
from src.utils import (
    DEFAULT_TAXONOMY_DIR,
    compute_perplexities,
    pick_device,
    init_model,
)

DEVICE = pick_device()

MODEL = init_model(device=DEVICE)


def ui_compute_secretion_efficiencies(
    dataset: pd.DataFrame,
) -> pd.DataFrame:
    """Function to compute secretion efficiencies from a DataFrame of sequences that the user uploaded.

    Args:
        sequences_df (pd.DataFrame): DataFrame of sequences uploaded by the user as CSV.

    Returns:
        pd.DataFrame: DataFrame with perplexities and sequences (downloadable by the user as CSV).
    """

    test_set = PredictionSeq2SeqDataset(
        dataset=dataset,
        taxonomy_dir=DEFAULT_TAXONOMY_DIR,
        levels_to_use=[
            "species",
            "genus",
            "family",
            "order",
            "class",
            "phylum",
            "kingdom",
            "superkingdom",
        ],
    )

    loader = torch.utils.data.DataLoader(
        test_set, collate_fn=test_set.collate_fn, batch_size=500
    )

    perplexities = compute_perplexities(
        model=MODEL,
        loader=loader,
        device=DEVICE,
    )

    test_set.df["perplexity"] = perplexities

    return test_set.df


st.title("Secretogen: Compute Secretion Efficiencies")

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
if uploaded_file is not None:

    dataset = pd.read_csv(uploaded_file, index_col=0)

    st.subheader("Your Upload")
    st.write(dataset)

    if st.button("Compute Secretion Efficiency Ranking"):
        with st.spinner("Wait for it..."):
            result_df = ui_compute_secretion_efficiencies(dataset)
        st.success("Done!")

        st.subheader("Results")
        st.text("Lower perplexity is associated with higher secretion efficiency:")

        # Display results in an interactive table sorted by perplexity (ascending)
        st.dataframe(result_df.sort_values(by="perplexity"))

        # Download link for results table
        csv = result_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download results as CSV",
            data=csv,
            file_name="secretion_efficiencies.csv",
            mime="text/csv",
        )
