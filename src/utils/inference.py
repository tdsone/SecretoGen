"""
Utility functions for using the model at inference time.
"""

import torch
import sys
import pandas as pd
from tqdm.auto import tqdm
import numpy as np
import argparse
import os
from pathlib import Path

from src.models.seq2seq import Seq2SeqTransformer
from src.utils.dataset import PredictionSeq2SeqDataset
from src.utils.alphabets import AA_TO_ID, CODON_TO_ID, CODON_TO_AA, DummyTaxonomyMapping
from src.utils.download import download_checkpoint

DEFAULT_MODEL_CHECKPOINT = "checkpoints/secretogen.pt"
DEFAULT_TAXONOMY_DIR = "data/taxonomy_mappings"


def pick_device() -> torch.device:

    # Pick first available device from list
    devices = [
        [torch.cuda.is_available(), torch.device("cuda")],
        [
            torch.backends.mps.is_available(),
            torch.device("mps"),
        ],  # speedup over cpu for mac users with M1 chip
        [True, torch.device("cpu")],
    ]

    device = next(filter(lambda x: x[0], devices))[1]

    return device


def load_model(weights, all_taxonomy_levels=True, codons=False):
    state_dict = torch.load(weights, map_location="cpu")["module"]
    state_dict["tok_emb.embedding.weight"] = state_dict["generator.weight"]
    model = Seq2SeqTransformer(
        12,
        12,
        1024,
        16,
        aa_vocab_size=66 if codons else 23,
        org_vocab_size=(
            [11097 + 1, 4414 + 1, 1448 + 1, 596 + 1, 232 + 1, 112 + 1, 3 + 1, 3 + 1]
            if all_taxonomy_levels
            else 11097 + 1
        ),
        dim_feedforward=2048,
        dropout=0.1,
        pad_idx=0,
    )
    model.load_state_dict(state_dict)
    model.eval()

    return model


def init_model(device: torch.device) -> Seq2SeqTransformer:
    # if the checkpoint file doesn't exist, we download it to the parent dir of args.checkpoint
    if not os.path.exists(DEFAULT_MODEL_CHECKPOINT):
        parent_dir = os.path.dirname(DEFAULT_MODEL_CHECKPOINT)
        download_checkpoint(parent_dir)

    model = load_model(
        DEFAULT_MODEL_CHECKPOINT,
    )
    model.to(device)

    return model


def compute_perplexities(
    model,
    loader,
    no_org=False,
    all_taxonomy_levels=True,
    translate_codons=False,
    device=torch.device("cpu"),
):

    ppl = []

    with torch.no_grad():
        for idx, batch in tqdm(enumerate(loader), total=len(loader)):

            proteins, prot_masks, sps, sp_masks, org_level_targets = batch
            proteins, prot_masks, sps, sp_masks, org_level_targets = (
                proteins.to(device),
                prot_masks.to(device),
                sps.to(device),
                sp_masks.to(device),
                org_level_targets.to(device),
            )
            if no_org:
                orgs = None
            if all_taxonomy_levels:
                orgs = org_level_targets
            else:
                orgs = org_level_targets[:, 0]  # species_id for conditioning token

            proteins = proteins.transpose(1, 0)
            sps = sps.transpose(1, 0)

            # reindex for correct next token prediction.
            sps_input = sps[:-1, :]
            sps_tgt = sps[1:, :]

            aa_logits, hidden_states, hidden_state_mask = model(
                proteins, sps_input, orgs
            )

            # default:
            sp_loss = torch.nn.functional.cross_entropy(
                aa_logits.reshape(-1, model.aa_vocab_size),
                sps_tgt.reshape(-1),
                reduction="mean",
                ignore_index=0,
            )
            # print(np.exp(sp_loss.mean().item()))
            for i in range(aa_logits.shape[1]):

                if translate_codons:

                    l = torch.nn.functional.nll_loss(
                        make_aa_logits_from_codon_logits(aa_logits[:, i, :]).reshape(
                            -1, 23
                        ),
                        make_aa_labels_from_codon_labels(sps_tgt[:, i]).reshape(-1),
                        reduction="mean",
                        ignore_index=0,
                    ).item()
                    # import ipdb; ipdb.set_trace()
                else:
                    l = torch.nn.functional.nll_loss(
                        make_aa_logits(aa_logits[:, i, :]).reshape(
                            -1, model.aa_vocab_size
                        ),
                        sps_tgt[:, i].reshape(-1),
                        reduction="mean",
                        ignore_index=0,
                    ).item()

                ppl.append(np.exp(l))

    return np.array(ppl)


def make_aa_logits_from_codon_logits(
    codon_logits: torch.Tensor, start_codon: str = "ATG"
) -> torch.Tensor:
    """
    Converts codon logits to amino acid logits.
    Note that this does not need to handle start codons, as when
    doing next token prediction we never predict the start codon,
    as the targets are shifted by one position.

    """

    # we do the summing in probs and then convert back to logits
    aa_probs = torch.zeros(codon_logits.shape[0], 23, device=codon_logits.device)

    codon_probs = torch.softmax(codon_logits, dim=-1)

    aa_probs[:, AA_TO_ID["<pad>"]] = codon_probs[:, CODON_TO_ID["<pad>"]]
    aa_probs[:, AA_TO_ID["<eos>"]] = codon_probs[:, CODON_TO_ID["<eos>"]]

    # special case on codon at position 0: could be a non-standard start codon. Can't just map to M.
    # instead, check what the actual start codon in the seq was and map that prob to M also instead.
    # start_codon_prob = codon_probs[0, CODON_TO_ID[start_codon]]
    # aa_probs[0, AA_TO_ID['M']] += start_codon_prob
    # for codon, aa in CODON_TO_AA.items():
    #     if codon != start_codon:
    #         aa_probs[0, AA_TO_ID[aa]] += codon_probs[0, CODON_TO_ID[codon]]

    for codon, aa in CODON_TO_AA.items():
        aa_probs[:, AA_TO_ID[aa]] += codon_probs[:, CODON_TO_ID[codon]]

    aa_logits = torch.log(aa_probs)
    return aa_logits


def make_aa_logits(aa_logits):
    """
    Process regular AA logits exactly the same as remapped ones.
    softmax, log. Should be the same as CrossEntropyLoss but better be sure.
    """
    aa_probs = torch.softmax(aa_logits, dim=-1)
    aa_logits = torch.log(aa_probs)
    return aa_logits


def make_aa_labels_from_codon_labels(codon_labels: torch.Tensor) -> torch.Tensor:
    """
    Converts codon labels to amino acid labels.
    Note that this does not need to handle start codons, as when
    doing next token prediction we never predict the start codon,
    as the targets are shifted by one position.

    """
    aa_labels = torch.zeros(
        codon_labels.shape[0], device=codon_labels.device, dtype=torch.long
    )
    aa_labels[codon_labels == CODON_TO_ID["<pad>"]] = AA_TO_ID["<pad>"]
    aa_labels[codon_labels == CODON_TO_ID["<eos>"]] = AA_TO_ID["<eos>"]

    for codon, aa in CODON_TO_AA.items():
        aa_labels[codon_labels == CODON_TO_ID[codon]] = AA_TO_ID[aa]
    return aa_labels
