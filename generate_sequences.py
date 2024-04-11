"""
Script to draw conditional samples from the model.
"""

import torch
import pandas as pd
from src.models.seq2seq import Seq2SeqTransformer, create_mask
from src.utils.alphabets import AA_TO_ID, CODON_TO_ID, TaxonomyMapping
from src.utils.dataset import string_to_codon_tokens
from src.utils.download import download_checkpoint

from tqdm.auto import tqdm
import numpy as np
import argparse
import os

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# device = torch.device('cpu')


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


def generate_sequence(
    self,
    src,
    org,
    stop_tokens=[21],
    max_len=100,
    num_samples: int = 1,
    no_prot=False,
    no_org=False,
    top_p: float = 0,
):
    """Autoregressive decoding with sampling to generate sequences."""

    stop_tokens = torch.from_numpy(stop_tokens).to(device)

    src = src.transpose(1, 0)
    src_emb = self.positional_encoding(self.tok_emb(src))

    src_mask, _, src_padding_mask, tgt_padding_mask = create_mask(
        src, src, self.pad_idx
    )

    if no_prot:
        hidden_state_enc = torch.zeros_like(
            src_emb[[0], :, :]
        )  # vanilla pytorch decoder stack needs memory.
        memory_key_padding_mask = torch.zeros_like(tgt_padding_mask[:, [0]])

    else:
        memory_key_padding_mask = src_padding_mask
        hidden_state_enc = self.transformer.encoder(src_emb, src_mask, src_padding_mask)

    if not no_org:
        organism_embeddings = self.organism_embedding(org).unsqueeze(
            0
        )  # (1, batch_size, embedding_dim)

        # repeat stuff for num_samples
        # org = org.tile(num_samples)
        # org = org.unsqueeze(1).tile((1,num_samples,1))
        organism_embeddings = organism_embeddings.tile((1, num_samples, 1))

    hidden_state_enc = hidden_state_enc.tile((1, num_samples, 1))
    memory_key_padding_mask = memory_key_padding_mask.tile((num_samples, 1))

    def _decode(tgt):
        _, tgt_mask, _, tgt_padding_mask = create_mask(
            src, tgt, self.pad_idx, tgt_additional_tokens=0 if no_org else 1
        )
        tgt_emb = self.positional_encoding(self.tok_emb(tgt))
        if not no_org:
            tgt_emb_with_org = torch.concat([organism_embeddings, tgt_emb], dim=0)
            tgt_padding_mask = torch.concat(
                [tgt_padding_mask[:, [0]], tgt_padding_mask], dim=1
            )  # batch_size, len
        else:
            tgt_emb_with_org = tgt_emb

        outs = self.transformer.decoder(
            tgt_emb_with_org,
            hidden_state_enc,
            tgt_mask,
            None,
            tgt_padding_mask,
            memory_key_padding_mask,
        )
        return outs

    y = (
        torch.ones(num_samples, device=org.device, dtype=org.dtype) * 13
    )  # start with M.
    y = y.unsqueeze(0)  # (1,batch_size)

    has_stopped = torch.zeros(num_samples, dtype=bool, device=org.device)
    for i in tqdm(range(max_len - 1), leave=False):
        # import ipdb; ipdb.set_trace()
        out = _decode(y)  # (seq_len, batch_size)
        out = out.transpose(0, 1)
        logits = self.generator(out[:, -1])  # last hidden state

        prob = torch.nn.functional.softmax(logits, dim=1)

        # do top-p sampling
        # https://nn.labml.ai/sampling/nucleus.html
        if top_p > 0.0:
            sorted_probs, sorted_indices = torch.sort(prob, dim=1, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=1)

            nucleus = cumulative_probs < top_p
            nucleus = torch.cat(
                [nucleus.new_ones(nucleus.shape[:-1] + (1,)), nucleus[..., :-1]], dim=-1
            )

            sorted_probs[~nucleus] = 0
            sampled_sorted_inds = sorted_probs.multinomial(
                num_samples=1, replacement=True
            )

            next_token = sorted_indices.gather(-1, sampled_sorted_inds)

        else:
            next_token = prob.multinomial(num_samples=1, replacement=True)

        y = torch.concat([y, next_token.transpose(0, 1)])
        # next_token = next_token.item()

        has_stopped = has_stopped + (torch.isin(next_token.squeeze(), stop_tokens))

        if has_stopped.all():
            break

    # compute perplexity

    # reindex for correct next token prediction.
    sps_input = y[:-1, :]
    sps_tgt = y[1:, :]

    if no_org:
        outs = _decode(sps_input)
    else:
        outs = _decode(sps_input)[1:]  # because of org token
    aa_logits = self.generator(outs)

    ppl = []
    for idx in range(aa_logits.shape[1]):
        l = torch.nn.functional.cross_entropy(
            aa_logits[:, idx].reshape(-1, self.aa_vocab_size),
            sps_tgt[:, idx].reshape(-1),
            reduction="mean",
            ignore_index=0,
        ).item()
        ppl.append(np.exp(l))

    return y.cpu().numpy(), np.array(ppl)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("out_file")
    # parser.add_argument('--all_taxonomy_levels', action='store_true')
    # parser.add_argument('--no_prot', action='store_true')
    # parser.add_argument('--no_org', action='store_true')
    parser.add_argument("--org_id", required=True, type=int)
    parser.add_argument("--seq", required=True)
    # parser.add_argument('--codons',action='store_true')

    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument(
        "--max_aas",
        type=int,
        default=120,
        help="seq will be trimmed to this - SecretoGen was trained on 120",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=100,
        help="Number of samples to generate in parallel",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1000,
        help="Number of samples to generate in total",
    )

    parser.add_argument("--taxonomy_dir", type=str, default="data/taxonomy_mappings")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/secretogen.pt")
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        parent_dir = os.path.dirname(args.checkpoint)
        download_checkpoint(parent_dir)

    model = load_model(
        args.checkpoint,
        all_taxonomy_levels=True,  # not args.no_org,
        codons=False,  # args.codons
    )
    model.to(device)

    levels_to_use = [
        "species",
        "genus",
        "family",
        "order",
        "class",
        "phylum",
        "kingdom",
        "superkingdom",
    ]
    taxonomy_labels = TaxonomyMapping(args.taxonomy_dir)

    id_to_aa = dict(zip(list(AA_TO_ID.values()), list(AA_TO_ID.keys())))

    # encode the conditioning data
    prot = (
        torch.LongTensor(
            np.array(
                [AA_TO_ID[x] for x in args.seq[: args.max_aas]] + [AA_TO_ID["<eos>"]]
            )
        )
        .unsqueeze(0)
        .to(device)
    )
    org = (
        torch.LongTensor(
            taxonomy_labels.get_taxonomy_labels(
                args.org_id, levels_to_return=levels_to_use
            )
        )
        .unsqueeze(0)
        .to(device)
    )

    stop_tokens = np.array([21])

    with torch.no_grad():
        torch.manual_seed(123)

        sps = []
        ppls = []
        n = args.num_samples // args.batch_size
        for _ in tqdm(range(n), total=n):
            preds, ppl = generate_sequence(
                model,
                prot,
                org,
                stop_tokens=stop_tokens,
                num_samples=args.batch_size,
                no_prot=False,
                no_org=False,
                top_p=args.top_p,
            )

            for i in range(preds.shape[1]):
                pred = preds[:, i]
                stop_idx = np.where(np.isin(pred, stop_tokens))[0][0]
                # print(pred.shape, stop_idx)
                if stop_idx > 1:  # it can happen that the sample is only length 1.
                    pred = pred[:stop_idx]
                    sp = "".join([id_to_aa[x] for x in pred.squeeze()])
                else:
                    sp = ""
                sps.append(sp)

            ppls.extend(ppl)

    pd.DataFrame([sps, ppls], index=["Sequence", "Perplexity"]).T.to_csv(args.out_file)


if __name__ == "__main__":
    main()
