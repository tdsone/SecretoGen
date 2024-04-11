# SecretoGen

A conditional autoregressive model of signal peptides. SecretoGen generates signal peptides conditional on the mature protein sequence and the host organism.

## Resources

The trained model weights are available at https://erda.ku.dk/archives/1a91453689c691c242f78268ff2fe1aa/published-archive.html. You don't have to download this manually, the scripts will automatically fetch the checkpoint.

## Overview

The SecretoGen model architecture is defined in `src/pretraining/models/seq2seq.py`. The taxonomy vocabulary is stored in `data/taxonomy_mappings`. We use this for mapping a species to its set of taxonomy tokens using `src.utils.alphabets.TaxonomyMapping`.

### Computing perplexities

SecretoGen requires `torch`, `pandas` and `tqdm`. You can install them with `pip install -r requirements.txt`.

The script `compute_perplexity.py` can score SecretoGen perplexities from csv-formatted input data. If `--checkpoint` is not specified, the script will automatically download the model weights and store them in `checkpoints/secretogen.pt`.

```sh
python3 compute_perplexity.py  --data data/efficiency_data/wu.csv --out_file test_run.csv
```

Note that as of torch version 2.2, a warning "enable_nested_tensor is True" will be printed. This flag was not available in the torch version used to train the model, and it should be safe to ignore the warning.

### Generating signal peptides

The script `generate_sequences.py` can generate signal peptides using top-p sampling for a given mature protein sequence and host organism.

This example generates 1000 SPs for a Xylanase in [_Bacillus subtilis_ (1423)](https://www.uniprot.org/taxonomy/1423).

```sh
python3 generate_sequences.py test_samples.csv --org_id 1423 --seq "GSRTITNNEMGNHSGYDYELWKDYGNTSMTLNNGGAFSAGWNNIGNALFRKGKKFDSTRTHHQLGNISINYNASFNPGGNSYLCVYGWTQSPLAEYYIVDSWGTYRPTGAYKGSFYADGGTYDIYETTRVNQPSIIGIATFKQYWSVRQTKRTSGTVSVSAHFRKWESLGMPIGKMYETAFTVEGYQSSGSANVMTNQLFIGN" --top_p 0.75
```

## Benchmark data

SecretoGen was evaluated on a set of studies that experimentally evaluated the secretion efficiency of signal peptides. If you reuse the data, please cite the respective original works.

## Development

This repository is formatted via [black](https://github.com/psf/black). Run `black .` from the root of this repository to format all files.

#### Grasso et al.

    @article{doi:10.1021/acssynbio.2c00328,
    author = {Grasso, Stefano and Dabene, Valentina and Hendriks, Margriet M. W. B. and Zwartjens, Priscilla and Pellaux, René and Held, Martin and Panke, Sven and van Dijl, Jan Maarten and Meyer, Andreas and van Rij, Tjeerd},
    title = {Signal Peptide Efficiency: From High-Throughput Data to Prediction and Explanation},
    journal = {ACS Synthetic Biology},
    volume = {12},
    number = {2},
    pages = {390-404},
    year = {2023},
    doi = {10.1021/acssynbio.2c00328},
    note ={PMID: 36649479},
    URL = {https://doi.org/10.1021/acssynbio.2c00328},
    eprint = { https://doi.org/10.1021/acssynbio.2c00328}
    }

#### Wu et al.

    @article{doi:10.1021/acssynbio.0c00219,
    author = {Wu, Zachary and Yang, Kevin K. and Liszka, Michael J. and Lee, Alycia and Batzilla, Alina and Wernick, David and Weiner, David P. and Arnold, Frances H.},
    title = {Signal Peptides Generated by Attention-Based Neural Networks},
    journal = {ACS Synthetic Biology},
    volume = {9},
    number = {8},
    pages = {2154-2161},
    year = {2020},
    doi = {10.1021/acssynbio.0c00219},
    note ={PMID: 32649182},
    URL = {https://doi.org/10.1021/acssynbio.0c00219},
    eprint = {https://doi.org/10.1021/acssynbio.0c00219}
    }

#### Xue et al.

    @article{https://doi.org/10.1002/advs.202203433,
    author = {Xue, Songlyu and Liu, Xiufang and Pan, Yuyang and Xiao, Chufan and Feng, Yunzi and Zheng, Lin and Zhao, Mouming and Huang, Mingtao},
    title = {Comprehensive Analysis of Signal Peptides in Saccharomyces cerevisiae Reveals Features for Efficient Secretion},
    journal = {Advanced Science},
    volume = {10},
    number = {2},
    pages = {2203433},
    doi = {https://doi.org/10.1002/advs.202203433},
    url = {https://onlinelibrary.wiley.com/doi/abs/10.1002/advs.202203433},
    eprint = {https://onlinelibrary.wiley.com/doi/pdf/10.1002/advs.202203433},
    year = {2023}
    }

#### Zhang et al.

    @article{zhang_optimal_2016,
    title = {Optimal secretion of alkali-tolerant xylanase in {Bacillus} subtilis by signal peptide screening},
    volume = {100},
    issn = {1432-0614},
    url = {https://doi.org/10.1007/s00253-016-7615-4},
    doi = {10.1007/s00253-016-7615-4},
    language = {en},
    number = {20},
    urldate = {2022-11-22},
    journal = {Applied Microbiology and Biotechnology},
    author = {Zhang, Weiwei and Yang, Mingming and Yang, Yuedong and Zhan, Jian and Zhou, Yaoqi and Zhao, Xin},
    month = oct,
    year = {2016},
    pages = {8745--8756},
    }

## Baseline methods

`compute_perplexity_progen.py` and `compute_perplexity_spgen.py` work on the same input format.
For SPGen, you will have to edit the checkpoint paths on lines 23 to 26. Checkpoints were prepared by extracting the state dicts from the checkpoints in the original SPGen repository, so that loading the checkpoint files does no longer depend on the SPGen directory structure for dependencies.
