# ICF Classification of Physiotherapy Goals with Large Language Models

This repository contains the analysis code for the paper:

> **Agentic ICF classification of physiotherapy goals using Large Language Models in inpatient rehabilitation**
> Mahdie Rafiei, Martin Müller, Ralf Strobl, Eva Grill, Susanne G. R. Klotz, Jan Baumbach, Linda Baumbach
> *Manuscript in preparation. The full citation will be added after publication.*

## About the study

Physiotherapists document patient goals as short free-text statements in electronic health records (EHRs). Manually coding these goals with the International Classification of Functioning, Disability and Health (ICF) is time-consuming. This project evaluates open-weight large language models (LLMs) for automatic ICF classification of German physiotherapy goal statements.

**Phase 1: Classification performance (RQ1)**
- **Data:** 4 reused datasets (4ReusD) with 2,555 expert-annotated goal statements, split 80/20 into training and test sets.
- **Prompting strategies:** zero-shot, few-shot and retrieval-augmented generation (RAG), compared with qwen3.6:35b.
- **Model comparison:** RAG applied to eight open-weight LLMs (8B to 120B parameters). The selected configuration, llama4 with RAG, reached 90.6% component-level accuracy (Cohen's κ = 0.808) and 57.7% exact code match.
- **Scoring:** at four ICF levels (component, chapter, second-level item, exact code).

**Phase 2: ICF characterisation and reliability scoring (RQ2)**
- **Data:** 529 unannotated goal statements from a university hospital EHR (EHR Uni).
- **Classification:** each goal is classified by llama4 with and without RAG.
- **Reliability levels:** agreement between the two runs and the model's confidence give each annotation a reliability level (high, medium or low), validated on the annotated 4ReusD test set.

## Repository contents

```
01_prepare_data.py              Load the four cohorts and create 80/20 train/test splits
02_build_index.py               Embed training goal statements and ICF code descriptions (RAG index)
03_zeroshot.py                  Zero-shot classification
04_fewshot.py                   Few-shot classification
05_rag.py                       RAG classification
06_evaluate.py                  Compare the three prompting strategies (Supplementary Table S6)
07_consistency_audit.py         Code variation among semantically similar goal statements (Table S2)
08_llm_comparison.py            RAG across eight LLMs (Table 4, Tables S7 and S13)
08_results.ipynb                Phase 1 results, tables and figures
09_embedding_comparison.py      Embedding model comparison (Table S3)
10_uke_annotation.py            Phase 2: ICF component and chapter annotation of EHR Uni goals
11_uke_results.ipynb            Phase 2 results, tables and figures
12_kappa.py                     Cohen's κ at all four levels (Table S15)
12_4reusd_tier_validation.py    Validation of reliability levels on the 4ReusD test set (Table S11)
13_uke_reliability_tiers.py     Reliability levels for EHR Uni annotations (Table S9)
14_repeat_runs.py               Run-to-run stability (Table S12)
15_fig_ehr_chapter_prevalence.py  Figure 2: ICF chapter prevalence in EHR Uni goals
16_gptoss_sensitivity.py        Sensitivity analysis with gpt-oss:120b (Table S14)
run_all.py                      Run the Phase 1 steps in sequence
scrape_icf_codes.py             Build the ICF code reference table (BfArM, ICF 2005, German version)
utils/                          Data loading, embeddings, LLM calls and evaluation metrics
```

Some file names use earlier working names: **UKE** refers to the EHR Uni dataset and **tiers** to reliability levels.

## Setup

Requirements: Python 3.10 or later and access to an [Ollama](https://ollama.com) server hosting the LLMs and the embedding model (qwen3-embedding:8b).

The scripts expect this folder layout, with the data, outputs and `.env` one level above the code:

```
project/
├── .env
├── dataset/        goal statement files (not included, see Data availability)
├── output/         created by the scripts
└── scripts/        this repository
```

```bash
mkdir project && cd project
git clone https://github.com/Mahdieh-Rafiei/icf-llm-physio-goals.git scripts
pip install -r scripts/requirements.txt
```

Create `project/.env`:

```
OLLAMA_API_URL=https://your-ollama-server
COSYBIO_API_KEY=your-api-key
LLM_MODEL=llama4:latest
EMBED_MODEL=qwen3-embedding:8b
# Optional: separate embedding server
# EMBED_API_URL=...
# EMBED_API_KEY=...
```

## Usage

```bash
cd project/scripts
python scrape_icf_codes.py          # ICF code reference table
python run_all.py                   # Phase 1: splits, index, zero-shot, few-shot, RAG, evaluation
python 08_llm_comparison.py         # compare the eight LLMs
python 10_uke_annotation.py         # Phase 2 annotation
python 12_4reusd_tier_validation.py  # validate reliability levels
python 13_uke_reliability_tiers.py  # assign reliability levels
```

Results are written to `project/output/`.

## Data availability

The goal statements are clinical data and are **not included** in this repository. The 4ReusD datasets were derived from previously published studies, and the EHR Uni data come from routine documentation at a university hospital. Access requests should be directed to the authors and are subject to data protection regulations and approval by the data owners.

## Contact

Mahdie Rafiei, Institute for Computational Systems Biomedicine, University of Hamburg.  
mahdie.rafiei@uni-hamburg.de
