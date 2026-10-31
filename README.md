# Australian Legal Q&A: Fine-tuning vs RAG

 This project compares retrieval-augmented generation (RAG) and LoRA fine-tuning for answering Australian legal questions with a small model, `google/gemma-3-1b-it`.

## Conditions compared

| Condition | Description |
|---|---|
| **A** | Base model, no context |
| **B** | RAG: hybrid BM25 + dense retrieval (RRF), passages given to the base model |
| **C** | LoRA fine-tuned on QA pairs, no context |
| **D** | LoRA fine-tuned with retrieved context (RAFT-style, with distractors) + RAG at inference |

## Data

[`isaacus/open-australian-legal-qa`](https://huggingface.co/datasets/isaacus/open-australian-legal-qa): 2,124 GPT-4-synthesised QA pairs. Each row includes the source chunk it was generated from, which serves as the known gold passage for retrieval evaluation. Split 80/20 by row (seed 42). Please check the licence on the dataset card and cite it.

## Pipeline

- **Retrieval:** BM25, dense (`BAAI/bge-small-en-v1.5`), hybrid with Reciprocal Rank Fusion, and a cross-encoder reranker (`cross-encoder/ms-marco-MiniLM-L-6-v2`). Evaluated with recall@k and MRR.
- **Fine-tuning:** LoRA (r=16, alpha=32, dropout 0.05) on all attention and MLP projections, 2 epochs, lr 2e-4, cosine schedule. Loss is computed on answer tokens only.
- **Generation:** greedy decoding, max 200 new tokens.
- **Metrics:** ROUGE-L, embedding cosine similarity, latency, new tokens, peak GPU memory.

## Results

### Retrieval (425 test questions)

| Method | Recall@1 | Recall@5 | MRR |
|---|---|---|---|
| BM25 | 0.776 | 0.894 | 0.831 |
| Dense | 0.732 | 0.835 | 0.776 |
| Hybrid (RRF) | 0.798 | 0.896 | 0.837 |
| Hybrid + rerank | 0.715 | 0.795 | 0.749 |

The reranker made retrieval worse here, so it is not used for the final runs.

### Generation (first run, 50 test questions)

| Condition | ROUGE-L | Emb. sim. | Latency (s) | Train time (s) |
|---|---|---|---|---|
| A: base | 0.277 | 0.873 | 9.1 | 0 |
| B: RAG | 0.437 | 0.844 | 3.7 | 0 |
| C: fine-tuned | 0.368 | 0.888 | 12.0 | 196 |
| **D: fine-tuned + RAG** | **0.578** | **0.934** | 9.5 | 612 |

### Part 2: chosen search method, 50 questions

After tuning the search method on training questions (hybrid RRF, weighted 2:1 with prefix), B and D were re-run:

| Condition | ROUGE-L | Emb. sim. | Gold passage retrieved |
|---|---|---|---|
| B2: base + RAG | 0.428 | 0.847 | 88% |
| D2: fine-tuned + RAG | 0.537 | 0.928 | 88% |

95% confidence intervals for ROUGE-L differences (n=50):

| Comparison | Mean diff | 95% CI |
|---|---|---|
| D2 vs B2 | +0.110 | [0.060, 0.158] |
| B2 vs A | +0.151 | [0.100, 0.199] |
| D2 vs C | +0.169 | [0.120, 0.218] |
| D2 vs A | +0.260 | [0.211, 0.310] |

**Unanswerable questions (60 prompts):** the base model with RAG declined 43% of the time when no answer existed (and 10% when one did). The fine-tuned model with RAG declined 0% of the time when no answer existed, so fine-tuning made it more likely to produce an answer when it should have declined.

### Takeaways

- Combining fine-tuning and RAG (D) gave the best answer quality, with gains over every other condition.
- RAG alone gives a large gain over the base model and is the fastest.
- The fine-tuned model is less willing to say "I don't know", which is a risk in legal settings.

### Caveats

- Fine-tuning used only 200 training examples (`N_TRAIN = 200`) and evaluation used 50 questions (`N_EVAL = 50`), so results are indicative rather than definitive.
- Questions and answers are GPT-4-synthesised, not written by lawyers.
- ROUGE-L and embedding similarity are proxy metrics for answer correctness.

## Repository layout

```
.
├── legal_qa_rag_vs_finetune_full.ipynb   # full experiment, run top to bottom
├── requirements.txt
└── results/
    ├── summary.csv, retrieval_results.csv, generation_results.csv
    ├── search_dev.csv, search_test.csv
    ├── answers_all.csv, answers_summary.csv, answers_split_by_gold.csv
    ├── unanswerable.csv, decline_rates.csv
    └── confidence_intervals.csv
```

The trained LoRA adapters (`adapter_C`: no context, `adapter_D`: with retrieved context) are about 52 MB each, so they are hosted separately. Add the link here once uploaded.

## How to run

1. Open the notebook in Google Colab and select a **T4 GPU** runtime.
2. Accept the [Gemma licence](https://huggingface.co/google/gemma-3-1b-it) on Hugging Face and add your token as a Colab secret named `HF_TOKEN`.
3. Run the notebook top to bottom. For a quick test, lower `N_TRAIN` and `N_EVAL` first.

### Loading an adapter

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = "google/gemma-3-1b-it"
tokenizer = AutoTokenizer.from_pretrained(base)
model = AutoModelForCausalLM.from_pretrained(base)
model = PeftModel.from_pretrained(model, "path/to/adapter_D")
```

## Licences and attribution

- Base model: Gemma, subject to the [Gemma Terms of Use](https://ai.google.dev/gemma/terms). The adapters are derivatives of it.
- Dataset: see the dataset card for its licence.
- Add a licence for your own code here (e.g. MIT) if you want others to reuse it.
