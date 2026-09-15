# GOING: Generative Mode Switching with Confidence Gain for Real-World RAG Noise Mitigation

![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch 2.1+](https://img.shields.io/badge/PyTorch-2.1%2B-orange)
![License MIT](https://img.shields.io/badge/License-MIT-green)

**Authors:** Hao Wu<sup>1</sup>, Sipeng Chen<sup>1</sup>, Tianyi Chen<sup>1</sup>, Qiancheng Yang<sup>1</sup>, Hang Zhou<sup>2</sup>, and Xiao Luo<sup>1</sup>.

<sup>1</sup> University of Wisconsin-Madison, USA.<br>
<sup>2</sup> University of North Carolina at Chapel Hill, USA.

## Overview
GOING is a training-free dual-gate framework for mitigating retrieval noise in real-world retrieval-augmented generation (RAG). It monitors token-level confidence to decide when a model should remain in closed-book mode, when retrieval should be invoked, and when retrieved evidence should be rejected after generation. In practice, GOING acts as a circuit breaker against semantically deceptive retrieval noise without requiring task-specific fine-tuning.

## Key Features
- Training-free: no task-specific fine-tuning required
- Dual-gate policy: efficiency gate (`τ`) + quality gate (`δ`)
- Uses generated-token probabilities without additional models or task-specific fine-tuning
- 5-fold cross-validation for automatic threshold calibration

## Method Overview
The ConfDelta policy uses three phases:

1. Phase 1: Efficiency Gate. Exit early if closed-book confidence is greater than or equal to `τ`.
2. Phase 2: Retrieval Execution. Retrieve top-k documents for samples that require external evidence.
3. Phase 3: Quality Gate. Accept the RAG answer only if `Δ = s_rag - s_cb > δ` and the answer passes the dataset-specific refusal check; otherwise, return the closed-book answer.

ConfThreshold accepts the RAG answer when `s_rag >= τ` and it passes the refusal check; otherwise, it returns the closed-book answer. This variant evaluates RAG confidence after generation and has no pre-retrieval early exit.

The manuscript reports the following selected strategies and thresholds:

| Dataset | Strategy | τ | δ |
|---------|----------|---|---|
| NQ | ConfThreshold | 0.604 | -- |
| TriviaQA | ConfDelta | 0.890 | -0.100 |
| WebQA | ConfDelta | 0.740 | 0.060 |
| TruthfulQA | ConfDelta | 0.740 | 0.050 |
| FactKG | ConfThreshold | 0.500 | -- |

## Answer Processing
Refusal checks are applied case-insensitively after dataset-specific answer cleaning. WebQA additionally requires at least eight whitespace-separated words. For TruthfulQA, GOING scores its closed-book output using the cleaned text before the first period, or the full cleaned text if no period occurs; the RAG branch uses `verbose_clean` without this period truncation. The complete cleaning functions, refusal phrases, and dataset configuration are defined in [scripts/run_going_analysis.py](scripts/run_going_analysis.py).

## Results
The table below reports manuscript results, with the best results in **bold** and the second best in <u>underline</u>. The manuscript reports paired bootstrap significance (`B=10,000`) against *Prior Judgment* on WebQA (`p < 0.01`) and TruthfulQA (`p < 0.05`). The bundled bootstrap script uses a separate out-of-fold evaluation protocol, described under [Reproducibility Notes](#reproducibility-notes); reproduction of these reported p-values has not been verified.

| Dataset | Metric | Closed Book<br>(CB) | Std RAG<br>(Top-10) | Semantic<br>(Reactive) | Prior Judgement<br>(Ren et al. 2025) | Naive<br>(MaxConf) | Iterative<br>(Stepwise) | GOING<br>(Ours) |
|---------|--------|---------------------|---------------------|------------------------|--------------------------------------|--------------------|-------------------------|-----------------|
| NQ | Span EM | 0.3806 | 0.3911 | 0.3817 | <u>0.4457</u> | 0.4141 | 0.3820 | **0.4474** |
| TriviaQA | Span EM | 0.6504 | 0.6183 | 0.6504 | <u>0.6986</u> | 0.6546 | 0.6540 | **0.7021** |
| WebQA | Span EM | <u>0.5030</u> | 0.2972 | <u>0.5030</u> | 0.4341 | 0.4616 | 0.4380 | **0.5034** |
| TruthfulQA | F1 | 0.2671 | 0.2424 | 0.2667 | <u>0.2682</u> | 0.2503 | 0.2590 | **0.2732** |
| TruthfulQA | ROUGE-L | <u>0.2512</u> | 0.2177 | 0.2508 | 0.2478 | 0.2444 | 0.2400 | **0.2561** |
| FactKG | Accuracy | 0.6458 | 0.2794 | 0.6459 | **0.6895** | 0.4408 | 0.6459 | <u>0.6893</u> |

## Quick Start
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Prepare data (see data/README.md)

# 3. Stage 1: Generate universal logs
# This offline stage runs model inference for every configured context level.
python scripts/run_universal_logging.py --data_root data_care/eval

# 4. Stage 2: Run GOING analysis
python scripts/run_going_analysis.py

# Optional: run the standalone prior-judgment evaluator (also runs inference)
python scripts/run_eval.py

# 5. Run the bundled ablation analysis (see Reproducibility Notes)
python scripts/run_ablation.py --data_path universal_logs

# 6. Run paired bootstrap on out-of-fold scores
python scripts/run_bootstrap.py --data_path universal_logs
```

## Data Preparation
GOING uses the CARE evaluation dataset with ColBERTv2 retrieval files. No data are bundled in this repository. Download the dataset from the CARE repository and place it under `data_care/eval` with one subdirectory per dataset. The expected layout is documented in [data/README.md](data/README.md).

## Requirements
- Python 3.10+
- PyTorch 2.1+
- GPU with 16GB+ VRAM recommended
- Llama-3-8B-Instruct from Hugging Face (downloaded automatically on first use)

Use Python 3.10 or 3.11 for the pinned dependencies in `requirements.txt`. The default checkpoint is `NousResearch/Meta-Llama-3-8B-Instruct`; pass `--model_name` to either inference script to select a checkpoint explicitly.

## Reproducibility Notes
- `scripts/run_universal_logging.py` precomputes all configured context levels from the supplied retrieval files. It does not execute online retrieval or early exit. Its context levels currently select newline-separated portions of the loaded context; document boundaries must be checked before interpreting a level as a document count. The JSON key `gears` is retained for compatibility with existing logs.
- Both `scripts/run_universal_logging.py` and `scripts/run_eval.py` run model inference. The latter is a standalone baseline evaluator with its own historical reference targets.
- `scripts/run_going_analysis.py`, `scripts/run_ablation.py`, and `scripts/run_bootstrap.py` operate on the generated `universal_logs` and can be rerun independently.
- Main analysis averages the thresholds selected on the training portions of five folds, evaluates those averaged thresholds on all samples, and selects the better strategy by that full-data score. Bootstrap instead selects a strategy and thresholds within each training fold and tests the resulting held-out scores. These produce different score vectors and should be reported as separate protocols.
- The bundled ablation uses its own strategy search, cleans the log's `pred` field rather than `raw`, and uses `>=` for the gain condition. Its threshold-only strategy checks closed-book confidence, whereas the main analysis's ConfThreshold checks RAG confidence. Its correspondence to manuscript Table II remains to be verified; `RETR%` in its output counts retrieval triggers.
- In the main analysis JSON, `semantic` denotes Semantic (Reactive), and `v25` (or `heuristic` in `final_comparison`) denotes Prior Judgment. The final comparison summarizes each dataset's primary metric, using F1 for TruthfulQA.
- Preserve the model checkpoint, prompts, data, decoding settings, answer processing, and calibration protocol when comparing runs. Offline routing results do not measure end-to-end latency savings.

## Citation
```bibtex
@unpublished{wu2026going,
  title={GOING: Generative Mode Switching with Confidence Gain for Real-World RAG Noise Mitigation},
  author={Wu, Hao and Chen, Sipeng and Chen, Tianyi and Yang, Qiancheng and Zhou, Hang and Luo, Xiao},
  year={2026},
  note={Manuscript}
}
```

## License
This project is released under the MIT License. See [LICENSE](LICENSE) for details.
