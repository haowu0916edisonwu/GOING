# GOING: Generative Mode Switching with Confidence Gain for Real-World RAG Noise Mitigation

![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch 2.1+](https://img.shields.io/badge/PyTorch-2.1%2B-orange)
![License MIT](https://img.shields.io/badge/License-MIT-green)

## Overview
GOING is a training-free dual-gate framework for mitigating retrieval noise in real-world retrieval-augmented generation (RAG). It monitors token-level confidence to decide when a model should remain in closed-book mode, when retrieval should be invoked, and when retrieved evidence should be rejected after generation. In practice, GOING acts as a circuit breaker against semantically deceptive retrieval noise without requiring task-specific fine-tuning.

## Key Features
- Training-free: no task-specific fine-tuning required
- Dual-gate policy: efficiency gate (`τ`) + quality gate (`δ`)
- Architecture-agnostic: works with any autoregressive LLM
- 5-fold cross-validation for automatic threshold calibration

## Method Overview
1. Phase 1: Efficiency Gate. Exit early if closed-book confidence is greater than or equal to `τ`.
2. Phase 2: Retrieval Execution. Retrieve top-k documents for samples that require external evidence.
3. Phase 3: Quality Gate. Accept the RAG answer only if the confidence gain `Δ` meets the acceptance threshold `δ`.

## Results
The best results are shown in **bold** and the second best in <u>underline</u>. Paired bootstrap tests (`B=10,000`) show that GOING significantly outperforms *Prior Judgement* on WebQA (`p < 0.01`) and TruthfulQA (`p < 0.05`), while achieving comparable performance on the remaining datasets.

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
# This stage requires model inference and is the only GPU-heavy step.
python scripts/run_universal_logging.py --data_root data_care/eval

# 4. Stage 2: Run GOING analysis
python scripts/run_going_analysis.py

# Optional: reproduce the prior-judgment baseline
python scripts/run_eval.py

# 5. Run bootstrap significance test
python scripts/run_bootstrap.py --data_path universal_logs
```

## Data Preparation
GOING uses the CARE evaluation dataset with ColBERTv2 retrieval files. No data are bundled in this repository. Download the dataset from the CARE repository and place it under `data_care/eval` with one subdirectory per dataset. The expected layout is documented in [data/README.md](data/README.md).

## Requirements
- Python 3.10+
- PyTorch 2.1+
- GPU with 16GB+ VRAM recommended
- Llama-3-8B-Instruct from Hugging Face (downloaded automatically on first use)

## Reproducibility Notes
- `scripts/run_universal_logging.py` is the only GPU-heavy stage because it runs model inference and writes universal logs.
- `scripts/run_going_analysis.py`, `scripts/run_ablation.py`, and `scripts/run_bootstrap.py` operate on the generated `universal_logs` and can be rerun independently.
- To reproduce the paper numbers, keep the model, prompts, data layout, and decoding settings identical.

## Citation
```bibtex
@article{going2026tist,
  title={GOING: Generative Mode Switching with Confidence Gain for Real-World RAG Noise Mitigation},
  author={Anonymous},
  journal={ACM Transactions on Intelligent Systems and Technology},
  year={2026},
  doi={10.1145/PLACEHOLDER}
}
```

## License
This project is released under the MIT License. See [LICENSE](LICENSE) for details.
