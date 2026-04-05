# GOING: Generative Mode Switching with Confidence Gain for Real-World RAG Noise Mitigation

![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch 2.1+](https://img.shields.io/badge/PyTorch-2.1%2B-orange)
![License MIT](https://img.shields.io/badge/License-MIT-green)

## Overview
GOING is a training-free dual-gate framework for mitigating retrieval noise in real-world retrieval-augmented generation (RAG). It monitors token-level confidence to decide when a model should stay in closed-book mode, when it should consult retrieval, and when retrieved evidence should be rejected. In practice, GOING acts as a circuit breaker against semantically deceptive retrieval noise without requiring task-specific fine-tuning.

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
| Dataset | Metric | Closed Book | Std RAG | Prior Judgment | GOING |
|---------|--------|-------------|---------|----------------|-------|
| NQ | Span EM | 0.3806 | 0.3911 | 0.4457 | **0.4474** |
| TriviaQA | Span EM | 0.6504 | 0.6183 | 0.6986 | **0.7021** |
| WebQA | Span EM | 0.5030 | 0.2972 | 0.4341 | **0.5034** |
| TruthfulQA | F1 | 0.2671 | 0.2424 | 0.2682 | **0.2732** |
| FactKG | Accuracy | 0.6458 | 0.2794 | 0.6895 | **0.6893** |

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
