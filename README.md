# TransEvalnia

TransEvalnia is a prompting-based translation evaluation and ranking system that
uses reasoning in performing its evaluations and ranking. The system presents
fine-grained evaluations based on a subset of the Multidimensional Quality
Metrics ([https://themqm.org/](https://themqm.org/)), returns an assessment of
which translation it deems the best, and provides numerical scores for the
various dimensions and for the overall translation. TransEvalnia performs as
well as or better than the state-of-the-art MT-Ranker ([Moosa et al.,
2024](https://arxiv.org/abs/2401.17099)) on our own English∼Japanese data as
well as several language pairs from various WMT shared tasks. Using Anthropic's
Claude-3.5-Sonnet and Qwen-2.5-72B-Instruct as the evaluation LLMs, the
evaluations returned are deemed highly acceptable to human raters, and the
scores assigned to the translations by Sonnet, as well as other LLMs, correlate
well with scores assigned by the human raters.

# Publication

Richard Sproat, Tianyu Zhao and Llion Jones. 2025. "TransEvalnia:
Reasoning-based Evaluation and Ranking of Translations."
[arXiv XXXX.XXXXX](https://arxiv.org/abs/XXXX.XXXXX).

# Installation

`pip install -r requirements.txt`

# Sample script

A sample of ten lines from the WMT-2021 English-Japanese dataset can be found in `sample_data`.

The script `scripts/sample_script.sh` does a complete run of all variants of the
system: one-step evaluation and ranking; two-step evaluation, followed by
ranking; three-step evaluation, interleaving and ranking; as well as LLM-based
scoring of the translations (which can then be ranked by score).

# Computing ordering bias

See the script `scripts/ordering_bias.py`

# Data from HuggingFace

The main set of data can be obtained at
[https://huggingface.co/datasets/SakanaAI/TransEvalnia](https://huggingface.co/datasets/SakanaAI/TransEvalnia).

# Other tools

After downloading the HuggingFace data:

`scripts/scoring.py` scores different system outputs.

`scripts/hand_labeled.py` computes statistics on the hand-labeled data.
