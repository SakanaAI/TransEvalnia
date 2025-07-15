"""Flags used by the entire system."""
from absl import flags

TRANSLATION_CORPUS = flags.DEFINE_string(
  "translation_corpus",
  None,
  "Original translation set.",
)

EVALUATION_CORPUS = flags.DEFINE_string(
  "evaluation_corpus",
  None,
  "Set with translation evaluations.",
)

PERMUTED_CORPUS = flags.DEFINE_string(
  "permuted_corpus",
  None,
  "Random subset of corpus, possibly with permutations.",
)

SCORED_CORPUS = flags.DEFINE_string(
  "scored_corpus",
  None,
  "Scored corpus, produced by score_translations.",
)

INTERLEAVED_EVALUATION_CORPUS = flags.DEFINE_string(
  "interleaved_evaluation_corpus",
  None,
  "Set with interleaved translation evaluations.",
)

RANKED_CORPUS = flags.DEFINE_string(
  "ranked_corpus",
  None,
  "Set translation rankings.",
)

RATED_CORPUS = flags.DEFINE_string(
  "rated_corpus",
  None,
  "Rated corpus, produced by rate_translations.",
)

TRANSLATION_EVALUATION_INSTRUCTIONS = flags.DEFINE_string(
  "translation_evaluation_instructions",
  None,
  "Instructions to use for translation evaluation.",
)

INTERLEAVING_INSTRUCTIONS = flags.DEFINE_string(
  "interleaving_instructions",
  None,
  "Instructions to use for interleaving the evaluations.",
)

RANKING_INSTRUCTIONS = flags.DEFINE_string(
  "ranking_instructions",
  None,
  "Instructions to use for ranking the translations.",
)

INTERLEAVED_RANKING_INSTRUCTIONS = flags.DEFINE_string(
  "interleaved_ranking_instructions",
  None,
  "Instructions to use for ranking the translations if evaluations are "
  "interleaved. Mutually exclusive with --ranking_instructions.",
)

ONE_STEP_RANKING_INSTRUCTIONS = flags.DEFINE_string(
  "one_step_ranking_instructions",
  None,
  "Instructions to use for evaluating and ranking the translations.",
)

SCORING_INSTRUCTIONS = flags.DEFINE_string(
  "scoring_instructions",
  None,
  "Instructions to use for scoring a translation given an evaluation.",
)

RATER_INSTRUCTIONS = flags.DEFINE_string(
  "rater_instructions",
  None,
  "Instructions to use for rating a translation given an evaluation.",
)

RANDOM_SUBSET_SIZE = flags.DEFINE_integer(
  "random_subset_size",
  -1,
  "Size of random subset.",
)

NUM_PERMUTATIONS = flags.DEFINE_integer(
  "num_permutations",
  0,
  "Number of permutations."
)

SKIP_EVALUATE_TRANSLATIONS = flags.DEFINE_bool(
  "skip_evaluate_translations",
  False,
  "Skip the translation evaluation phase.",
)

SKIP_PERMUTE_CORPUS = flags.DEFINE_bool(
  "skip_permute_corpus",
  False,
  "Skip the permuting the corpus (use already permuted corpus).",
)

SKIP_INTERLEAVE_EVALUATIONS = flags.DEFINE_bool(
  "skip_interleave_evaluations",
  False,
  "Skip interleaving the evaluations (use already interleaved version).",
)

SKIP_RANKING = flags.DEFINE_bool("skip_ranking", False, "Skip ranking phase.")

_REQUIRED_FLAGS = [
  "evaluation_corpus",
  "translation_corpus",
  "translation_evaluation_instructions",
  "ranked_corpus",
]


def mark_flags_as_required():
  for flag in _REQUIRED_FLAGS:
    flags.mark_flag_as_required(flag)
