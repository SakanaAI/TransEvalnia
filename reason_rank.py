"""Reason ranking system for translations."""
import llm

from absl import app
from absl import logging
from evaluate_translations import evaluate_translations
from interleave_evaluations import interleave_evaluations
from global_flags import *
from permute_corpus import permute_corpus
from rank_translations import rank_translations


def main(unused_argv):
  client = llm.client()
  if not SKIP_EVALUATE_TRANSLATIONS.value:
    logging.info("Evaluating translations...")
    evaluate_translations(
      client,
      llm.MODEL.value,
      input_corpus=TRANSLATION_CORPUS.value,
      output_corpus=EVALUATION_CORPUS.value,
    )
  evaluated_corpus = EVALUATION_CORPUS.value
  if PERMUTED_CORPUS.value:
    if not SKIP_PERMUTE_CORPUS.value:
      logging.info("Permuting corpus...")
      permute_corpus(
        EVALUATION_CORPUS.value,
        PERMUTED_CORPUS.value,
      )
    evaluated_corpus = PERMUTED_CORPUS.value
  if INTERLEAVED_EVALUATION_CORPUS.value:
    if not SKIP_INTERLEAVE_EVALUATIONS.value:
      logging.info("Interleaving evaluations...")
      interleave_evaluations(
        client,
        llm.MODEL.value,
        input_corpus=evaluated_corpus,
        output_corpus=INTERLEAVED_EVALUATION_CORPUS.value,
      )
    evaluated_corpus = INTERLEAVED_EVALUATION_CORPUS.value
  if not SKIP_RANKING.value:
    logging.info("Ranking translations...")
    rank_translations(
      client,
      llm.MODEL.value,
      evaluated_corpus,
      RANKED_CORPUS.value,
      interleaved=True if INTERLEAVED_EVALUATION_CORPUS.value else False,
    )


if __name__ == "__main__":
  mark_flags_as_required()
  app.run(main)
