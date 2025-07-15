"""Reason ranking system for translations.

This version does the translation evaluation and ranking in one step.
"""
import llm

from absl import app
from absl import logging
from global_flags import *
from rank_translations_one_step import rank_translations_one_step


def main(unused_argv):
  client = llm.client()
  logging.info("Evaluating and ranking translations...")
  rank_translations_one_step(
    client,
    llm.MODEL.value,
    TRANSLATION_CORPUS.value,
    RANKED_CORPUS.value,
  )


if __name__ == "__main__":
  mark_flags_as_required()
  app.run(main)
