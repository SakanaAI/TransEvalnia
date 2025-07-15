"""Score translations given evaluations."""
import llm

from absl import app
from absl import logging
from global_flags import *
from score_translations import score_translations


def main(unused_argv):
  client = llm.client()
  logging.info("Scoring translations...")
  score_translations(
    client,
    llm.MODEL.value,
    EVALUATION_CORPUS.value,
    SCORED_CORPUS.value,
  )


if __name__ == "__main__":
  mark_flags_as_required()
  app.run(main)
