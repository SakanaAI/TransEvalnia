"""Permute elements of the corpus and randomize the corpus."""
import jsonlines
import random

from absl import logging
from global_flags import *
from utils import permutations


def permute_corpus(input_corpus, output_corpus) -> None:
  """Randomizes and permutes a corpus.

  Args:
    input_corpus: path to input corpus.
    output_corpus: path to output corpus.
  """
  corpus = list(jsonlines.open(input_corpus, "r"))
  if RANDOM_SUBSET_SIZE.value > 0:
    random.seed(RANDOM_SUBSET_SIZE.value)
    random.shuffle(corpus)
    corpus = corpus[:RANDOM_SUBSET_SIZE.value]
  else:
    random.seed(len(corpus))
  for elt in corpus:
    if NUM_PERMUTATIONS.value > 0:
      for new_elt in permutations(elt):
        with jsonlines.open(output_corpus, "a") as writer:
          writer.write(new_elt)
    else:
      with jsonlines.open(output_corpus, "a") as writer:
        writer.write(elt)
