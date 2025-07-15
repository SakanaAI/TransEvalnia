"""Checks the inconsistency of rankings for shuffled translation/eval orders.

Usage, e.g.:

python3 scripts/ordering_bias.py \
    --input sample_data/wmt2021_en_ja_10_eval_ranked_one_step_qwen.jsonl

"""
import jsonlines

from absl import app
from absl import flags
from collections import defaultdict


INPUT = flags.DEFINE_string("input", None, "Input data.")


def main(unused_argv):
  stats = defaultdict(set)
  top_translator_stats = defaultdict(int)
  variants = defaultdict(set)

  # either TransEvalnia's or MT-Ranker's output
  def best(elt):
    mt_ranker = False
    try:
      return elt["best_translator"], mt_ranker
    except KeyError:
      mt_ranker = True
      return elt["mt_ranker_best"], mt_ranker

  with jsonlines.open(INPUT.value) as reader:
    for elt in reader:
      top, mt_ranker = best(elt)
      top_translator_stats[top] += 1
      translations = tuple(elt["translations"])
      sorted_translations = tuple(sorted(elt["translations"]))
      variants[elt["src_text"], sorted_translations].add(translations)
      stats[elt["src_text"], sorted_translations].add(top)
  inconsistency = sum([len(stats[e]) for e in stats]) / len(stats)
  div = "-" * 80
  # Hack this since mt_ranker permutations are done in code rather than in the
  # JSONL.
  if mt_ranker:
    nvariants = 2
  else:
    nvariants = max([len(variants[v]) for v in variants])
  print(div)
  print(f"Overall inconsistency: {inconsistency:0.2f}\t/{nvariants}")
  print(div)
  print(f"Count\tBest system")
  for key, value in sorted(
      top_translator_stats.items(),
      key=lambda x: x[1],
      reverse=True,
  ):
    print(f"{value}\t{key}")
  print(div)


if __name__ == "__main__":
  flags.mark_flag_as_required("input")
  app.run(main)
