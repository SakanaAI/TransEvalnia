"""Download datasets used in the TransEvalnia paper.

Usage, e.g.:

python3 scripts/download_dataset.py \
    --output_dir data/

"""
from collections import defaultdict
import datasets
import jsonlines
import os

from absl import app
from absl import flags


OUTPUT_DIR = flags.DEFINE_string(
  "output_dir",
  "data/",
  "Directory to save the downloaded datasets."
)


def main(unused_argv):
  data = datasets.load_dataset(
    "SakanaAI/TransEvalnia",
    "with_human_ranking",
    split="test"
  )

  dataset_name2dataset = defaultdict(list)
  for item in data:
    dataset_name = item["dataset"]
    dataset_name = dataset_name.replace(" ", "_").lower()
    dataset_name2dataset[dataset_name].append({
      "src_text": item["src_text"],
      "translations": item["tgt_texts"],
      "translators": ["reference", "other"],
      "human_scores": {"reference": item["human_scores"][0], 
                       "other": item["human_scores"][1]},
      "src_lang": item["src_lang"],
      "tgt_lang": item["tgt_lang"],
      "src_lang_long": item["src_lang_long"],
      "tgt_lang_long": item["tgt_lang_long"],
    })

  os.makedirs(OUTPUT_DIR.value, exist_ok=True)
  for dataset_name, items in dataset_name2dataset.items():
    output_filepath = os.path.join(OUTPUT_DIR.value, f"{dataset_name}.jsonl")
    print(f"Saving {len(items)} items to {output_filepath}")
    with jsonlines.open(output_filepath, "w") as writer:
      for item in items:
        writer.write(item)


if __name__ == "__main__":
  app.run(main)