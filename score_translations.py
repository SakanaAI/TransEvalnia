"""Score a translation along the various evaluation dimensions."""
import jsonlines
import copy
import re

from absl import flags
from absl import logging
from global_flags import *
from jinja2 import Template
from llm import llm_predict, LLMClient
from typing import Dict, Sequence, Tuple, Union
from utils import load_system_instructions


def _make_scoring_prompt(translation, evaluation) -> str:
  """Construct a user prompt from translation and evaluation.

  Args:
    translation: A translation.
    evaluation: An evaluation.
  Returns:
    A combined prompt.
  """
  prompt = [
    f"Translation:\n\n{translation}",
    f"Evaluation:\n\n{evaluation}",
  ]
  return "\n\n".join(prompt).strip()


def _score_translations(
    client: LLMClient,
    model_name: str,
    elt: Dict[str, Union[str, Sequence[str]]],
    system_instructions_template: str,
    log_system_instructions: bool=False,
) -> Dict[str, Union[str, Sequence[str]]]:
  """Scores a translation given its evaluations.

  Args:
    client: LLM client.
    model_name: LLM name.
    elt: input dictionary of data.
    system_instructions_template: template for system instructions.
    log_system_instructions: whether to print out the system instructions.
  Returns:
    An output dictionary of data.
  """
  system_instructions = Template(system_instructions_template).render(
    source_language=elt["src_lang_long"],
    target_language=elt["tgt_lang_long"],
    source_text=elt["src_text"],
  )
  if log_system_instructions:
    logging.info(
      f"rank_translations() system_instructions:\n{system_instructions}",
    )
  scores = []
  for i, translation in enumerate(elt["translations"]):
    evaluation = elt["evaluations"][i]
    user_prompt = _make_scoring_prompt(translation, evaluation)
    score_evaluation = llm_predict(
      client,
      model_name,
      system_instructions,
      user_prompt,
      max_tokens=2048,
    )
    score = score_evaluation.split(
      "<EVALUATION>",
    )[-1].split("</EVALUATION>")[0].strip()
    scores.append(score)
  new_elt = copy.deepcopy(elt)
  new_elt["scores"] = scores
  return new_elt


def score_translations(
    client: LLMClient,
    model_name: str,
    input_corpus: str,
    output_corpus: str,
) -> None:
  """Score evaluated translations.

  Args:
    client: LLM client.
    model_name: LLM name.
    input_corpus: path to input corpus.
    output_corpus: path to output corpus.
  """
  dataset = list(jsonlines.open(input_corpus))
  system_instructions_template = load_system_instructions(
    SCORING_INSTRUCTIONS.value,
  )
  length = len(dataset)
  log_system_instructions = True
  for i, elt in enumerate(dataset):
    logging.info(f"{i}/{length}")
    new_elt = _score_translations(
      client,
      model_name,
      elt,
      system_instructions_template,
      log_system_instructions,
    )
    logging.info(
      f'Appending analysis of `{new_elt["src_text"]}` to {output_corpus}',
    )
    with jsonlines.open(output_corpus, "a") as writer:
      writer.write(new_elt)
    log_system_instructions = False
