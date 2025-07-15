"""Evaluate a set of translations."""
import copy
import jsonlines

from absl import logging
from global_flags import *
from jinja2 import Template
from llm import llm_predict, LLMClient
from typing import Dict, Tuple, Sequence, Set, Union
from utils import load_system_instructions


def _evaluate_translations(
    client: LLMClient,
    model_name: str,
    elt: Dict[str, Union[str, Sequence[str]]],
    system_instructions_template: str,
    log_system_instructions: bool=False,
) -> Dict[str, Union[str, Sequence[str]]]:
  """Evaluates a set of translations.

  Args:
    client: LLM client.
    model_name: LLM name.
    elt: input dictionary of data.
    system_instructions_template: template for system instructions.
    log_system_instructions: whether to print out the system instructions.

  Returns:
    An output dictionary of data.
  """
  source_text = elt["src_text"]
  translators = elt["translators"]
  evaluations = []
  system_instructions = Template(system_instructions_template).render(
    source_language=elt["src_lang_long"],
    target_language=elt["tgt_lang_long"],
  )
  if log_system_instructions:
    logging.info(
      f"evaluate_translations() system_instructions:\n{system_instructions}",
    )
  for j, translator in enumerate(translators):
    translation = elt["translations"][j]
    user_prompt = f"SOURCE TEXT:\n{source_text}\nTRANSLATION:\n{translation}"
    evaluation = llm_predict(
      client,
      model_name,
      system_instructions,
      user_prompt,
    )
    evaluation = (
      evaluation.split("<EVALUATION>")[-1].split("</EVALUATION>")[0].strip()
    )
    evaluations.append(evaluation)
  new_elt = copy.deepcopy(elt)
  new_elt["evaluations"] = evaluations
  return new_elt


def evaluate_translations(
    client: LLMClient,
    model_name: str,
    input_corpus: str,
    output_corpus: str,
):
  """Evaluate a set of translations.

  Args:
    client: LLM client.
    model_name: LLM name.
    input_corpus: path to input corpus.
    output_corpus: path to output corpus.
  """
  system_instructions_template = load_system_instructions(
    TRANSLATION_EVALUATION_INSTRUCTIONS.value,
  )
  log_system_instructions = True
  for elt in jsonlines.open(input_corpus, "r"):
    new_elt = _evaluate_translations(
      client,
      model_name,
      elt,
      system_instructions_template,
      log_system_instructions
    )
    logging.info(
      f'Appending analysis of `{new_elt["src_text"]}` to {output_corpus}',
    )
    with jsonlines.open(output_corpus, "a") as writer:
      writer.write(new_elt)
    log_system_instructions = False
