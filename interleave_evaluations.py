"""Interleaves a set of evaluations by theme."""
import copy
import jsonlines

from absl import logging
from llm import LLMClient, llm_predict
from global_flags import *
from jinja2 import Template
from typing import Dict, Sequence, Union, Tuple
from utils import load_system_instructions


def _make_evaluation_prompt(
    evaluations: Sequence[str],
    translations: Sequence[str],
) -> str:
  """Generate instructions to split and align evaluations.

  Args:
    evaluations: a sequence of evaluations.
    translations: a sequence of translations.

  Returns:
    A prompt.
  """
  prompt = []
  for i, evaluation in enumerate(evaluations):
    prompt.append("-" * 80)
    prompt.append(f"Translation {i + 1}:")
    prompt.append(translations[i])
    prompt.append(f"Evaluation {i + 1}:")
    prompt.append(evaluation)
  return "\n".join(prompt)


def _interleave_evaluations(
    client: LLMClient,
    model_name: str,
    elt: Dict[str, Union[str, Sequence[str]]],
    system_instructions_template: str,
    log_system_instructions: bool=False,
) -> Dict[str, Union[str, Sequence[str]]]:
  """Interleaves a set of evaluations.

  Args:
    client: LLM client.
    model_name: LLM name.
    elt: input dictionary of data.
    system_instructions_template: template for system instructions.
    log_system_instructions: whether to print out the system instructions.
  Returns:
    An output dictionary of data.
  """
  translators = elt["translators"]
  translations = elt["translations"]
  evaluations = elt["evaluations"]
  system_instructions = Template(system_instructions_template).render(
    source_language=elt["src_lang_long"],
    target_language=elt["tgt_lang_long"],
    number=len(translations),
    source_text=elt["src_text"],
  )
  if log_system_instructions:
    logging.info(
      f"interleave_evaluations() system_instructions:\n{system_instructions}",
    )
  user_prompt = _make_evaluation_prompt(evaluations, translations)
  interleaved_evaluations = llm_predict(
    client,
    model_name,
    system_instructions,
    user_prompt,
    max_tokens=4096,
  )
  new_elt = copy.deepcopy(elt)
  new_elt["interleaved_evaluations"] = interleaved_evaluations
  return new_elt


def interleave_evaluations(
    client: LLMClient,
    model_name: str,
    input_corpus: str,
    output_corpus: str,
) -> None:
  """Interleave evaluations.

  Args:
    client: LLM client.
    model_name: LLM name.
    input_corpus: path to input corpus.
    output_corpus: path to output corpus.
  """
  dataset = list(jsonlines.open(input_corpus))
  system_instructions_template = load_system_instructions(
    INTERLEAVING_INSTRUCTIONS.value,
  )
  length = len(dataset)
  log_system_instructions = True
  for i, elt in enumerate(dataset):
    logging.info(f"{i}/{length}")
    new_elt = _interleave_evaluations(
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
