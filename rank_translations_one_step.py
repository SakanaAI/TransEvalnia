"""Rank translations including doing the evaluation."""
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

MATCHER = re.compile(r"[Tt]ranslation ([123456])([*][*])? is (the )?best")


def _make_evaluation_prompt(translations: Sequence[str]) -> str:
  """Construct a user prompt from prior evaluations and translations.

  Args:
    translations: A sequence of translations.
  Returns:
    A combined prompt.
  """
  prompt = []
  for i, translation in enumerate(translations):
    prompt.append(f"Translation {i + 1}:\n\n{translation}")
  return "\n\n".join(prompt).strip()


def _rank_translations_one_step(
    client: LLMClient,
    model_name: str,
    elt: Dict[str, Union[str, Sequence[str]]],
    system_instructions_template: str,
    log_system_instructions: bool=False,
) -> Dict[str, Union[str, Sequence[str]]]:
  """Ranks a set of translations given their evaluations.

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
  system_instructions = Template(system_instructions_template).render(
    source_language=elt["src_lang_long"],
    target_language=elt["tgt_lang_long"],
    number=len(translations),
    source_text=elt["src_text"],
  )
  if log_system_instructions:
    logging.info(
      f"rank_translations() system_instructions:\n{system_instructions}",
    )
  user_prompt = _make_evaluation_prompt(translations)
  ranking_evaluation = llm_predict(
    client,
    model_name,
    system_instructions,
    user_prompt,
    max_tokens=2048,
  )
  matcher = MATCHER.search(ranking_evaluation)
  if matcher:
    best_translator = translators[int(matcher.group(1)) - 1]
  else:
    best_translator = "??"
  new_elt = copy.deepcopy(elt)
  # So as not to break downstream tools, just zero these out, but keep the lists
  # the same length:
  if "evaluations" in new_elt:
    new_elt["evaluations"] = [""] * len(new_elt["evaluations"])
  if "interleaved_evaluations" in new_elt:
    del new_elt["interleaved_evaluations"]
  new_elt["ranking_evaluation"] = ranking_evaluation
  new_elt["best_translator"] = best_translator
  return new_elt


def rank_translations_one_step(
    client: LLMClient,
    model_name: str,
    input_corpus: str,
    output_corpus: str,
) -> None:
  """Rank evaluated translations.

  Args:
    client: LLM client.
    model_name: LLM name.
    input_corpus: path to input corpus.
    output_corpus: path to output corpus.
  """
  dataset = list(jsonlines.open(input_corpus))
  system_instructions_template = load_system_instructions(
    ONE_STEP_RANKING_INSTRUCTIONS.value,
  )
  length = len(dataset)
  log_system_instructions = True
  for i, elt in enumerate(dataset):
    logging.info(f"{i}/{length}")
    new_elt = _rank_translations_one_step(
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
