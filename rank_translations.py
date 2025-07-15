"""Rank translations based on a set of prior evaluations of the translations."""
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

BEST_MATCHER = re.compile(r"[Tt]ranslation ([123456])([*][*])? is (the )?best")
WORST_MATCHER = re.compile(
  r"[Tt]ranslation ([123456])([*][*])? is (the )?worst",
)


def _make_evaluation_prompt(
    evaluations: Union[Sequence[str], str],
    translations: Sequence[str],
) -> str:
  """Construct a user prompt from prior evaluations and translations.

  Args:
    evaluations: Either an evaluation or a sequence of evaluations.
    translations: A sequence of translations.
  Returns:
    A combined prompt.
  """
  prompt = []
  if type(evaluations) is str:
    for i, translation in enumerate(translations):
      prompt.append(f"Translation {i + 1}:\n\n{translation}")
    prompt.append("Interleaved evaluations:")
    prompt.append(evaluations)
  else:
    for i, translation in enumerate(translations):
      prompt.append(f"Translation {i + 1}:\n\n{translation}")
      prompt.append(f"Evaluation {i + 1}:\n\n{evaluations[i]}")
  return "\n\n".join(prompt).strip()


def _rank_translations(
    client: LLMClient,
    model_name: str,
    elt: Dict[str, Union[str, Sequence[str]]],
    system_instructions_template: str,
    interleaved: bool=False,
    log_system_instructions: bool=False,
) -> Dict[str, Union[str, Sequence[str]]]:
  """Ranks a set of translations given their evaluations.

  Args:
    client: LLM client.
    model_name: LLM name.
    elt: input dictionary of data.
    system_instructions_template: template for system instructions.
    interleaved: use interleaved evaluations.
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
  # Use the interleaved evaluations if they got produced, otherwise if there is
  # an **ERROR** we still back off to the non-interleaved evaluations...
  if interleaved:
    evaluations = elt["interleaved_evaluations"]
    if evaluations == "**ERROR**":
      logging.info(
        "Interleaved evaluations are **ERROR**, backing off to evaluations",
      )
      evaluations = elt["evaluations"]
  else:
    evaluations = elt["evaluations"]
  user_prompt = _make_evaluation_prompt(evaluations, translations)
  ranking_evaluation = llm_predict(
    client,
    model_name,
    system_instructions,
    user_prompt,
    max_tokens=4096 if interleaved else 2048,
  )
  matcher = BEST_MATCHER.search(ranking_evaluation)
  if matcher:
    best_translator = translators[int(matcher.group(1)) - 1]
  else:
    best_translator = "??"
  matcher = WORST_MATCHER.search(ranking_evaluation)
  if matcher:
    worst_translator = translators[int(matcher.group(1)) - 1]
  else:
    worst_translator = "??"
  new_elt = copy.deepcopy(elt)
  new_elt["ranking_evaluation"] = ranking_evaluation
  new_elt["best_translator"] = best_translator
  new_elt["worst_translator"] = worst_translator
  return new_elt


def rank_translations(
    client: LLMClient,
    model_name: str,
    input_corpus: str,
    output_corpus: str,
    interleaved: bool=False,
) -> None:
  """Rank evaluated translations.

  Args:
    client: LLM client.
    model_name: LLM name.
    input_corpus: path to input corpus.
    output_corpus: path to output corpus.
    interleaved: use interleaved evaluations.
  """
  dataset = list(jsonlines.open(input_corpus))
  if interleaved:
    system_instructions_template = load_system_instructions(
      INTERLEAVED_RANKING_INSTRUCTIONS.value,
    )
  else:
    system_instructions_template = load_system_instructions(
      RANKING_INSTRUCTIONS.value,
    )
  length = len(dataset)
  log_system_instructions = True
  for i, elt in enumerate(dataset):
    logging.info(f"{i}/{length}")
    new_elt = _rank_translations(
      client,
      model_name,
      elt,
      system_instructions_template,
      interleaved,
      log_system_instructions,
    )
    logging.info(
      f'Appending analysis of `{new_elt["src_text"]}` to {output_corpus}',
    )
    with jsonlines.open(output_corpus, "a") as writer:
      writer.write(new_elt)
    log_system_instructions = False
