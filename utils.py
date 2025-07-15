"""Utilities."""
import copy
import random
import re

from global_flags import *
from typing import Dict, Union, Sequence


def _validate(elt: Dict[str, Union[str, Sequence[str]]]) -> bool:
  """Checks that counts of translations, translators, etc. match.

  Args:
     elt: Data dictionary.
  Raises:
     AssertionError
  """
  k1 = "translations"
  for k2 in ["translators", "evaluations"]:
    if k2 not in elt:
      continue
    msg = f'{len([k1])=} != {len([k2])=}'
    assert len([k1]) == len([k2]), msg


def permutations(
    elt: Dict[str, Union[str, Sequence[str]]],
) -> Dict[str, Union[str, Sequence[str]]]:
  """Yields a random permutation followed by n - 1 shifted permutations.

  Args:
     elt: Data dictionary.
  Yields:
     Elements with the translators, translators and evaluations permuted.
  """
  _validate(elt)

  def rearrange(seq, indices):
    return [seq[i] for i in indices]

  indices = list(range(len(elt["translations"])))
  random.shuffle(indices)
  for _ in range(NUM_PERMUTATIONS.value):
    new_elt = copy.deepcopy(elt)
    new_elt["translations"] = rearrange(elt["translations"], indices)
    new_elt["translators"] = rearrange(elt["translators"], indices)
    if "evaluations" in elt:
      new_elt["evaluations"] = rearrange(elt["evaluations"], indices)
    yield new_elt
    indices = indices[1:] + [indices[0]]


COMMENT = re.compile("<!--.*?-->")


def load_system_instructions(
    system_instructions: str,
    verbose: bool=False,
) -> str:
  """Loads system instructions from file.

  Args:
    system_instructions: A path.
    verbose: boolean, whether to print out the instructions.
  Returns:
    A string containing the system instructions.
  """
  with open(system_instructions, encoding="utf-8") as stream:
    instructions = stream.readlines()
    instructions = "".join(instructions)
    instructions = instructions.replace("<INSTRUCTIONS>", "")
    instructions = instructions.replace("</INSTRUCTIONS>", "")
    instructions = COMMENT.sub("", instructions)
    instructions = instructions.strip()
    return instructions
