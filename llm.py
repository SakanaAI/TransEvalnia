"""Interface to various LLMs.

For OpenAI an OpenAI key is required.  Access to Anthropic's models is assumed
to be via Amazon Bedrock. Other models are assumed to have been installed
locally from HuggingFace.
"""
import botocore
import boto3
import random
import re
import transformers
import torch

from absl import flags
from absl import logging

from botocore.config import Config
from botocore.exceptions import ClientError
from openai import OpenAI
from time import sleep
from typing import Callable, Union
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

CLAUDE_WAIT_TIME = flags.DEFINE_integer(
  "claude_wait_time",
  5,
  "Number of seconds to wait if Claude is busy",
)
MODEL = flags.DEFINE_enum(
  "model",
  "claude",
  ["gpt-4o-mini", "gpt-4-1106-preview", "claude", "qwen", "qwen32", "llama"],
  "Model to use for eval",
)
OPEN_AI_API_KEY = flags.DEFINE_string(
  "open_ai_api_key",
  None,
  "Your OpenAI API key",
)
PARALLEL_SIZE = flags.DEFINE_integer(
  "parallel_size",
  4,
  "Parallel size"
)

TEMPERATURE = flags.DEFINE_float("temperature", 0.0, "LLM temperature")

QWEN_PATH = "Qwen/Qwen2.5-72B-Instruct-GPTQ-Int4"

QWEN32_PATH = "Qwen/QwQ-32B-Preview"

LLAMA_PATH = "unsloth/Llama-3.3-70B-Instruct-bnb-4bit"

LLMClient = Union[LLM, OpenAI, botocore.client.BaseClient, Callable]


def llm_predict(
    client: LLMClient,
    model_name: str,
    system_instructions: str,
    user_prompt: str,
    max_tokens: int=2048,
) -> str:
  """Run the LLM to predictg given the instructions and user prompt.

  Args:
    client: LLM client.
    model_name: Name of the model.
    system_instructions: System-level instructions.
    user_prompt: A user prompt.
    max_tokens: Maximum number of tokens for the LLM.
  Returns:
    The LLM response as a string.
  """
  if model_name == "claude":
    model_id = "us.anthropic.claude-3-5-sonnet-20240620-v1:0"
    resource_arn = f"arn:aws:bedrock:us-east-1::foundation-model/{model_id}"
    instructions = system_instructions + "\n" + user_prompt
    messages = [
      {
        "role": "user",
        "content": [{"text": instructions}],
      },
    ]

    def bad_response(exception):
      if "ServiceUnavailableException" in str(exception):
        return True
      if "Read timeout on endpoint URL" in str(exception):
        return True
      if "An error occurred" in str(exception):
        return True
      return False

    def wrap_claude_call(messages, model_id, max_tokens):
      while True:
        try:
          response = client.converse(
            modelId=model_id,
            messages=messages,
            inferenceConfig={
              "maxTokens": max_tokens,
              "temperature": TEMPERATURE.value,
              "topP": 1.0,
            },
          )
          return response["output"]["message"]["content"][0]["text"]
        except (ClientError, Exception) as e:
          logging.info(
            f"ERROR: Can't invoke '{model_id}'. Reason: {e}",
          )
          if bad_response(e):
            logging.info(
              f"Retrying in {CLAUDE_WAIT_TIME.value}",
            )
            sleep(CLAUDE_WAIT_TIME.value)
          else:
            return "**ERROR**"

    return wrap_claude_call(messages, model_id, max_tokens)
  elif (
      model_name in ("qwen", "qwen32", "llama") or
      (QWEN_LOCAL_DIR.value and LORA_FINETUNED_MODEL_DIR.value)
  ):
    tokenizer = client.get_tokenizer()
    sampling_params = SamplingParams(
      temperature=TEMPERATURE.value,
      top_p=0.95,
      max_tokens=max_tokens,
      n=1,
      seed=1,
    )
    messages = [
      tokenizer.apply_chat_template(
        [
          {
            "role": "user",
            "content": system_instructions + "\n" + user_prompt,
          }
        ],
        add_generation_prompt=True,
        tokenize=False,
      )
    ]
    llm_output = client.generate(messages, sampling_params=sampling_params)
    for completion in llm_output:
      return completion.outputs[0].text
  else:  # GPT
    messages = [
      {
        "role": "system",
        "content": system_instructions,
      },
    ]
    messages.append(
      {
        "role": "user",
        "content": user_prompt,
      },
    )
    response = client.chat.completions.create(
      model=model_name,
      messages=messages,
      temperature=TEMPERATURE.value,
      max_tokens=max_tokens,
      top_p=1,
      frequency_penalty=0,
      presence_penalty=0,
    )
    return response.choices[0].message.content


def client() -> LLMClient:
  """Set up the LLM client.

  Returns:
    The LLM client depending on the name of the model.
  """
  if MODEL.value == "claude":
    config = Config(read_timeout=1000)
    return boto3.client(
      service_name="bedrock-runtime",
      region_name="us-east-1",
      config=config,
    )
  elif MODEL.value == "qwen":
    return LLM(model=QWEN_PATH, quantization="gptq_marlin")
  elif MODEL.value == "qwen32":
    return LLM(model=QWEN32_PATH, max_model_len=16384)
  elif MODEL.value == "llama":
    return LLM(
      model=LLAMA_PATH,
      dtype=torch.bfloat16,
      quantization="bitsandbytes",
      load_format="bitsandbytes",
      max_model_len=65536,
    )
  else:  # GPT
    return OpenAI(api_key=OPEN_AI_API_KEY.value)
