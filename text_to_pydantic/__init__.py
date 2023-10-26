import json
from llama_cpp import Llama, LlamaGrammar
from .json_schema_to_grammar import SchemaConverter
import weakref
from typing import Self, Callable
from pydantic import BaseModel, ConfigDict

class LLM(BaseModel):
  model_config = ConfigDict(arbitrary_types_allowed=True)

  llama: Llama
  # different LLMs (vicuna, llama, mistral etc) use different instruction tuning formats
  prompt_template: Callable[[str], str] = lambda instruction: f"<s>[INST] {instruction} [/INST] "

grammars = weakref.WeakKeyDictionary()

class AIBaseModel(BaseModel):
  @classmethod
  def from_natural_language(cls, llm: LLM, text: str, instruction_template=lambda text, json_schema: f"Output JSON parsed from natural language text that adheres to the provided JSON schema.\n\nJSON schema:\n\n{json_schema()}\n\nNatural language text:\n\n{text}") -> Self:
    if cls not in grammars:
      converter = SchemaConverter({})
      json_schema = cls.model_json_schema()
      converter.visit(json_schema, "") # type: ignore
      grammar = LlamaGrammar.from_string(converter.format_grammar())
      grammars[cls] = grammar
    grammar = grammars[cls]
    response = llm.llama(
      # the custom instruction_template lets you customize the instructions to the model for this
      # particular instance. usually, the default instruction_template is fine
      llm.prompt_template(instruction_template(text, lambda: json.dumps(cls.model_json_schema()))),
      grammar=grammar, max_tokens=-1
    )
    return cls(**json.loads(response['choices'][0]['text'])) # type: ignore
