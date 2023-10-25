import json
from llama_cpp import Llama, LlamaGrammar
from .json_schema_to_grammar import SchemaConverter
import weakref
from typing import Self, Callable
from pydantic import BaseModel, ConfigDict

class LLM(BaseModel):
  model_config = ConfigDict(arbitrary_types_allowed=True)

  llama: Llama
  prompt_template: Callable[[str], str] = lambda text: f"<s>[INST] Output JSON parsed from the following natural language text:\n\n{text} [/INST] "

grammars = weakref.WeakKeyDictionary()

class AIBaseModel(BaseModel):
  @classmethod
  def from_natural_language(cls, llm: LLM, text: str) -> Self:
    if cls not in grammars:
      converter = SchemaConverter({})
      converter.visit(cls.model_json_schema(), "") # type: ignore
      grammar = LlamaGrammar.from_string(converter.format_grammar())
      grammars[cls] = grammar
    grammar = grammars[cls]
    response = llm.llama(
      llm.prompt_template(text),
      grammar=grammar, max_tokens=-1
    )
    return cls(**json.loads(response['choices'][0]['text'])) # type: ignore
