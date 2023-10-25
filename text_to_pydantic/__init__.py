import json
from llama_cpp import Llama, LlamaGrammar
from .json_schema_to_grammar import SchemaConverter
import weakref

grammars = weakref.WeakKeyDictionary()
def text_to_pydantic(llm: Llama, pydantic_model, text: str, prompt_template=lambda text: f"<s>[INST] Output JSON parsed from the following natural language text:\n\n{text} [/INST] "):
  if pydantic_model not in grammars:
    converter = SchemaConverter({})
    converter.visit(pydantic_model.model_json_schema(), "")
    grammar = LlamaGrammar.from_string(converter.format_grammar())
    grammars[pydantic_model] = grammar
  grammar = grammars[pydantic_model]
  response = llm(
    prompt_template(text),
    grammar=grammar, max_tokens=-1
  )
  return pydantic_model(**json.loads(response['choices'][0]['text']))