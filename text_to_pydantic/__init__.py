import json
from llama_cpp import Llama, LlamaGrammar
from .json_schema_to_grammar import SchemaConverter
from typing import TypeVar, Type, Iterable
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def text_to_pydantic(
    llama: Llama,
    to_model: Type[T],
    texts: Iterable[str],
    prompt=lambda text, json_schema: f"<s>[INST] Output JSON parsed from natural language text that adheres to the provided JSON schema.\n\nJSON schema:\n\n{json_schema}\n\nNatural language text:\n\n{text} [/INST] ",
) -> Iterable[T]:
    converter = SchemaConverter({})
    json_schema = to_model.model_json_schema()
    converter.visit(json_schema, "")  # type: ignore
    grammar = LlamaGrammar.from_string(converter.format_grammar())
    for text in texts:
        response = llama(
            prompt(text, json_schema),
            grammar=grammar,
            max_tokens=-1,
        )
        yield to_model(**json.loads(response["choices"][0]["text"]))
