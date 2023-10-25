from pathlib import Path
from typing import List
from pydantic import BaseModel
from llama_cpp import Llama

from text_to_pydantic import text_to_pydantic

class Person(BaseModel):
  person_name: str
  person_age: int

class Response(BaseModel):
  people_mentioned_in_text: List[Person]

llm = Llama(
  "../llama.cpp/models/mistral-7b-instruct-v0.1.Q5_K_M.gguf",
  n_gpu_layers=35,
  temperature=0,
  seed=0)

response = text_to_pydantic(
  llm,
  Response,
  """
    i am barney rubble, my age is 57, and i have lived in bedrock for 2 years.
    my wife is betty and she has the same last name as me, and is 2 years younger than me
  """
)
print(response)
assert response == Response(
  people_mentioned_in_text=[
    Person(person_name="Barney Rubble", person_age=57),
    Person(person_name='Betty Rubble', person_age=55)
  ]
)
