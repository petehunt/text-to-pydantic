from pathlib import Path
from typing import List
from pydantic import BaseModel
from llama_cpp import Llama
from enum import Enum

from text_to_pydantic import LLM, AIBaseModel

class Person(BaseModel):
  person_name: str
  person_age: int

class Response(AIBaseModel):
  people_mentioned_in_text: List[Person]

llm = LLM(
  llama=Llama(
    "../llama.cpp/models/mistral-7b-instruct-v0.1.Q5_K_M.gguf",
    n_gpu_layers=35,
    temperature=0,
    n_ctx=4096,
    seed=0,
  )
)

response = Response.from_natural_language(
  llm,
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

class EntityType(Enum):
  scientist = 'scientist'
  mathematician = 'mathematician'
  artist = 'artist'
  politician = 'politician'
  other = 'other'

class ParsedWikiPage(AIBaseModel):
  wiki_page_entity_name: str
  wiki_page_entity_type: EntityType
  wiki_page_summary: str
  topic_tags: List[str]

parsed_wiki_page = ParsedWikiPage.from_natural_language(llm, """
Howard Walter Florey, Baron Florey OM FRS FRCP (24 September 1898 – 21 February 1968) was an Australian pharmacologist and pathologist who shared the Nobel Prize in Physiology or Medicine in 1945 with Ernst Chain and Sir Alexander Fleming for his role in the development of penicillin.

Although Fleming received most of the credit for the discovery of penicillin, it was Florey and his team at the University of Oxford who made it into a useful and effective drug, ten years after Fleming had abandoned its development. They developed techniques for growing, purifying and manufacturing the drug, tested it for toxicity and efficacy on animals, and carried out the first clinical trials. In 1941, they used it to treat a police constable from Oxford. He started to recover, but subsequently died because Florey was unable, at that time, to make enough penicillin. Later trials in Britain, the United States and North Africa were highly successful.

A graduate of the University of Adelaide, Florey studied at the University of Oxford as a Rhodes Scholar and in the United States on a fellowship from the Rockefeller Foundation. In 1935, he became the director of the Sir William Dunn School of Pathology at Oxford. He assembled a multidisciplinary staff that could tackle major research projects. In addition to his work on penicillin, he researched many other subjects, most notably lysozyme, contraception and cephalosporins. He was involved in the founding of the Australian National University in Canberra and the establishment of its John Curtin School of Medical Research, and he served as Chancellor of the Australian National University from 1965 until his death in 1968. He was elected a Fellow of the Royal Society in 1941, and as its president from 1960 to 1965, he oversaw its move to new accommodations at Carlton House Terrace and the establishment of links with European organisations. In 1962, he became provost of The Queen's College, Oxford.

Florey's discoveries are estimated to have saved over 80 million lives, and he is regarded by the Australian scientific and medical community as one of its greatest figures. Australian Prime Minister Sir Robert Menzies said, "In terms of world well-being, Florey was the most important man ever born in Australia."
""")
print(parsed_wiki_page)
assert parsed_wiki_page.wiki_page_entity_name == 'Howard Walter Florey'
assert parsed_wiki_page == ParsedWikiPage(
  wiki_page_entity_name='Howard Walter Florey',
  wiki_page_entity_type=EntityType.scientist,
  wiki_page_summary='Howard Walter Florey was an Australian pharmacologist and pathologist who shared the Nobel Prize in Physiology or Medicine in 1945 with Ernst Chain and Sir Alexander Fleming. He made penicillin into a useful and effective drug, ten years after it was discovered by Fleming, and carried out the first clinical trials. Florey researched many other subjects, including lysozyme, contraception and cephalosporins. He founded the Australian National University in Canberra and served as its Chancellor until his death. His discoveries are estimated to have saved over 80 million lives, and he is regarded by the Australian scientific and medical community as one of its greatest figures.',
  topic_tags=['Nobel Prize', 'Penicillin', 'Howard Florey', 'Medical Research'],
)