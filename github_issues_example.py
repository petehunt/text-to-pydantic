from pathlib import Path
from typing import List
from pydantic import BaseModel
from llama_cpp import Llama
from enum import Enum
import json

from text_to_pydantic import LLM, AIBaseModel

class GithubIssueType(Enum):
  bug = "bug"
  feature = "feature"
  other = "other"

class GithubIssueQuality(Enum):
  high_quality = "high_quality"
  medium_quality = "medium_quality"
  low_quality = "low_quality"

class GithubIssueTeam(Enum):
  frontend = "frontend"
  core_apis = "core_apis"
  backend_infrastructure = "backend_infrastructure"
  other = "other"



llm = LLM(
  llama=Llama(
    "mistral-7b-instruct-v0.1.Q5_K_M.gguf",
    n_gpu_layers=35,
    temperature=0,
    n_ctx=4096,
    seed=0,
  )
)

class GithubIssue(AIBaseModel):
  github_issue_one_line_description: str
  github_issue_type: GithubIssueType
  github_issue_quality: GithubIssueQuality
  github_issue_triaged_team: GithubIssueTeam


for issue in json.loads(Path("issues.json").read_text()):
  if "pull_request" in issue:
    continue
  text = f"Title: {issue['title']}\n\nBody:\n{issue['body']}"
  parsed = GithubIssue.from_natural_language(llm, text)
  #print("text:")
  #print(text)
  #print("parsed:")
  print(parsed)