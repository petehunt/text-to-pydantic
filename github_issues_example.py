from pathlib import Path
from pydantic import BaseModel
from llama_cpp import Llama
from enum import Enum
import json
from tqdm import tqdm
import csv

from text_to_pydantic import text_to_pydantic


class GithubIssueType(Enum):
    bug = "bug"
    feature = "feature"
    question = "question"
    other = "other"


class GithubIssueQuality(Enum):
    high_quality = "high_quality"
    medium_quality = "medium_quality"
    low_quality = "low_quality"


class GithubIssueProductArea(Enum):
    user_interface = "user_interface"
    assets_and_scheduling = "assets_and_scheduling"
    backend_infrastructure = "backend_infrastructure"
    other = "other"


class GithubClassificationConfidence(Enum):
    low = "low"
    medium = "medium"
    high = "high"


llama = Llama(
    "mistral-7b-instruct-v0.1.Q5_K_M.gguf",
    n_gpu_layers=35,
    temperature=0,
    n_ctx=4096,
    seed=0,
)


class GithubIssue(BaseModel):
    github_issue_one_line_description: str
    github_issue_type: GithubIssueType
    github_issue_type_confidence: GithubClassificationConfidence


issues = [
    issue
    for issue in json.loads(Path("issues.json").read_text())
    if "pull_request" not in issue
]
texts = [
    f"GitHub issue title: {issue['title']}\n\nGitHub issue body:\n{issue['body'][:8192]}"
    for issue in issues
]

with open("output.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerow(["summary", "type", "confidence", "url"])

    for issue, parsed in zip(issues, text_to_pydantic(llama, GithubIssue, tqdm(texts))):
        print(parsed)
        writer.writerow(
            [
                parsed.github_issue_one_line_description,
                parsed.github_issue_type.value,
                parsed.github_issue_type_confidence.value,
                issue["html_url"],
            ]
        )
