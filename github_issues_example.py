from pathlib import Path
from pydantic import BaseModel
from llama_cpp import Llama
from enum import Enum
import json
from tqdm import tqdm
import csv
import random

from text_to_pydantic import text_to_pydantic


class GithubIssueType(Enum):
    bug = "bug"
    feature_request = "feature_request"
    unknown = "unknown"


llama = Llama(
    "mistral-7b-instruct-v0.1.Q5_K_M.gguf",
    n_gpu_layers=35,
    temperature=0,
    n_ctx=4096,
    seed=0,
)


def issue_data_to_text(issue_data):
    return f"GitHub issue title: {issue_data['title']}\n\nGitHub issue body:\n{issue_data['body'][:8192]}"


def issue_data_to_github_issue_type(issue_data):
    label_names = set(label["name"] for label in issue_data["labels"])
    if "type: bug" in label_names:
        return GithubIssueType.bug

    elif "type: feature-request" in label_names:
        return GithubIssueType.feature_request

    else:
        return GithubIssueType.unknown


def load_issue_data_from_filename(filename):
    raw_data = json.loads(Path(filename).read_text())
    return [
        issue
        for issue in raw_data
        if "pull_request" not in issue and issue["body"] and len(issue["labels"]) > 0
    ]


def load_issue_data_from_directory(path):
    for filename in Path(path).glob("*"):
        yield from load_issue_data_from_filename(filename)


class GithubIssue(BaseModel):
    github_issue_type: GithubIssueType
    # github_issue_type_confidence: GithubClassificationConfidence
    # github_issue_quality: GithubIssueQuality
    # github_issue_quality_confidence: GithubClassificationConfidence
    # github_issue_product_area: GithubIssueProductArea
    # github_issue_product_area_confidence: GithubClassificationConfidence


random.seed(0)

train_issues = list(load_issue_data_from_directory("issues_train"))
test_issues = list(load_issue_data_from_directory("issues_test"))

random.shuffle(train_issues)
random.shuffle(test_issues)

train_issues = train_issues[:2]
test_issues = test_issues[:10]

examples = [
    (
        issue_data_to_text(issue),
        GithubIssue(github_issue_type=issue_data_to_github_issue_type(issue)),
    )
    for issue in train_issues
]


issues = test_issues
texts = [issue_data_to_text(issue) for issue in issues]
with open("output.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerow(["summary", "type", "url"])

    for issue, parsed in zip(
        issues, text_to_pydantic(llama, GithubIssue, tqdm(texts), examples=examples)
    ):
        print(issue["title"], parsed)
        writer.writerow(
            [
                issue["title"],
                parsed.github_issue_type.value,
                issue["html_url"],
            ]
        )
