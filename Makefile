all: venv issues.json labels.json mistral-7b-instruct-v0.1.Q5_K_M.gguf

venv:
	python3 -m venv venv && venv/bin/pip install --upgrade pip && venv/bin/pip install -r requirements.txt

issues.json:
	curl 'https://api.github.com/repos/dagster-io/dagster/issues?state=all&per_page=100' > issues.json

labels.json:
	curl 'https://api.github.com/repos/dagster-io/dagster/labels?per_page=100' > labels.json

mistral-7b-instruct-v0.1.Q5_K_M.gguf:
	echo "populate"