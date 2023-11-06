all: venv issues.json mistral-7b-instruct-v0.1.Q5_K_M.gguf

venv:
	python3 -m venv venv && venv/bin/pip install --upgrade pip && venv/bin/pip install -r requirements.txt

issues.json:
	curl 'https://api.github.com/repos/dagster-io/dagster/issues?state=all&per_page=100' > issues.json

mistral-7b-instruct-v0.1.Q5_K_M.gguf:
	wget https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q5_K_M.gguf