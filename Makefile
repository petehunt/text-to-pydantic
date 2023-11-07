all: venv issues_train issues_test mistral-7b-instruct-v0.1.Q5_K_M.gguf

venv:
	python3 -m venv venv && venv/bin/pip install --upgrade pip && venv/bin/pip install -r requirements.txt

issues_train:
	mkdir -p issues_train && cd issues_train && bash -c 'wget https://api.github.com/repos/dagster-io/dagster/issues?state=all\&per_page=100\&page={20..30}'

issues_test:
	mkdir -p issues_test && cd issues_test && bash -c 'wget https://api.github.com/repos/dagster-io/dagster/issues?state=all\&per_page=100\&page={10..20}'

mistral-7b-instruct-v0.1.Q5_K_M.gguf:
	wget https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q5_K_M.gguf