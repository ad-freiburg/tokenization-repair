.PHONY: clean-pyc clean-build docs help
.DEFAULT_GOAL := help
define BROWSER_PYSCRIPT
import os, webbrowser, sys
try:
	from urllib import pathname2url
except:
	from urllib.request import pathname2url

webbrowser.open("file://" + pathname2url(os.path.abspath(sys.argv[1])))
endef
export BROWSER_PYSCRIPT
BROWSER := python -c "$$BROWSER_PYSCRIPT"

help:
	@perl -nle'print $& if m{^[a-zA-Z_-]+:.*?## .*$$}' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-25s\033[0m %s\n", $$1, $$2}'

all: test clean lint-flake

clean: ## Remove Python file artifacts
	find . -name '__pycache__' -exec rm -rf {} +
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*.sw*' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +

lint-flake: ## Check style with flake8, details in the 'Linting' section.
	flake8 data models utils handlers configs.py constants.py run_train.py run_benchmarks.py extract_editops.py

run-polyaxon:  ## Run polyaxon
	polyaxon run -u -f polyaxon.yml

init-polyaxon:  ## Initialize polyaxon project
	polyaxon project create --name="token-repair" --description="Tokenization repair paper"
	polyaxon init "token-repair"

logs-polyaxon:  ## Polyaxon logs
	polyaxon experiment -xp $(id) logs --past

monitor-polyaxon:  ## Polyaxon monitor resources
	polyaxon experiment -xp $(id) get

stop-polyaxon:  ## Polyaxon stop resources
	polyaxon experiment -xp $(id) stop

statuses-polyaxon:
	polyaxon experiment statuses

validations-polyaxon:
	polyaxon experiment -xp $(id) logs --past | grep val_ --binary-files=text
