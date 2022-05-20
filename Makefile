.DEFAULT_GOAL := help # when you run make, it defaults to printing available commands

# Optimus Repo Path
# discover the absolute path to the project repo on the host machine
ifeq ($(OS),Windows_NT)
	# * docker driver does not support path-based volume mounts with special characters in the source or target mount path
	# * 'space' character is considered a special character and is quite common as a user name on firm laptops
	#   for example: C:\Users\Tom Smith\Documents\optimus
	# * here we are using powershell to discover the MS DOS 'short-path'
	#   which is an equivalant path expression but without any spaces.
	#   for example, the user path above would look something like this: C:\Users\TOM~1\DOCUME~1\optimus
	OPTIMUS_DIR := $(shell powershell "(New-Object -ComObject Scripting.FileSystemObject).GetFolder('.').ShortPath")
else
	OPTIMUS_DIR := "$$(pwd)"
endif

.PHONY: base-build
base-build: ## build the optimus base docker image
	docker build -t optimus-base-clisham .

.PHONY: base-specs
base-specs: ## run the base image specs
	docker run -t --workdir="/optimus" -v $(OPTIMUS_DIR):/optimus optimus-base-clisham /bin/bash -c "bats ./support/base_image_specs.bats"

.PHONY: base-interactive
base-interactive: ## get a bash shell in the base image
	docker run -it --workdir="/optimus" -v miniconda:/miniconda -v $(OPTIMUS_DIR):/optimus optimus-base-clisham /bin/bash

.PHONY: dependencies-install
dependencies-install: ## install packages from poetry.lock
	docker run -it --workdir="/optimus" -v miniconda:/miniconda -v $(OPTIMUS_DIR):/optimus optimus-base-clisham /bin/bash -ci "poetry install"

.PHONY: dependencies-update
dependencies-update: ## update poetry.lock
	docker run -it --workdir="/optimus" -v miniconda:/miniconda -v $(OPTIMUS_DIR):/optimus optimus-base-clisham /bin/bash -ci "poetry update --lock"

.PHONY: docker-clean
docker-clean: ## stop+kill all running containers. prune stopped containers. remove all untagged images
ifeq ($(OS),Windows_NT)
	powershell "docker ps -qa | %{docker kill $$_}; docker container prune --force; docker system prune --force;"
else
	docker ps -qa | xargs docker kill; docker container prune --force; docker system prune --force;
endif

.PHONY: docs-build
docs-build: ## build the top level docs
	docker run -t --workdir="/optimus" -v miniconda:/miniconda -v $(OPTIMUS_DIR):/optimus optimus-base-clisham /bin/bash -ci "kedro build-docs"

.PHONY: pipeline-run
pipeline-run: ## run the full pipeline
	docker run -t --workdir="/optimus" -v miniconda:/miniconda -v $(OPTIMUS_DIR):/optimus optimus-base-clisham /bin/bash -ci "kedro run"

.PHONY: jupyter-lab
# you can overide the default port by setting the shell variable JUPYTER_PORT (Ex: make jupyter-lab JUPYTER_PORT=4500)
JUPYTER_PORT ?= 8888
jupyter-lab: ## start jupyter lab within the docker container. (http://localhost:<port> | <port> defaults to 8888)
	docker run -t --rm --workdir="/optimus" -p $(JUPYTER_PORT):$(JUPYTER_PORT) -v miniconda:/miniconda -v $(OPTIMUS_DIR):/optimus optimus-base-clisham /bin/bash -ci "kedro jupyter lab --allow-root --port $(JUPYTER_PORT) --ip 0.0.0.0 --no-browser --NotebookApp.token=''"

.PHONY: jupyter-notebook
# you can overide the default port by setting the shell variable JUPYTER_PORT (Ex: make jupyter-notebook JUPYTER_PORT=4500)
JUPYTER_PORT ?= 8888
jupyter-notebook: ## start jupyter notebook within the docker container. (http://localhost:<port> | <port> defaults to 8888)
	docker run -t --rm --workdir="/optimus" -p $(JUPYTER_PORT):$(JUPYTER_PORT) -v miniconda:/miniconda -v $(OPTIMUS_DIR):/optimus optimus-base-clisham /bin/bash -ci "kedro jupyter notebook --allow-root --port $(JUPYTER_PORT) --ip 0.0.0.0 --no-browser --NotebookApp.token=''"

.PHONY: kedro-viz
# you can overide the default port by setting the shell variable VIZ_PORT (Ex: make kedro-viz VIZ_PORT=4500)
VIZ_PORT ?= 4141
kedro-viz: ## start kedro viz within the docker container. (http://localhost:<port> | <port> defaults to 4141)
	docker run -t --rm --workdir="/optimus" -p $(VIZ_PORT):$(VIZ_PORT) -v miniconda:/miniconda -v $(OPTIMUS_DIR):/optimus optimus-base-clisham /bin/bash -ci "kedro viz --no-browser --host=0.0.0.0 --port=$(VIZ_PORT)"

.PHONY: optimizer-tests
optimizer-tests: ## run unit tests for the optimizer
	docker run -t --workdir="/optimus/optimizer" -v miniconda:/miniconda -v $(OPTIMUS_DIR):/optimus optimus-base-clisham /bin/bash -ci "pytest tests"

.PHONY: pipeline-tests
pipeline-tests: ## run unit tests for the pipeline
	docker run -t --workdir="/optimus/src" -v miniconda:/miniconda -v $(OPTIMUS_DIR):/optimus optimus-base-clisham /bin/bash -ci "pytest tests"

.PHONY: pipeline-kedro-test
pipeline-kedro-test: ## run unit tests for the pipeline
	docker run -t --workdir="/optimus" -v miniconda:/miniconda -v $(OPTIMUS_DIR):/optimus optimus-base-clisham /bin/bash -ci "kedro test"

.PHONY: help
help:  ## show all make commands
ifeq ($(OS),Windows_NT)
	powershell "((type Makefile) -match '##') -notmatch 'grep'"
else
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' Makefile | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'
	@test -f ./utilities/Makefile && grep -E '^[a-zA-Z_-]+:.*?## .*$$' ./utilities/Makefile | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}' || true
endif

.PHONY: local-test
local-test: ## local version of integration tests
	docker run -t --workdir="/optimus" -v miniconda:/miniconda -v $(OPTIMUS_DIR):/optimus optimus-base-clisham /bin/bash -ci "kedro run --env integration --pipeline full"

.PHONY: pipeline-list
pipeline-list: ## list kedro pipelines
	docker run -t --workdir="/optimus" -v miniconda:/miniconda -v $(OPTIMUS_DIR):/optimus optimus-base-clisham /bin/bash -ci "kedro pipeline list"

.PHONY: pipeline-run-name
PIPE_NAME ?= __default__
pipeline-run-name: ## Run a specific pipeline via the PIPE_NAME env
		docker run -t --workdir="/optimus" -v miniconda:/miniconda -v $(OPTIMUS_DIR):/optimus optimus-base-clisham /bin/bash -ci "kedro run --pipeline $(PIPE_NAME)"

.PHONY: pipeline-run-namedev
PIPE_NAME ?= __default__
pipeline-run-namedev: ## Run a specific pipeline via the PIPE_NAME env
		docker run -t --workdir="/optimus" -v miniconda:/miniconda -v $(OPTIMUS_DIR):/optimus optimus-base-clisham /bin/bash -ci "kedro run --pipeline $(PIPE_NAME) --env=cloud_recommend_dev"

.PHONY: pipeline-run-name-env
PIPE_NAME ?= __default__
ENV_NAME ?= __default__
pipeline-run-name-env: ## Run a specific pipeline via the PIPE_NAME env
		docker run -t --workdir="/optimus" -v miniconda:/miniconda -v $(OPTIMUS_DIR):/optimus optimus-base-clisham /bin/bash -ci "kedro run --pipeline $(PIPE_NAME) --env $(ENV_NAME)"

.PHONY: pipeline-run-tag
TAG_NAME ?= __default__
pipeline-run-tag: ## Run a specific tag via the TAG_NAME env
		docker run -t --workdir="/optimus" -v miniconda:/miniconda -v $(OPTIMUS_DIR):/optimus optimus-base-clisham /bin/bash -ci "kedro run --tag $(TAG_NAME)"

.PHONY: pipeline-run-pt 
TAG_NAME ?= __default__
PIPE_NAME ?= __default__
pipeline-run-pt: ## Run a specific pipeline and tag via PIPE_NAME & TAG_NAME env
		docker run -t --workdir="/optimus" -v miniconda:/miniconda -v $(OPTIMUS_DIR):/optimus optimus-base-clisham /bin/bash -ci "kedro run --pipeline $(PIPE_NAME) --tag $(TAG_NAME)"

.PHONY: streamlit-show
STREAMLIT_PORT ?= 4142
MODEL_NAME ?= ma2
streamlit-show: ## start streamlit sensitivity app
	docker run -t --rm --workdir="/optimus" -p $(STREAMLIT_PORT):$(STREAMLIT_PORT) -v miniconda:/miniconda -v $(OPTIMUS_DIR):/optimus optimus-base-clisham /bin/bash -ci "streamlit run src/project_clisham/pipelines/optimization/sensitivity/streamlit_app.py $(MODEL_NAME) --server.port $(STREAMLIT_PORT)"

.PHONY: pipeline-kedro-install
pipeline-kedro-install: ## install pipeline related dependencies with kedro + install pipeline project in edit mode for development
	docker run -t --workdir="/optimus" -v miniconda:/miniconda -v $(OPTIMUS_DIR):/optimus optimus-base-clisham /bin/bash -ci "pip install --no-deps -e ./src"

.PHONY: mlflow-ui
# you can overide the default port by setting the shell variable MLFLOW_PORT (Ex: make jupyter-lab MLFLOW_PORT=4500)
MLFLOW_PORT ?= 5000
mlflow-ui: ## start jupyter lab within the docker container. (http://localhost:<port> | <port> defaults to 8888)
	docker run -t --rm --workdir="/optimus" -p $(MLFLOW_PORT):$(MLFLOW_PORT) -v miniconda:/miniconda -v $(OPTIMUS_DIR):/optimus optimus-base-clisham /bin/bash -ci "mlflow ui --host 0.0.0.0"
