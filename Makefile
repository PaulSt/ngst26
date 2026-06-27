PYTHON       ?= python
BOOK_TOOLS   = scripts/book_tools.py
VENV_DIR     ?= .venv
export VENV_DIR

JUPYTER_HOST ?= 127.0.0.1
JUPYTER_PORT ?= 8888
TOKEN        ?= myst-local
OPEN_BROWSER ?= 1
NOTEBOOKS_DIR ?= notebooks

STATIC_TIKZ_DIR ?= chapters/_static/tikz
LATEXMK         ?= latexmk
PDFTOCAIRO      ?= pdftocairo

.DEFAULT_GOAL := help

.PHONY: help install start notebooks live book jupyter figures tikz clean clean-notebooks clean-env

help:
	@$(PYTHON) $(BOOK_TOOLS) help

install:
	@$(PYTHON) $(BOOK_TOOLS) install

start:
	@$(PYTHON) $(BOOK_TOOLS) start --host "$(JUPYTER_HOST)" --port "$(JUPYTER_PORT)" --token "$(TOKEN)" --open-browser "$(OPEN_BROWSER)" --notebooks-dir "$(NOTEBOOKS_DIR)"

notebooks:
	@$(PYTHON) $(BOOK_TOOLS) notebooks --notebooks-dir "$(NOTEBOOKS_DIR)"

live:
	@$(PYTHON) $(BOOK_TOOLS) live --host "$(JUPYTER_HOST)" --port "$(JUPYTER_PORT)" --token "$(TOKEN)"

jupyter:
	@$(PYTHON) $(BOOK_TOOLS) jupyter --host "$(JUPYTER_HOST)" --port "$(JUPYTER_PORT)" --token "$(TOKEN)"

book:
	@$(PYTHON) $(BOOK_TOOLS) book

figures: tikz

tikz:
	@$(PYTHON) $(BOOK_TOOLS) tikz --static-tikz-dir "$(STATIC_TIKZ_DIR)" --latexmk "$(LATEXMK)" --pdftocairo "$(PDFTOCAIRO)"

clean:
	@$(PYTHON) $(BOOK_TOOLS) clean

clean-notebooks:
	@$(PYTHON) $(BOOK_TOOLS) clean-notebooks --notebooks-dir "$(NOTEBOOKS_DIR)"

clean-env:
	@$(PYTHON) $(BOOK_TOOLS) clean-env
