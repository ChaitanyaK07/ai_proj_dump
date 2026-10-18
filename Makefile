# Convenience targets (Unix shells). On Windows run the same `python -m wayfinder ...` commands directly.
PY ?= python

.PHONY: install test test-fast pipeline preprocess labels node2vec train evaluate study serve

install:
	$(PY) -m pip install -r requirements-dev.txt
	$(PY) -m pip install -e . --no-deps

test:
	$(PY) -m pytest

test-fast:
	$(PY) -m pytest -m "not slow"

pipeline:
	$(PY) -m wayfinder all

preprocess:
	$(PY) -m wayfinder preprocess

labels:
	$(PY) -m wayfinder labels

node2vec:
	$(PY) -m wayfinder node2vec

train:
	$(PY) -m wayfinder train --hparam

evaluate:
	$(PY) -m wayfinder evaluate

study:
	$(PY) -m wayfinder study-prepare

serve:
	$(PY) -m wayfinder serve
