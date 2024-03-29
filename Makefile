all: format clean pre test build docs
	echo 'finished'

.PHONY: build
build: clean
	python3 setup.py sdist bdist_wheel

.PHONY: test_upload
test_upload:
	python3 -m twine upload --repository testpypi dist/*

.PHONY: upload
upload:
	python3 -m twine upload --repository pypi dist/*

.PHONY: docs
docs:
	cd docs && make clean
	cd docs && sphinx-apidoc -o . ../rex &&	make html

.PHONY: format
format:
	isort --profile black --filter-files .
	black .

.PHONY: test
test:
	coverage run --source rex -m pytest -vv .
	coverage report -m
	flake8

.PHONY: pre
pre:
	pre-commit run --all-files

.PHONY: debug
debug:
	pytest -vv tests/tasks/test_re.py

.PHONY: clean
clean:
	cd docs && make clean
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -f .coverage
	rm -f coverage.xml
	find . | grep -E '(__pycache__|\.pyc|\.pyo$$)' | xargs rm -rf

.PHONY: serve_docs
serve_docs:
	python -m http.server --directory docs/_build/html
