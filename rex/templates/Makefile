all: format clean test
	echo 'finished'

.PHONY: format
format:
	isort --profile black --filter-files .
	black .

.PHONY: test
test:
	coverage run --source rex -m pytest -vv .
	coverage report -m
	flake8

.PHONY: clean
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -f .coverage
	rm -f coverage.xml
	find . | grep -E '(__pycache__|\.pyc|\.pyo$$)' | xargs rm -rf
