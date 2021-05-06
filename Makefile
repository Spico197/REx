.PHONY: build test_upload upload docs test test_report clean

build: clean
	python3 setup.py sdist bdist_wheel

test_upload:
	python3 -m twine upload --repository testpypi dist/*

upload:
	python3 -m twine upload --repository pypi dist/*

docs:
	cd docs && sphinx-apidoc -o . ../rex &&	make html

test:
	python -m unittest -v
	flake8

test_report:
	coverage run -m unittest -v && coverage report

clean:
	cd docs && make clean
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -f .coverage
	rm -f coverage.xml
	find . | grep -E '(__pycache__|\.pyc|\.pyo$$)' | xargs rm -rf
