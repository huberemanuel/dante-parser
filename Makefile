install:
	pip install -r requirements.txt
	pip install -e .

test:
	pytest -s --pyargs dante_parser
