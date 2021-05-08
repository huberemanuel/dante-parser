install:
	pip install -r requirements.txt
	pip install -e .

download-bosque:
	git clone https://github.com/UniversalDependencies/UD_Portuguese-Bosque data/bosque
	cd data/bosque && git switch workbench

test:
	pytest -s --pyargs dante_parser
