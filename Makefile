install:
	pip install -r requirements.txt
	pip install -e .

download-bosque:
	git clone https://github.com/UniversalDependencies/UD_Portuguese-Bosque dante_parser/datasets/bosque
	cd dante_parser/datasets/bosque && git switch workbench

test:
	pytest -s --pyargs dante_parser
