PYTHON_INCLUDE=/usr/local/include/python3.8
THIRD_PARTY_PATH=./third_party
UDPIPE_LIB_PATH=$(THIRD_PARTY_PATH)/udpipe/bindings/python

install:
	pip install -r requirements.txt
	pip install -e .

download-bosque:
	git clone https://github.com/UniversalDependencies/UD_Portuguese-Bosque dante_parser/datasets/bosque
	cd dante_parser/datasets/bosque

download-udpipe: $(THIRD_PARTY_PATH)
	git clone https://github.com/huberemanuel/udpipe $(THIRD_PARTY_PATH)/udpipe
	cd $(THIRD_PARTY_PATH)/udpipe && git switch tagger_only

udpipe-bind: $(PYTHON_INCLUDE) $(UDPIPE_LIB_PATH)
	cd $(UDPIPE_LIB_PATH) && PYTHON_INCLUDE=$(PYTHON_INCLUDE) make

udpipe-train: $(UDPIPE_LIB_PATH)
	PYTHONPATH=$PYTHONPATH:$(UDPIPE_LIB_PATH)  python -m dante_parser.apps.train_udpipe --all_data

udpipe-evaluate: $(UDPIPE_LIB_PATH) $(model_path) $(input_conllu)
	PYTHONPATH=$PYTHONPATH:$(UDPIPE_LIB_PATH) python -m dante_parser.apps.evaluate_udpipe $(model_path) $(input_conllu)

udify-evaluate: $(UDPIPE_LIB_PATH) $(pred_file) $(true_file)
	PYTHONPATH=$PYTHONPATH:$(UDPIPE_LIB_PATH) python -m dante_parser.apps.evaluate_udify $(pred_file) $(true_file)

udpipe2-create-files:
	python -m dante_parser.apps.create_files_udpipe2

test:
	pytest -s

install-spacy-sm:
	python -m spacy download pt_core_news_sm

install-spacy-lg:
	python -m spacy download pt_core_news_lg

first-parser-trainer: $(train_conllu) $(val_conllu) 
	python -m dante_parser.parser.first_parser.trainer\
		--train_conllu $(train_conllu)\
		--val_conllu $(train_conllu)\
		--log_level debug