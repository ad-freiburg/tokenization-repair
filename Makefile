help:
	@echo "-- Installation --"
	@echo "Step 1: type 'make download-data' to get the data."
	@echo "Step 2: type 'make build' to build the Docker container."
	@echo "Step 3: type 'make start' to start the Docker container."
	@echo
	@echo "-- Usage --"
	@echo "Type 'make repair' to get a help text explaining how to run our tokenization repair methods."
	@echo "Type 'make repair-corpus' to get a help text explaining how to repair all files in a directory."
	@echo "Type 'make evaluation' to get a help text explaining how to run the evaluation."
	@echo "Type 'make spelling-evaluation' to get a help text explaining how to run the spelling evaluation."
	@echo "Type 'make web-demo' to start the web demo."

download-data:
	wget https://ad-publications.informatik.uni-freiburg.de/CoNLL_tokenization_repair_BHM_2021.materials/data.zip
	unzip data.zip
	chmod -R 777 data

build:
	docker build -t tokenization-repair .

start:
	docker run -it -p 1234:1234 -v ${PWD}/data:/external tokenization-repair

web-demo:
	python3 scripts/web_demo.py

repair:
	python3 scripts/tokenization_repair.py -h

repair-corpus:
	python3 scripts/repair_corpus.py -h

evaluation:
	python3 scripts/evaluation.py

spelling-evaluation:
	python3 scripts/spelling_evaluation.py -h
