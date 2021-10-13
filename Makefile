help:
	@echo "-- Installation --"
	@echo "Step 1: type 'make download-data' to get the data."
	@echo "Step 2: type 'make build' to build the Docker container."
	@echo "Step 3: type 'make start' to start the Docker container."
	@echo
	@echo "-- Usage --"
	@echo "Type 'make repair' to get a help text explaining how to run our tokenization repair methods."
	@echo "Type 'make evaluation' to get a help text explaining how to run the evaluation."
	@echo "Type 'make web-demo' to start the web demo."

download-data:
	wget https://tokenization.cs.uni-freiburg.de/data.zip
	unzip data.zip
	chmod -R 777 data

build:
	docker build -t tokenization-repair .

start:
	docker run -it -p 1234:1234 -v ${PWD}/data:/external tokenization-repair

download-data:
	wget https://tokenization.cs.uni-freiburg.de/data.zip
	unzip data.zip

repair:
	python3 demos/tokenization_repair.py -h

evaluation:
	python3 demos/evaluation.py

web-demo:
	python3 scripts/server2.py
