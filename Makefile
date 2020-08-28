help:
	@echo "Type 'make repair' to get a help text explaining how to run the tokenization repair."
	@echo "Type 'make evaluation' to get a help text explaining how to run the evaluation."
	@echo "Type 'make spelling' to start the interactive spell checking beam search."
	@echo "Type 'make start-server' to start the web demo."

repair:
	python3 demos/tokenization_repair.py

evaluation:
	python3 demos/evaluation.py

spelling:
	python3 scripts/spelling_beam_search.py -benchmark 0 -set 0 -n -1 -b 100 -sp 0 -cp 8 -f 0 -seg 0 -c 0

start-server:
	python3 scripts/server2.py
