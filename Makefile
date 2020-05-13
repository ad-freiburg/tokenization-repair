help:
	@echo "Type 'make iterative' to start the iterative tokenization repair."
	@echo "Type 'make beam-search' to start the beam search tokenization repair."
	@echo "Type 'make spelling' to start the beam search spelling correction."

test:
	nosetests tests

iterative:
	python3 scripts/iterative_window_correction.py -b 0 -t 0 -c 0

beam-search:
	python3 scripts/batched_beam_search.py -m fwd1024 -b 0 -t 0

spelling:
	python3 scripts/spelling_beam_search.py -benchmark 0 -n -1 -b 20 -sp 0 -cp 8 -f 0
