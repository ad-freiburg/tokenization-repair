help:
	@echo "Type 'make iterative' to start the iterative approach."
	@echo "Type 'make beam-search' to start the beam search approach."

test:
	nosetests tests

iterative:
	python3 scripts/iterative_window_correction.py -b 0 -t 0 -c 0

beam-search:
	python3 scripts/batched_beam_search.py -m fwd1024 -b 0 -t 0

