help:
	@echo "Type one of the following commands to start correcting tokens in an interactive shell:"
	@echo "  'make beam-search'"
	@echo "  'make beam-search-bwd'"
	@echo "  'make beam-search-robust'"
	@echo "  'make beam-search-bwd-robust'"
	@echo "  'make beam-search-bidir'"
	@echo "  'make beam-search-bidir-robust'"
	@echo "Type 'make spelling' to start interactive spell checking beam search."
	@echo "Type 'make start-server' to start the web demo."

beam-search:
	python3 scripts/batched_beam_search.py -m fwd1024 -b 0 -set 0 -seq 0 -n -1 -c 0 -w 5 -p 0 -pm 0 -f 0 -labeling 0 -l 0

beam-search-bwd:
	python3 scripts/batched_beam_search.py -m bwd1024 -b 0 -set 0 -seq 0 -n -1 -c 0 -w 5 -p 0 -pm 0 -f 0 -labeling 0 -l 0

beam-search-robust:
	python3 scripts/batched_beam_search.py -m fwd1024_noise0.2 -b 0 -set 0 -seq 0 -n -1 -c 0 -w 5 -p 0 -pm 0 -f 0 -labeling 0 -l 0

beam-search-bwd-robust:
	python3 scripts/batched_beam_search.py -m bwd1024_noise0.2 -b 0 -set 0 -seq 0 -n -1 -c 0 -w 5 -p 0 -pm 0 -f 0 -labeling 0 -l 0

beam-search-bidir:
	python3 scripts/batched_beam_search.py -m fwd1024 -b 0 -set 0 -seq 0 -n -1 -c 0 -w 5 -p 0 -pm 0 -f 0 -labeling labeling_ce -l 0

beam-search-bidir-robust:
	python3 scripts/batched_beam_search.py -m fwd1024_noise0.2 -b 0 -set 0 -seq 0 -n -1 -c 0 -w 5 -p 0 -pm 0 -f 0 -labeling labeling_noisy_ce -l 0

spelling:
	python3 scripts/spelling_beam_search.py -benchmark 0 -set 0 -n -1 -b 100 -sp 0 -cp 8 -f 0 -seg 0 -c 0

start-server:
	python3 scripts/start_server.py
