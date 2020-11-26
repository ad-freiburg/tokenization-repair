This is version 1.1.2 (beta) of the Tokenization Repair software.
The latest stable version is 1.1.1 (June 17, 2020).

## Installation guide ##

1. Get the code (code.zip) of the latest stable version from here: https://ad-research.cs.uni-freiburg.de/data/tokenization-repair/

1. Download the data (data.zip) from the same link and extract it to a directory DATA-DIR. It contains benchmarks, result files and trained models. Additionally, you can get the training data (training_shuffled.zip) from the same link.

1. In case you have a GPU supporting tensorflow 1.12, change line 8 in the requirements.txt to:
	tensorflow-gpu==1.12.0

1. Build the docker container. For help with Docker visit: https://docs.docker.com/get-docker/

       docker build -t tokenization-repair .

1. Start the Docker container and mount the data directory.

       docker run -it -p <PORT>:1234 -v <DATA-DIR>:/external tokenization-repair

1. Inside the container, repair some tokens!
   Type `make help` to get a specification of all the make targets.

## EMNLP 2020 Reproducibility Webpage ##

For instructions on how to start the web server and demo, see 
https://github.com/ad-freiburg/tokenization-repair/tree/master/repro-emnlp2020
