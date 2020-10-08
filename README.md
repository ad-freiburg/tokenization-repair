This is version 1.1.1 (June 17, 2020) of the Tokenization Repair software.

For help with Docker visit: https://docs.docker.com/get-docker/

## Installation guide ##

1. Download the data from http://emnlp2020-367.hopto.org/data.zip and extract it to a directory DATA-DIR.

2. In case you have a GPU supporting tensorflow 1.12, change line 8 in the requirements.txt to:
	tensorflow-gpu==1.12.0

3. Build the docker container.

       docker build -t tokenization-repair .

4. Start the Docker container and link it to the data directory.

       docker run -it -v <DATA-DIR>:/external tokenization-repair

5. Inside the container, repair some tokens!
   Type `make help` to get a specification of all the make targets.

## EMNLP 2020 Reproducibility Webpage ##

For instructions on how to start the web server and demo, see 
https://github.com/ad-freiburg/tokenization-repair/tree/master/repro-emnlp2020
