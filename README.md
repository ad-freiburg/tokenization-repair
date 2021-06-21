This is version 1.2.1 of the Tokenization Repair software.

## Installation guide ##

1. Get the code (code.zip) of the latest stable version from our website.

1. Download the data (data.zip) from the same link and extract it to a directory DATA-DIR. It contains benchmarks, result files and trained models.
   To be able to run the Docker container, the data directory must be writable by any user.
   
       chmod -R 777 <DATA_DIR>
   
   Additionally, you can get the training data from the same link.

1. In case you have a GPU supporting tensorflow 1.12, change the base image in the Dockerfile to:
	tensorflow/tensorflow:1.12.0-gpu-py3

1. Build the Docker container. For help with Docker visit: https://docs.docker.com/get-docker/

       docker build -t tokenization-repair .

1. Start the Docker container and mount the data directory.

       docker run -it -p <PORT>:1234 -v <DATA-DIR>:/external tokenization-repair

1. Inside the container, repair some tokens!
   Type `make help` to get a specification of all the make targets.
