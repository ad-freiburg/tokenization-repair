This is version 1.2.1 of the Tokenization Repair software.

## Installation guide ##

1. Clone our repository.
   
       git clone https://github.com/ad-freiburg/tokenization-repair.git

2. Download and extract the data with the command `make download-data`.
   It contains benchmarks, result files and trained models.
   To be able to run the Docker container, the data directory must be writable by any user.
   This can be ensured with `chmod -R 777 data`.

3. If you have a GPU that supports tensorflow 1.12, change the base image in the first line of the Dockerfile to:
    tensorflow/tensorflow:1.12.0-gpu-py3

4. Build the Docker container. For help with Docker visit: https://docs.docker.com/get-docker/

       docker build -t tokenization-repair .
    
    The build command can also be called with `make build`.

5. Start the Docker container and mount the data directory.

       docker run -it -p <PORT>:1234 -v <DATA-DIRECTORY>:/external tokenization-repair
    
    The start command can also be called with `make start`.
    The default port for the web demo is 1234,
    and the default data directory is "data".

7. Inside the container, repair some tokens!
   Type `make help` to get a specification of all the make targets.

## User guide

