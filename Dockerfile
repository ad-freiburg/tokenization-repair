FROM ubuntu:18.04
MAINTAINER Matthias Hertel hertelm@informatik.uni-freiburg.de
RUN apt-get update
RUN apt-get install -y --no-install-recommends apt-utils
RUN apt-get install -y make vim
RUN apt-get install -y python3 python3-pip
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
COPY Makefile Makefile
COPY bashrc bashrc
COPY test test
COPY src src
COPY scripts scripts
CMD ["/bin/bash", "--rcfile", "bashrc"]

### Build the container using docker or wharfer.
# docker build -t tokenization-repair .
# wharfer build -t tokenization-repair .

### Run the container somewhere in the ad-network.
# docker run -it -v /nfs/students/matthias-hertel/tokenization-repair-paper:/external tokenization-repair
# wharfer run -it -v /nfs/students/matthias-hertel/tokenization-repair-paper:/external tokenization-repair

