FROM ubuntu:18.04
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
COPY demos demos
COPY html html
ENV PYTHONIOENCODING=utf-8
CMD ["/bin/bash", "--rcfile", "bashrc"]

### Build the container:
# docker build -t tokenization-repair .

### Run the container:
# docker run -it -v <DATA-DIR>:/external -p 1234:1234 tokenization-repair
