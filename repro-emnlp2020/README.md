# EMNLP 2020 Repro instructions

The URL http://emnlp2020-367.hopto.org currently routes to filicudi.informatik.privat 
From there it is currently proxied to http://tarka.informatik.privat:20367 
The configuration for this is in /etc/apache2/sites-available/repro-emnlp2020 
If you change something in this configuration and want that change to become active: `sudo /etc/init.d/apache2 restart`
If you don't have sudo rights, pray or open the window and cry for help.

To simply serve all files in a certain directory on this port: `python3 -m http.server 20367`
If you see a web page under http://emnlp2020-367.hopto.org the server is already running.

This file is currently located under tarka:/local/data/tokenization-repair/repro-emnlp2020
The working copy there has read and write permissions for group student, so we should all be able to make changes there (and commit them to GitHub via our own respective accounts).

## How to start the file server

On tarka:

    cd /local/data/tokenization-repair/repro-emnlp2020
    python3 -m http.server 20367

This does not include the web demo (see next section).

## How to start the web demo

On tarka:

    cd /local/data/tokenization-repair
    wharfer run -it -v /local/data/tokenization-repair/data:/external -p 1234:1234 tokenization-repair
    make start-server

To change the port where the web demo is served, replace the first number of the `-p` argument: `-p XXXX:1234`

The directory /local/data/tokenization-repair/data is necessary because it contains the trained models.

### In case you need to (re-)build the container

Rebuild the container on tarka:

    cd /local/data/tokenization-repair
    wharfer build -t tokenization-repair .

Or get the code from Github and build the container:

    git clone https://github.com/ad-freiburg/tokenization-repair.git
    cd tokenization-repair
    wharfer build -t tokenization-repair .

To be able to run the container, you need access to a directory with the trained models,
and mount it as explained in the section above.
You can get the data from the following locations:
- tarka:/local/data/tokenization-repair/data
- tarka:/local/data/tokenization-repair/repro-emnlp2020/data.zip (compressed)
- sirba:/local/data/hertelm/tokenization-repair-dumps/data.zip (compressed)
- vulcano:/local/data/hertelm/tokenization-repair-backup/data.zip (compressed)