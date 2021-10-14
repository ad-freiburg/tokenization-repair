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

### Web demo

To start the web demo, run `make web-demo`.
The default port is 1234.
Access the web demo by typing localhost:1234 in the address bar of a web browser.
Type a text into the text field, and select one of six methods to correct it.
These include our methods from the paper (UNI, UNI+, BID, BID+, The One),
and the spell checking beam search.
Insertion and deletion penalties are set to 0 for UNI(+) and BID(+),
because they differ for every benchmark - this means that for these approaches
all spaces are ignored in the web demo.

### Run our method

Type `make repair` to get a help text explaining how to run our method interactively, on a specified file, or on a benchmark.

Benchmarks must be located at `<DATA-DIRECTORY>/benchmarks` and contain folders *development* and *test*.
Results will be written to `<DATA-DIRECTORY>/results`.

### Penalties

The following table shows the penalties for the different approaches, optimized for the five benchmarks with spaces.
The first number in a cell is the insertion penalty P_ins,
and the second number is the deletion penalty P_del.

| Approach | ACL | arXiv OCR | arXiv PDF | Wiki | Wiki+ |
| --- | --- | --- | --- | --- | --- |
| UNI | 7.2, 4,2 | 6.2, 4.7 | 8.2, 2.7 | 3.1, 3.9 | 10.2, 9.5 |
| UNI+ | 4.5, 5.0 | 3.8, 4.4 | 11.4, 3.3 | 2.2, 4.2 | 4.8, 6.3 |
| BID | 8.1, 7.1 | 10.2, 7.2 | 9.9, 4.9 | 5.7, 8.3 | 12.3, 16.3 |
| BID+ | 7.6, 9.2 | 6.8, 6.2 | 5.9, 3.7 | 3.7, 6.9 | 7.4, 10.4 |

The penalties for *BID+ The One* are P_ins = 6.32 and P_del = 6.9 for the five benchmarks.
This differs slightly from the averaged penalties for BID+,
because some penalties for BID+ were updated after the experiments with *BID+ The One* were run.

The penalties for the *Wiki+ no spaces* benchmark are 0 for all approaches.

### Repair all files in a directory (e.g. a corpus)

Type `make repair-corpus` to get a help text explaining how to run our method on all files in a folder.
We used this script to repair all files in the ACL corpus.
The files are processed line by line.
Per default, lines ending with a dash are concatenated with the next line before repairing the tokenization
(and split again after the tokenization was repaired).

