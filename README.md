# Tokenization Repair in the Presence of Spelling Errors

This software attempts to solve the following *Tokenization Repair* problem:
Given a text with missing and spurious spaces, correct those.
Spelling errors and OCR errors can be present, but it's not part of the problem to correct them.

Visit [tokenization.cs.uni-freiburg.de](https://tokenization.cs.uni-freiburg.de) for a web demo and interactive evaluation results.

If you use the software in your research, please cite our [CoNLL 2021 paper](https://aclanthology.org/2021.conll-1.22/) as below.

## Quickstart with Docker

Install and run the software on a text file in four easy steps.
For GPU support see the step-by-step guide below.

1. Clone the repository.

       git clone https://github.com/ad-freiburg/tokenization-repair.git
       cd tokenization-repair

2. Download the data, including trained models.

       make download-data

3. Build the docker container.

       make build

4. Run our method on a file *input_file.txt* in the current directory. 
   The results will be written to a file *output_file.txt*.

       docker run -v $(pwd)/data:/external -v $(pwd):/pwd tokenization-repair \
         python3 scripts/tokenization_repair.py -f /pwd/input_file.txt -o /pwd/output_file.txt

## Step-by-step installation guide ##

1. Clone the repository.
   
       git clone https://github.com/ad-freiburg/tokenization-repair.git
       cd tokenization-repair

3. Download and extract the data with the command `make download-data`.
   It contains benchmarks, result files and trained models.
   To be able to run the Docker container, the data directory must be writable by any user.
   This can be ensured with `chmod -R 777 data`.

4. If you have a GPU that supports tensorflow 1.12, change the base image in the first line of the Dockerfile to:
    tensorflow/tensorflow:1.12.0-gpu-py3

5. Build the Docker container. For help with Docker visit: https://docs.docker.com/get-docker/

       docker build -t tokenization-repair .
    
    The build command can also be called with `make build`.

6. Start the Docker container and mount the data directory.

       docker run -it -p <PORT>:1234 -v <DATA-DIRECTORY>:/external tokenization-repair
    
    The start command can also be called with `make start`.
    The default port for the web demo is 1234,
    and the default data directory is "data".

7. Inside the container, repair some tokens!
   Type `make help` to get a specification of all the make targets.

## User guide

Inside the Docker container, you can start the interactive web demo,
run our method on a file (or all files from a directory), 
and run the evaluations from our paper.

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

### Run evaluations

Type `make evaluation` to get a help text explaining how to run the tokenization repair evaluation,
and `make spelling-evaluation` for the spelling evaluation.

## Training models on custom data

### Training data

Prepare a text file with one example sequence (e.g., a paragraph of text) per line.
The spaces in the text must be correct, but the text may contain other errors
like spelling mistakes or wrongly OCR'd characters.

### Unidirectional model

Run `python3 scripts/train_estimator.py -name <UNI_MODEL_NAME> -data <TEXT_FILE>`
to train the unidirectional model.
List additional arguments with `python3 scripts/train_estimator.py -h`.

### Bidirectional model

Run `python3 scripts/train_labeling_estimator.py -name <BID_MODEL_NAME> -data <TEXT_FILE>`
to train the bidirectional model.
List additional arguments with `python3 scripts/train_labeling_estimator.py -h`.

### Use custom models

Run `python3 scripts/tokenization_repair.py -a CUSTOM -fwd <UNI_MODEL_NAME> -bid <BID_MODEL_NAME>`
to use the models with the specified names interactively, and specify`-f <INPUT_TEXT_FILE>`
to run them on a text file with tokenization errors.
For the best performance, the penalties P_ins and P_del must be set.
You can try the penalties `-p_ins 6.9 -p_del 6.32`, which gave good results on all our benchmarks,
or optimize them on a held-out dataset with ground truth.

## Version

This is version 1.2.2 of the Tokenization Repair software.

## Paper

```bibtex
@inproceedings{bast-etal-2021-tokenization,
    title = "Tokenization Repair in the Presence of Spelling Errors",
    author = "Bast, Hannah  and
      Hertel, Matthias  and
      Mohamed, Mostafa M.",
    booktitle = "Proceedings of the 25th Conference on Computational Natural Language Learning",
    month = nov,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.conll-1.22",
    pages = "279--289",
    abstract = "We consider the following tokenization repair problem: Given a natural language text with any combination of missing or spurious spaces, correct these. Spelling errors can be present, but it{'}s not part of the problem to correct them. For example, given: {``}Tispa per isabout token izaionrep air{''}, compute {``}Tis paper is about tokenizaion repair{''}. We identify three key ingredients of high-quality tokenization repair, all missing from previous work: deep language models with a bidirectional component, training the models on text with spelling errors, and making use of the space information already present. Our methods also improve existing spell checkers by fixing not only more tokenization errors but also more spelling errors: once it is clear which characters form a word, it is much easier for them to figure out the correct word. We provide six benchmarks that cover three use cases (OCR errors, text extraction from PDF, human errors) and the cases of partially correct space information and all spaces missing. We evaluate our methods against the best existing methods and a non-trivial baseline. We provide full reproducibility under https://ad.informatik.uni-freiburg.de/publications.",
}
```
