from create_wikipedia_benchmark import TypoNoiseInducer


if __name__ == "__main__":
    noiser = TypoNoiseInducer(0, 42)
    print(len(noiser.typos), "words")
    pairs = set()
    for w in noiser.typos:
        for misspelling in noiser.typos[w]:
            pairs.add((w, misspelling))
    print(len(pairs), "unique pairs")
    count = 0
    for w in noiser.typos:
        count += len(noiser.typos[w])
    print(count, "total misspellings")