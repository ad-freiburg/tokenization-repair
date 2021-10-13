from typing import List

import sys
if "spacy" in sys.argv:
    import spacy

import project
from src.settings import paths
from src.helper.files import read_sequences
from src.helper.time import timestamp, time_diff
from src.interactive.sequence_generator import interactive_sequence_generator
from src.sequence.sentence_splitter import NLTKSentenceSplitter, WikiPunktTokenizer


class SpacySentenceSplitter:
    def __init__(self):
        self.model = spacy.load("en_core_web_sm")

    def split(self, text: str) -> List[str]:
        doc = self.model(text)
        sentences = [sent.text for sent in doc.sents]
        return sentences


test_sequences = [
    "The more melodic sound showcased on this album is increasingly cited as an influence on present-day alternative rock bands by the rock press, i.e. Thursday, Biffy Clyro, Jimmy Eat World.",
    """The printer Heinrich Ludwig (Henry Ludwig) (?-1872) was a Palatine German American from Columbia County, New York. He started a retail book business in New York city selling German schoolbooks and hymnals printed in Philadelphia; in 1834, he established his own printing house and published the newspaper ' ("General Newspaper") (1835-1840?); in the 1840s he printed and imported German books; from 1852 to 1872 he published and edited ' ("The Lutheran Herald") for the New York State Lutheran Ministerium. Ludwig and Radde, both German Lutheran printers of similar age, collaborated for many years. Radde was an energetic entrepreneur and philanthropist who did not set up his own printing press in America but rather used skilled local printers. Henry Ludwig printed the "The North American Journal of Homeopathy" from 1856 to 1870, initially as "Book & Job Printer, 45 Vesey-st.," later as "Book and Job Printer and Stereotyper, Nos. 39 and 41 Centre Street." A hymnal printed by Ludwig in 1834 was stereotyped by Henry W. Rees. By 1854, a Ludwig publication was "to be had of all the principal Booksellers throughout the United States\"""",
    """In 2000, Oregon Restaurant Association v. City of Corvallis affirmed that cities had the right to enact their own, more strict ordinances on smoking and led the way towards removing preemption with the Oregon Indoor Clean Air Act. However, until this law passed in 2009, preemption allowed all cities to enact their own regulations for tobacco smoking to include establishments not explicitly defined in ORS 433.835. In 2007, the Oregon Clean Air Act (O.R.S. §§ 433.835 et seq. (2007)) passed to remove preemption and include bars, restaurants, and all workplaces as smoke-free.""",
    """Roachdale is a town in Franklin and Jackson townships, Putnam County, in the U.S. state of Indiana. The population was 926 at the 2010 census.""",
    """He works for Prof. Dr. Prename Lastname.""",
    """She was entitled Dr. Prename Lastname.""",
    """He is born in Washington D.C. in the U.S.A. and lived there.""",
    """I did three things, e.g. one thing.""",
    """I did more, e. g. another thing.""",
    """Read sentences, i.e. this sentence.""",
    """She met Mr. Lastname and Mrs. Lastname at their house.""",
    """The vote elected Mr. Lastname as president.""",
    """The vote elected Mrs. Lastname as president.""",
    """Prename Lastname (ca. 1950-2000) lived."""
]


if __name__ == "__main__":
    if "i" in sys.argv:
        paragraphs = interactive_sequence_generator()
    elif "t" in sys.argv:
        paragraphs = []
        wiki_paragraphs = read_sequences(paths.WIKI_TRAINING_PARAGRAPHS)
        for _ in range(1000):
            paragraphs.append(next(wiki_paragraphs))
    else:
        paragraphs = test_sequences

    if "spacy" in sys.argv:
        splitter = SpacySentenceSplitter()
    elif "wiki" in sys.argv:
        print("loading wiki punkt tokenizer...")
        splitter = WikiPunktTokenizer()
    else:
        splitter = NLTKSentenceSplitter()
    total_runtime = 0
    for paragraph in paragraphs:
        print(paragraph)
        start_time = timestamp()
        sentences = splitter.split(paragraph)
        runtime = time_diff(start_time)
        total_runtime += runtime
        for s in sentences:
            print(">", s)
        print()
    print("total runtime:", total_runtime)
