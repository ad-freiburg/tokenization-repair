from typing import Optional, List

from src.ngram.tokenizer import Tokenizer, Token, tokens2sequence
from src.ngram.unigram_holder import UnigramHolder
from src.ngram.bigram_holder import BigramHolder


class CorrectionCandidate:
    def __init__(self, score: int, tokens: List[Token], consume_previous: bool, consume_next: bool):
        self.score = score
        self.tokens = tokens
        self.consume_previous = consume_previous
        self.consume_next = consume_next

    def __str__(self):
        return "CorrectionCandidate(%i, %s)" % (self.score, str(self.tokens))

    def __repr__(self):
        return str(self)


class UnigramCorrector:
    def __init__(self, n: Optional[int]):
        self.tokenizer = Tokenizer()
        self.holder = UnigramHolder(n)
        self.bigrams = BigramHolder.load()

    def split_candidates(self, token: Token) -> List[CorrectionCandidate]:
        text = token.text
        candidates = []
        for i in range(1, len(text)):
            left = text[:i]
            right = text[i:]
            frequency = self.bigrams.get((left, right))
            if frequency > 0:
                tokens =  [Token(left, token.delimiter_before),
                           Token(right, True)]
                candidates.append(CorrectionCandidate(frequency,
                                                      tokens,
                                                      False,
                                                      False))
        return candidates

    def merge_candidates(self,
                         token: Token,
                         previous_token: Optional[Token],
                         next_token: Optional[Token]):
        candidates = []
        if previous_token is not None:
            merged = previous_token.text + token.text
            frequency = self.holder.get(merged)
            if frequency > 0:
                candidates.append(CorrectionCandidate(frequency,
                                                      [Token(merged, previous_token.delimiter_before)],
                                                      consume_previous=True,
                                                      consume_next=False))
        if next_token is not None:
            merged = token.text + next_token.text
            frequency = self.holder.get(merged)
            if frequency > 0:
                candidates.append(CorrectionCandidate(frequency,
                                                      [Token(merged, token.delimiter_before)],
                                                      consume_previous=False,
                                                      consume_next=True))
        return candidates

    def select_best_candidate(self, candidates: List[CorrectionCandidate]):
        best_score = -1
        best = None
        for candidate in candidates:
            if candidate.score > best_score:
                best_score = candidate.score
                best = candidate
        return best

    def repair_token(self, token: Token, next_token: Optional[Token], predicted_tokens: List[Token]) -> int:
        candidates = [CorrectionCandidate(self.holder.get(token.text), [token], False, False)]
        candidates.extend(self.split_candidates(token))
        previous_token = predicted_tokens[-1] if len(predicted_tokens) > 0 else None
        candidates.extend(self.merge_candidates(token, previous_token, next_token))
        if len(candidates) > 0:
            best_candidate = self.select_best_candidate(candidates)
            if best_candidate.consume_previous:
                predicted_tokens.pop()
            predicted_tokens.extend(best_candidate.tokens)
            return 2 if best_candidate.consume_next else 1
        predicted_tokens.append(token)
        return 1

    def correct(self, sequence: str) -> str:
        tokens = self.tokenizer.tokenize(sequence)
        n_tokens = len(tokens)
        tokens.append(None)
        predicted_tokens = []
        t_i = 0
        while t_i < n_tokens:
            token = tokens[t_i]
            next_token = tokens[t_i + 1]
            t_i += self.repair_token(token, next_token, predicted_tokens)
        predicted = tokens2sequence(predicted_tokens)
        return predicted