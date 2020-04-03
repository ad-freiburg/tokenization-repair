import random


def flip_coin(random_obj: random.Random, p: float) -> bool:
    return random_obj.uniform(0, 1) < p


class TokenCorruptor:
    def __init__(self, p=None, positions_per_token=None, token_pairs_per_token=None, p_insert=None, p_delete=None,
                 seed=42, insert=True, delete=True):
        if p is not None and positions_per_token is not None and token_pairs_per_token is not None:
            self.p_insert = p / 2 / positions_per_token
            self.p_delete = p / 2 / token_pairs_per_token
        elif p_insert is not None and p_delete is not None:
            self.p_insert = p_insert
            self.p_delete = p_delete
        else:
            raise Exception("Either (p and positions_per_token and token_pairs_per_token) " 
                            "or (p_insert and p_delete) must be set.")
        self.n_insertions = 0
        self.n_deletions = 0
        self.n_insertion_candidate_positions = 0
        self.n_deletion_candidate_positions = 0
        self.n_sequences = 0
        self.n_tokens = 0
        self.random = random.Random(seed)
        self.insert = insert
        self.delete = delete

    def corrupt(self, sequence):
        corrupt_sequence = ""
        self.n_tokens += 1
        for pos, char in enumerate(sequence):
            if char == ' ':
                if self.delete and flip_coin(self.random, self.p_delete):
                    self.n_deletions += 1
                else:
                    corrupt_sequence += char
                self.n_tokens += 1
                self.n_deletion_candidate_positions += 1
            else:
                if pos > 0 and char != ' ' and sequence[pos - 1] != ' ':
                    if self.insert and flip_coin(self.random, self.p_insert):
                        corrupt_sequence += ' '
                        self.n_insertions += 1
                    self.n_insertion_candidate_positions += 1
                corrupt_sequence += char
        self.n_sequences += 1
        if self.n_sequences % 10000 == 0:
            self.print_summary()
        return corrupt_sequence

    def print_summary(self):
        print("TokenCorruptor: %i insertions (of %i) and %i deletions (of %i) in %i tokens in %i sequences." %
              (self.n_insertions, self.n_insertion_candidate_positions, self.n_deletions,
               self.n_deletion_candidate_positions, self.n_tokens, self.n_sequences))
