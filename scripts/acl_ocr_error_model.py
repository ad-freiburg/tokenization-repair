import project
from src.noise.acl_noise_inducer import ACLNoiseInducer


if __name__ == "__main__":
    errors_tsv_file = "ocr_error_frequencies.tsv"
    inducer = ACLNoiseInducer(0.1, 0.2, 42)
    error_dict = inducer.error_dict
    print(len([e for e, f in error_dict[""] if f >= 0]), "insertions")
    total_count = 0
    insertion_count = 0
    for correct in error_dict:
        for wrong, freq in error_dict[correct]:
            total_count += freq
            if correct == "":
                insertion_count += freq
    insertion_prob = insertion_count / total_count
    print(f"{insertion_prob * 100:.2f}% char insertions ({insertion_count}/{total_count})")
    while True:
        sequence = input("> ")
        for _ in range(100):
            corrupt = inducer.induce_noise(sequence)
            print(corrupt)
