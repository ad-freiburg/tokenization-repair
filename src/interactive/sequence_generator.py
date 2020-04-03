

def interactive_sequence_generator():
    while True:
        sequence = input("> ")
        if sequence == "exit":
            break
        yield sequence
