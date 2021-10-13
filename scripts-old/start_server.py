import project
from src.server.server import Server


if __name__ == "__main__":
    PORT = 1234
    server = Server()
    server.serve(PORT)
