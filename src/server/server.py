import socket
import sys
import time
import re

from src.server.backend import Backend


class Server:
    def __init__(self):
        self.backend = Backend()

    def serve(self, port):
        # Create communication socket and listen on given port.
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((socket.gethostname(), port))
        server.listen(10)
        # Server loop.
        while True:
            print("\x1b[1mWaiting for requests on port %d ... \x1b[0m" % port)
            sys.stdout.flush()
            (client, address) = server.accept()
            print("Incoming request at " + time.ctime())
            request = client.recv(1 << 28).decode("utf8")
            print("Request: \"" + request + "\"")
            self.handle_request(client, request)
            client.close()

    def handle_request(self, client, request):
        match = re.match("^GET /(.*) HTTP", request)
        if not match:
            return
        query = match.group(1)
        print("server query:", query)
        content = self.backend.answer(query)
        content = content.encode(encoding="utf-8")
        content_type = "text/html; charset=utf-8"
        result = ("HTTP/1.1 200 OK\r\n"
                  "Content-type: %s\r\n"
                  "Content-length: %s\r\n\r\n") % (content_type,
                                                   len(content))
        result = result.encode(encoding="utf-8")
        result += content
        client.send(result)
