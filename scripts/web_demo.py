import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

import project
from src.server.backend import Backend


class RequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        query = self.path
        if query.startswith("/"):
            query = query[1:]

        answer = backend.answer(query)

        self.protocol_version = "HTTP/1.1"
        self.send_response(200)
        self.send_header("Content-Length", len(answer))
        self.end_headers()

        self.wfile.write(bytes(answer, "utf8"))
        return


print("loading backend...")
backend = Backend()


def run():
    if len(sys.argv) > 1:
        port = int(sys.argv[1])
    else:
        port = 1234
    server = ('', port)
    httpd = HTTPServer(server, RequestHandler)
    print("serving at port", port)
    httpd.serve_forever()


run()
