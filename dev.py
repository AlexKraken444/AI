"""Same Python handler locally, static files served only from public/."""
import argparse
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit
from api.chat import handler


class DevelopmentHandler(handler, SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(Path(__file__).parent / "public"), **kwargs)

    def do_GET(self):
        if urlsplit(self.path).path == "/api/chat":
            return handler.do_GET(self)
        return SimpleHTTPRequestHandler.do_GET(self)

    def do_POST(self):
        if urlsplit(self.path).path != "/api/chat":
            return self.json_response(404, {"error": "Not found"})
        return handler.do_POST(self)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    print(f"Kraken: http://127.0.0.1:{args.port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), DevelopmentHandler).serve_forever()
