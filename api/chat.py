import json
import logging
import sys
from http.server import BaseHTTPRequestHandler
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from neural.assistant import validate
from neural.memory import validate_memory
from neural.dialogue import reply_events


class handler(BaseHTTPRequestHandler):
    def json_response(self, code, body):
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        self.json_response(200, {"ok": True, "model": "Kraken Context 2", "type": "causal transformer", "external_ai": False, "memory": "browser-owned per-request context"})

    def do_POST(self):
        if self.headers.get("Content-Type", "").split(";")[0].strip().lower() != "application/json":
            self.json_response(415, {"error": "Используйте Content-Type: application/json."})
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= 65536:
                self.json_response(413, {"error": "Запрос слишком большой или пустой."})
                return
            payload = json.loads(self.rfile.read(size).decode("utf-8"))
            messages, personality = validate(payload)
            memory = validate_memory(payload.get("memory"))
            mode = payload.get("mode", "context")
            if mode not in ("context", "context3", "reference"):
                raise ValueError("Unknown model mode")
        except (ValueError, UnicodeDecodeError):
            self.json_response(400, {"error": "Некорректный запрос: до 24 сообщений, каждое от 1 до 4000 символов."})
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
        self.send_header("Cache-Control", "no-store, no-transform")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()

        def emit(event, **data):
            self.wfile.write((json.dumps({"type": event, **data}, ensure_ascii=False) + "\n").encode("utf-8"))
            self.wfile.flush()

        try:
            for event in reply_events(messages, personality, memory, mode):
                emit(event["type"], **{key: value for key, value in event.items() if key != "type"})
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            return
        except Exception:
            logging.exception("Kraken inference failed")
            emit("error", text="Не удалось получить ответ модели. Попробуйте ещё раз.")

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Allow", "GET, POST, OPTIONS")
        self.end_headers()
