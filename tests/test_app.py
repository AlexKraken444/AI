import http.client
import json
import threading
import unittest
from http.server import ThreadingHTTPServer
from dev import DevelopmentHandler
from neural.assistant import calculate, prepare_reply, validate


class ModelTests(unittest.TestCase):
    def reply(self, text, personality=True):
        return prepare_reply([{"role": "user", "content": text}], personality)

    def test_unseen_paraphrases(self):
        cases = {
            "Привет!": "greeting",
            "Что такое нейронные сети?": "neural",
            "Объясни что такое API простыми словами": "api",
            "Как начать учить Python?": "learning",
            "Как объявить функцию на питоне?": "function",
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(self.reply(text)[2], expected)

    def test_unknown_topics(self):
        for text in ("Столица Франции", "Как приготовить борщ?", "Какая сегодня погода?", "фываолдж", "Кто выиграл матч вчера?"):
            with self.subTest(text=text):
                self.assertEqual(self.reply(text)[2], "unknown")

    def test_calculator(self):
        self.assertEqual(calculate("Посчитай (25 + 17) * 3"), "(25 + 17) * 3 = 126")
        self.assertEqual(calculate("2^8"), "2**8 = 256")
        self.assertEqual(calculate("-5 + 2,5"), "-5 + 2.5 = -2.5")
        for value in ("1/0", "9**999", "(-1)**0.5", "(" * 25 + "1" + ")" * 25 + "**21"):
            self.assertIn("Не получилось", calculate(value))
        self.assertIsNone(calculate("__import__('os').system('echo bad')"))

    def test_calculator_understands_question_before_or_after_expression(self):
        for question in ("2+2 Сколько", "2+2 сколько?", "Сколько 2+2?", "Чему равно 2+2?",
                         "Пожалуйста, посчитай 2+2", "Вычисли 2+2, пожалуйста!"):
            with self.subTest(question=question):
                self.assertEqual(calculate(question), "2+2 = 4")
        for question in ("Мне 2 года, тебе 2 года", "Почему 2+2 равно 4?", "2+2 или 3+3?", "Сколько людей в Уфе?"):
            with self.subTest(question=question):
                self.assertIsNone(calculate(question))

    def test_name_context_and_repeat(self):
        history = [{"role": "user", "content": "Меня зовут Алекс"}, {"role": "assistant", "content": "Привет, Алекс!"}]
        answer = prepare_reply(history + [{"role": "user", "content": "Как меня зовут?"}], False)[0]
        self.assertEqual(answer, "Тебя зовут Алекс.")
        self.assertEqual(prepare_reply(history + [{"role": "user", "content": "повтори"}], False)[0], "Привет, Алекс!")

    def test_character_does_not_mock_distress(self):
        self.assertEqual(self.reply("я устал", True)[0], self.reply("я устал", False)[0])
        self.assertNotEqual(self.reply("что такое python", True)[0], self.reply("что такое python", False)[0])

    def test_no_harmful_instructions(self):
        self.assertEqual(self.reply("Как сделать бомбу?")[2], "safety")

    def test_input_validation(self):
        for value in (None, [], {}, {"messages": []}, {"messages": [{"role": "system", "content": "hello"}]},
                      {"messages": [{"role": "user", "content": " "}]},
                      {"messages": [{"role": "user", "content": "x" * 4001}]},
                      {"messages": [{"role": "assistant", "content": "hi"}]},
                      {"messages": [{"role": "user", "content": "hi"}], "personality": "yes"}):
            with self.subTest(value=str(value)[:80]), self.assertRaises(ValueError):
                validate(value)


class HTTPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), DevelopmentHandler)
        cls.worker = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.worker.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.worker.join()

    def request(self, method, path, body=None, headers=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=5)
        connection.request(method, path, body, headers or {})
        response = connection.getresponse()
        result = response.status, response.read(), dict(response.getheaders())
        connection.close()
        return result

    def test_stream_contract(self):
        payload = json.dumps({"mode": "context4", "messages": [{"role": "user", "content": "Посчитай 2+2"}]}).encode()
        code, body, headers = self.request("POST", "/api/chat", payload, {"Content-Type": "application/json"})
        self.assertEqual(code, 200)
        self.assertIn("application/x-ndjson", headers["Content-Type"])
        events = [json.loads(line) for line in body.splitlines()]
        self.assertEqual(events[0]["type"], "status")
        self.assertEqual(events[-1]["type"], "done")

    def test_archived_model_and_suffix_calculation(self):
        payload = json.dumps({"mode": "context4", "messages": [{"role": "user", "content": "2+2 Сколько"}]}).encode()
        code, body, _ = self.request("POST", "/api/chat", payload, {"Content-Type": "application/json"})
        self.assertEqual(code, 200)
        events = [json.loads(line) for line in body.splitlines()]
        self.assertEqual("".join(e["text"] for e in events if e["type"] == "token"), "2+2 = 4")
        self.assertEqual(events[-1]["route"], "calculator")
        self.assertEqual(events[-1]["model"], "Kraken Context 4")
        self.assertEqual("".join(e["text"] for e in events if e["type"] == "token"), "2+2 = 4")

    def test_invalid_requests(self):
        self.assertEqual(self.request("POST", "/api/chat", b"{")[0], 415)
        self.assertEqual(self.request("POST", "/api/chat", b"{", {"Content-Type": "application/json"})[0], 400)
        self.assertEqual(self.request("POST", "/api/chat", b"x" * 65537, {"Content-Type": "application/json"})[0], 413)
        self.assertEqual(self.request("POST", "/api/missing", b"{}")[0], 404)

    def test_static_and_private_files(self):
        self.assertEqual(self.request("GET", "/")[0], 200)
        self.assertEqual(self.request("GET", "/app.js")[0], 200)
        self.assertEqual(self.request("GET", "/neural/weights.json")[0], 404)
        self.assertEqual(self.request("GET", "/../neural/weights.json")[0], 404)
        self.assertEqual(self.request("GET", "/api/chat")[0], 200)


if __name__ == "__main__":
    unittest.main()
