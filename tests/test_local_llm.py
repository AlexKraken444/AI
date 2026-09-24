import io
import json
import os
import unittest
from unittest.mock import patch
from urllib.error import URLError

from neural.local_llm import build_messages, reply_events
from neural.memory import validate_memory


class LocalModelTests(unittest.TestCase):
    def test_second_question_is_last_and_history_stays_in_order(self):
        history = [{"role": "user", "content": "Как дела?"},
                   {"role": "assistant", "content": "Готов помочь."},
                   {"role": "user", "content": "Что такое Python?"}]
        prepared = build_messages(history, True, validate_memory(None))
        self.assertEqual(prepared[1:], history)
        self.assertEqual(prepared[-1], history[-1])

    def test_context_limit_never_removes_latest_question(self):
        history = [{"role": "user" if i % 2 == 0 else "assistant", "content": str(i) * 2000} for i in range(22)]
        latest = {"role": "user", "content": "Расскажи о Луне."}
        prepared = build_messages(history + [latest], False, validate_memory(None))
        self.assertEqual(prepared[-1], latest)
        self.assertEqual(prepared[1]["role"], "user")
        self.assertLessEqual(sum(len(m["content"]) for m in prepared), 10000)

    @patch.dict(os.environ, {"GPU_CHAT_URL": "https://owner.example", "GPU_CHAT_KEY": "private-test-key"})
    @patch("neural.local_llm.urlopen")
    def test_each_request_uses_current_question_and_only_visible_tokens(self, urlopen):
        def response(request, timeout):
            payload = json.loads(request.data)
            answer = payload["messages"][-1]["content"]
            chunks = [{"choices": [{"delta": {"reasoning_content": "hidden"}}]},
                      {"choices": [{"delta": {"content": answer}}]},
                      {"choices": [{"delta": {}, "finish_reason": "stop"}]}]
            return io.BytesIO(b"".join(b"data: " + json.dumps(c).encode() + b"\n\n" for c in chunks) + b"data: [DONE]\n\n")
        urlopen.side_effect = response
        history = []
        for question in ["Как дела?", "Что такое Python?", "А чем он отличается от Java?"]:
            history.append({"role": "user", "content": question})
            events = list(reply_events(history, True, validate_memory(None)))
            answer = "".join(e["text"] for e in events if e["type"] == "token")
            self.assertEqual(answer, question)
            self.assertEqual(events[-1]["type"], "done")
            history.append({"role": "assistant", "content": answer})
        self.assertEqual(urlopen.call_count, 3)

    @patch.dict(os.environ, {"GPU_CHAT_URL": "https://owner.example", "GPU_CHAT_KEY": "private-test-key"})
    @patch("neural.local_llm.urlopen", side_effect=URLError("private-test-key"))
    def test_offline_does_not_fall_back_or_leak_secrets(self, _):
        events = list(reply_events([{"role": "user", "content": "Привет"}], True, validate_memory(None)))
        self.assertEqual(events[-1]["type"], "error")
        self.assertNotIn("private-test-key", json.dumps(events))
        self.assertFalse(any(e["type"] in ("done", "token") for e in events))

    def test_memory_is_isolated_per_request(self):
        memory = {"enabled": True, "facts": [{"text": "Имя: Маша"}], "excerpts": []}
        messages = [{"role": "user", "content": "Как меня зовут?"}]
        self.assertIn("Маша", build_messages(messages, True, memory)[0]["content"])
        self.assertNotIn("Маша", build_messages(messages, True, validate_memory(None))[0]["content"])

    @patch.dict(os.environ, {"GPU_CHAT_URL": "https://example.trycloudflare.com", "GPU_CHAT_KEY": "test"})
    @patch("neural.local_llm.urlopen")
    def test_buffered_tunnel_preserves_answer_and_disables_sse(self, urlopen):
        urlopen.return_value = io.BytesIO(json.dumps({"choices": [{"message": {"content": "Ответ", "reasoning_content": "hidden"}, "finish_reason": "stop"}]}).encode())
        events = list(reply_events([{"role": "user", "content": "Вопрос"}], False, validate_memory(None)))
        self.assertFalse(json.loads(urlopen.call_args.args[0].data)["stream"])
        self.assertEqual(events[-2], {"type": "token", "text": "Ответ"})
        self.assertEqual(events[-1]["type"], "done")

    @patch.dict(os.environ, {"GPU_CHAT_URL": "https://owner.example", "GPU_CHAT_KEY": "test"})
    @patch("neural.local_llm.urlopen", return_value=io.BytesIO(b'data: {"choices":[{"delta":{"content":"partial"}}]}\n\n'))
    def test_broken_stream_is_not_reported_as_complete(self, _):
        events = list(reply_events([{"role": "user", "content": "Вопрос"}], False, validate_memory(None)))
        self.assertEqual(events[-1]["type"], "error")
        self.assertFalse(any(e["type"] == "done" for e in events))
