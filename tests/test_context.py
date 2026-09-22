import json
import unittest
import numpy as np
from neural.transformer import Transformer
from neural.tokenizer import Tokenizer, prompt_tokens, USER, ASSISTANT, MEMORY, END
from neural.memory import validate_memory, memory_reply, memory_context
from neural.dialogue import reply_events


class TransformerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = Transformer()

    def test_tokenizer_roundtrip_and_role_boundaries(self):
        text = "Аня любит чай, а Борис — кофе.\nПример: print(42)"
        self.assertEqual(self.model.tokenizer.decode(self.model.tokenizer.encode(text)), text)
        # Writing a literal special-token name cannot inject an actual role delimiter.
        self.assertNotIn(ASSISTANT, self.model.tokenizer.encode("<assistant>"))

    def test_future_tokens_do_not_change_past_logits(self):
        a = self.model.tokenizer.encode("Аня любит чай")
        b = self.model.tokenizer.encode("Борис любит кофе")
        first, _, _ = self.model.forward(a)
        longer, _, _ = self.model.forward(a + b)
        np.testing.assert_allclose(first, longer[:len(a)], atol=3e-5, rtol=3e-5)

    def test_cache_matches_full_forward(self):
        ids = prompt_tokens(self.model.tokenizer, [{"role":"user", "content":"Что такое Python?"}])
        first, cache, _ = self.model.forward(ids[:-1])
        last, _, _ = self.model.forward(ids[-1:], cache)
        full, _, _ = self.model.forward(ids)
        np.testing.assert_allclose(last[-1], full[-1], atol=3e-5, rtol=3e-5)

    def test_order_changes_logits(self):
        a = prompt_tokens(self.model.tokenizer,[{"role":"user","content":"Аня любит чай. Борис любит кофе. Что любит Аня?"}])
        b = prompt_tokens(self.model.tokenizer,[{"role":"user","content":"Борис любит чай. Аня любит кофе. Что любит Аня?"}])
        left, _, _ = self.model.forward(a)
        right, _, _ = self.model.forward(b)
        self.assertGreater(float(np.abs(left[-1] - right[-1]).max()), .1)

    def test_long_context_is_bounded_and_latest_question_kept(self):
        messages = [{"role":"user","content":"текст " * 1000}, {"role":"assistant","content":"ответ " * 1000}, {"role":"user","content":"Последний вопрос?"}]
        ids = prompt_tokens(self.model.tokenizer, messages, "факт " * 1000, limit=100)
        self.assertLessEqual(len(ids),100)
        self.assertIn("Последний вопрос?",self.model.tokenizer.decode(ids))
        self.assertEqual(ids[-1],ASSISTANT)

    def test_generation_events_are_real_incremental_tokens(self):
        events = list(self.model.generate_events([{"role":"user","content":"Привет!"}],max_tokens=50))
        self.assertEqual(events[-1]["type"],"generation_done")
        self.assertTrue(any(e["type"] == "token" for e in events))
        self.assertLessEqual(events[-1]["tokens"],50)

    def test_new_model_is_an_explicit_separate_mode(self):
        from neural.dialogue import language_model
        baseline=language_model("context")
        candidate=language_model("context3")
        self.assertNotEqual(baseline.config["version"],candidate.config["version"])
        self.assertEqual(candidate.config["context"],512)
        events=list(reply_events([{"role":"user","content":"Привет!"}],False,validate_memory(None),"context3"))
        self.assertEqual(events[-1]["type"],"done")
        self.assertEqual(events[-1]["model"],"Kraken Context 3")


class MemoryTests(unittest.TestCase):
    def sample(self):
        return validate_memory({"enabled":True,"facts":[{"id":"one","chatId":"old","key":"name","value":"Алекс","text":"Имя: Алекс"}],"excerpts":[]})

    def test_memory_fact_and_sources(self):
        answer, sources = memory_reply("Как меня зовут?",self.sample())
        self.assertEqual(answer,"Тебя зовут Алекс.")
        self.assertEqual(sources[0]["chatId"],"old")

    def test_request_memory_never_leaks_to_next_request(self):
        first = list(reply_events([{"role":"user","content":"Как меня зовут?"}],True,self.sample()))
        second = list(reply_events([{"role":"user","content":"Как меня зовут?"}],True,validate_memory(None)))
        self.assertIn("Алекс","".join(e.get("text","") for e in first))
        self.assertNotIn("Алекс","".join(e.get("text","") for e in second))

    def test_disabled_memory_drops_even_supplied_facts(self):
        value = self.sample(); value["enabled"] = False
        self.assertEqual(validate_memory(value)["facts"],[])

    def test_memory_bounds(self):
        for payload in ({"enabled":"yes"}, {"enabled":True,"facts":[{}]}, {"enabled":True,"facts":[{}]*9}, {"enabled":True,"excerpts":[{}]*5}):
            with self.subTest(payload=payload),self.assertRaises(ValueError):
                validate_memory(payload)

    def test_grounded_history_recall(self):
        memory = validate_memory({"enabled":True,"facts":[],"excerpts":[{"id":"m","chatId":"a","title":"Python","text":"Хочу изучать Python"}]})
        answer, sources = memory_reply("Что мы обсуждали в прошлом чате?",memory)
        self.assertIn("Хочу изучать Python",answer)
        self.assertEqual(len(sources),1)

    def test_unsaved_secret_is_not_acknowledged(self):
        events = list(reply_events([{"role":"user","content":"Запомни: пароль abc"}],True,self.sample()))
        self.assertIn("Не вижу сохранённой записи","".join(e.get("text","") for e in events))

    def test_sources_fit_in_actual_model_memory_budget(self):
        model=Transformer()
        messages=[{"role":"user","content":"длинный "*500}]
        context,sources=memory_context(self.sample(),model.tokenizer,messages)
        self.assertEqual(context,"")
        self.assertEqual(sources,[])


if __name__ == "__main__":
    unittest.main()
