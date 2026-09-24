import unittest
from unittest.mock import patch
from neural.facts import fact_reply
from neural.dialogue import reply_events
from neural.memory import validate_memory


class FactTests(unittest.TestCase):
    def test_name_and_obvious_typos(self):
        for question in ["Президент России как зовут?", "Призедент Росии как завут?", "Кто президент РФ?", "Как зовут президента США?"]:
            if "США" in question:
                self.assertNotIn("Путин", fact_reply(question) or "")
            else:
                self.assertIn("Владимир Владимирович Путин", fact_reply(question))
                self.assertIn("https://kremlin.ru", fact_reply(question))

    def test_dont_confuse_history_and_policy_with_name(self):
        for question in ["Кто президент России в 2009?", "Почему президент России принял закон?", "Кто президент России и Франции?"]:
            self.assertNotIn("Путин", fact_reply(question) or "")

    def test_second_question_uses_reference_not_previous_answer(self):
        with patch("neural.dialogue.language_model", side_effect=AssertionError("must not generate fact")):
            events = list(reply_events([{"role":"user","content":"Привет"},{"role":"assistant","content":"Привет!"},{"role":"user","content":"Президент России как зовут?"}], True, validate_memory(None)))
        self.assertEqual(events[-1]["route"], "verified_reference")
        self.assertIn("Путин", "".join(e.get("text", "") for e in events))
