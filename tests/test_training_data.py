import json
import tempfile
import unittest
from pathlib import Path
from neural.tokenizer import Tokenizer, ASSISTANT
from training.prepare_data import split_examples, oasst_examples, acceptable


class TokenizerTrainingTests(unittest.TestCase):
    def test_bpe_roundtrip_and_persistence(self):
        texts=["Аня любит чай. Борис любит кофе.","Аня любит кофе. Борис любит чай.","Python print(42)"]*5
        tokenizer=Tokenizer.fit_bpe(texts,vocabulary=140)
        for text in texts:
            self.assertEqual(tokenizer.decode(tokenizer.encode(text)),text)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"tokenizer.json"
            tokenizer.save(path)
            loaded=Tokenizer.load(path)
            self.assertEqual(loaded.encode(texts[0]),tokenizer.encode(texts[0]))
        self.assertNotIn(ASSISTANT,tokenizer.encode("<assistant>"))

    def test_bpe_cache_does_not_retain_arbitrary_user_words(self):
        tokenizer=Tokenizer.fit_bpe(["simple words and characters"],vocabulary=100)
        tokenizer.encode("privatepasswordxyz")
        self.assertNotIn("privatepasswordxyz",tokenizer.cache)


class DatasetTests(unittest.TestCase):
    def test_groups_and_duplicate_inputs_do_not_cross_splits(self):
        examples=[{"group":str(i//3),"messages":[{"role":"user","content":str(i)}],"answer":"x","memory":""} for i in range(150)]
        examples.append(dict(examples[0],group="different"))
        splits=split_examples(examples)
        sets={name:{e["group"] for e in rows} for name,rows in splits.items()}
        self.assertFalse(sets["train"] & sets["validation"])
        self.assertFalse(sets["train"] & sets["test"])
        self.assertFalse(sets["validation"] & sets["test"])
        self.assertEqual(sum(len(rows) for rows in splits.values()),150)

    def row(self,id,role,parent=None,text="Пример текста"):
        return {"message_id":id,"role":role,"parent_id":parent,"message_tree_id":"tree","lang":"ru","deleted":False,"review_result":True,"rank":0,"text":text,"labels":{"quality":{"value":1}}}

    def test_reconstructs_multi_turn_context_and_preserves_source(self):
        rows=[self.row("1","prompter"),self.row("2","assistant","1"),self.row("3","prompter","2","Уточнение вопроса"),self.row("4","assistant","3","Ответ на уточнение")]
        examples,_=oasst_examples(rows)
        self.assertEqual(len(examples),2)
        self.assertEqual(len(examples[1]["messages"]),3)
        self.assertEqual(examples[1]["source_id"],"4")
        self.assertEqual(examples[0]["group"],examples[1]["group"])

    def test_bad_ancestor_cannot_be_silently_dropped(self):
        rows=[self.row("1","prompter"),self.row("2","assistant","1")]
        rows[0]["deleted"]=True
        self.assertEqual(oasst_examples(rows)[0],[])
        self.assertEqual(oasst_examples([self.row("2","assistant","missing")])[0],[])

    def test_excludes_negative_reviews_and_contact_details(self):
        row=self.row("1","assistant",text="Напиши мне на test@example.com")
        self.assertFalse(acceptable(row))
        row=self.row("2","assistant"); row["review_result"]=False
        self.assertFalse(acceptable(row))


if __name__=="__main__": unittest.main()
