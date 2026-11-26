"""
Hızlı birim testleri (GPU/model gerektirmez):
    python -m unittest discover -s tests -v
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from japonca_tts.qc import cer, wav_stats, file_stats          # noqa: E402
from japonca_tts.characters import list_characters, load_character  # noqa: E402
from japonca_tts.synth import _step_of, pick_checkpoint       # noqa: E402


class TestCER(unittest.TestCase):
    def test_identical(self):
        self.assertEqual(cer("コンニチハ", "コンニチハ"), 0.0)

    def test_one_sub(self):
        self.assertAlmostEqual(cer("アイウエオ", "アイウエカ"), 0.2)

    def test_empty_hyp(self):
        self.assertEqual(cer("アイウ", ""), 1.0)

    def test_reading_level(self):
        from japonca_tts.qc import reading_cer
        # Farklı yazım, aynı okunuş -> 0 olmalı
        self.assertEqual(reading_cer("ありがとう", "アリガトウ"), 0.0)


class TestCharacters(unittest.TestCase):
    def test_at_least_five_presets(self):
        names = list_characters()
        self.assertGreaterEqual(len(names), 5, names)

    def test_all_loadable_and_valid(self):
        for n in list_characters():
            c = load_character(n)
            self.assertIn(c.style_id, {"Neutral", "normal", "whisper", "falsetto"}, n)
            self.assertGreater(c.length_scale, 0.5, n)
            self.assertLess(c.length_scale, 2.0, n)
            self.assertTrue(0.0 <= c.sdp_ratio <= 1.0, n)

    def test_presets_are_distinct(self):
        seen = set()
        for n in list_characters():
            c = load_character(n)
            key = (c.speaker_id, c.style_id)
            self.assertNotIn(key, seen, f"{n} başka bir preset ile aynı (speaker, style)")
            seen.add(key)

    def test_unknown_raises(self):
        with self.assertRaises(KeyError):
            load_character("yok_boyle_biri")


class TestSentences(unittest.TestCase):
    def test_twenty_unique_sentences(self):
        data = json.loads((ROOT / "tests" / "test_sentences.json").read_text(encoding="utf-8"))
        s = data["sentences"]
        self.assertEqual(len(s), 20)
        self.assertEqual(len({x["id"] for x in s}), 20)
        self.assertEqual(len({x["text"] for x in s}), 20)
        for x in s:
            self.assertTrue(x["text"].strip())


class TestCheckpointPick(unittest.TestCase):
    def test_step_parse(self):
        self.assertEqual(_step_of(Path("jvs_multi12_e12_s6000.safetensors")), 6000)
        self.assertEqual(_step_of(Path("model.safetensors")), -1)

    def test_pick_highest(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            for n in ["m_e1_s50.safetensors", "m_e10_s500.safetensors", "m_e2_s100.safetensors"]:
                (Path(d) / n).write_bytes(b"x")
            self.assertEqual(pick_checkpoint(Path(d)).name, "m_e10_s500.safetensors")


class TestFileStats(unittest.TestCase):
    def test_missing(self):
        self.assertFalse(wav_stats(Path("yok.wav"))["ok"])
        self.assertFalse(file_stats("yok.mp3")["ok"])

    def test_zero_byte(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "z.wav"
            p.write_bytes(b"")
            self.assertEqual(wav_stats(p)["error"], "0 bayt")


if __name__ == "__main__":
    unittest.main()


class TestTextFrontend(unittest.TestCase):
    """normalize(): SBV2'nin yanlış/eksik bıraktığı girdiler (ölçüldü, 2026-09-17)."""

    def setUp(self):
        from japonca_tts.text_frontend import normalize, split_sentences
        self.n, self.split = normalize, split_sentences

    def test_percent_and_units(self):
        self.assertEqual(self.n("税率10%です。"), "税率10パーセントです。")
        self.assertEqual(self.n("約2.5kmです。"), "約2.5キロメートルです。")
        self.assertEqual(self.n("重さは5kgです。"), "重さは5キログラムです。")

    def test_minus_and_degree(self):
        self.assertEqual(self.n("気温は−3℃です。"), "気温はマイナス3度です。")
        self.assertEqual(self.n("-5度"), "マイナス5度")

    def test_phone_digit_by_digit(self):
        self.assertEqual(self.n("03-1234-5678"), "ゼロサンのイチニーサンヨンのゴーロクナナハチ")

    def test_date_time_fraction(self):
        self.assertEqual(self.n("2026/9/17 15:30"), "2026年9月17日 15時30分")
        self.assertEqual(self.n("2024-12-31"), "2024年12月31日")
        self.assertEqual(self.n("1/2"), "2分の1")

    def test_currency_prefix(self):
        self.assertEqual(self.n("価格は$20です。"), "価格は20ドルです。")
        self.assertEqual(self.n("¥1,500"), "1,500円")

    def test_lexicon(self):
        self.assertEqual(self.n("iPhoneでYouTube"), "アイフォンでユーチューブ")
        self.assertEqual(self.n("OK、行こう！"), "オーケー、行こう！")
        # kelime sınırı: "ok" içeren başka kelime değişmemeli
        self.assertNotIn("オーケー", self.n("Tokyo"))

    def test_punctuation_collapse(self):
        self.assertEqual(self.n("本当に？！"), "本当に？")
        self.assertEqual(self.n("すごい！！！"), "すごい！")
        self.assertEqual(self.n("えっ…"), "えっ…")
        self.assertEqual(self.n("😊ありがとう"), "ありがとう")

    def test_brackets_become_pause(self):
        self.assertEqual(self.n("「こんにちは」と言った。"), "こんにちは、と言った。")

    def test_split_sentences(self):
        text = "一文目です。二文目です！三文目？" + "四" * 100 + "。"
        chunks = self.split(text, max_chars=80)
        self.assertGreaterEqual(len(chunks), 2)
        self.assertEqual("".join(chunks), text)
        self.assertEqual(self.split("短い。"), ["短い。"])

    def test_long_sentence_split_on_comma(self):
        text = "、".join(["これはとても長い節です"] * 12) + "。"
        chunks = self.split(text, max_chars=80)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(c) <= 130 for c in chunks))
