from __future__ import annotations

import json
import unittest

from config import AUDIO_MMPROJ_PATH, AUDIO_MODEL_PATH, AUDIO_SERVER_EXE, WAKE_WORD
from wake_voice import audio_server_command, extract_transcription


class FakeResponse:
    def __init__(self, body: object, text: str = "") -> None:
        self.body = body
        self.text = text or (body if isinstance(body, str) else json.dumps(body))

    def json(self) -> object:
        return self.body


class AlexaVoiceTests(unittest.TestCase):
    def test_wake_word_is_fixed_to_alexa(self) -> None:
        self.assertEqual(WAKE_WORD, "alexa")

    def test_audio_server_command_uses_portable_runtime_files(self) -> None:
        command = audio_server_command()
        self.assertEqual(command[0], str(AUDIO_SERVER_EXE))
        self.assertIn(str(AUDIO_MODEL_PATH), command)
        self.assertIn(str(AUDIO_MMPROJ_PATH), command)
        self.assertIn("--reasoning", command)
        self.assertIn("off", command)

    def test_extracts_openai_style_transcription_text(self) -> None:
        response = FakeResponse({"text": "Open Chrome"})
        self.assertEqual(extract_transcription(response), "Open Chrome")

    def test_accepts_plain_text_transcription(self) -> None:
        class PlainTextResponse(FakeResponse):
            def json(self) -> object:
                raise ValueError("not JSON")

        response = PlainTextResponse("hello", text=" hello ")
        self.assertEqual(extract_transcription(response), "hello")


if __name__ == "__main__":
    unittest.main()
