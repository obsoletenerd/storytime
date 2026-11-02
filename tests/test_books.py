import io
import tempfile
import unittest
import sqlite3
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import app as storytime


class BookFlowTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        storytime.app.config.update(TESTING=True, DATA_DIR=self.temp.name)
        self.client = storytime.app.test_client()

    def tearDown(self):
        self.temp.cleanup()

    def create(self, format="ongoing", cover="none"):
        fields = dict(title="Moon Harbour", format=format, characters="Rex and Mira",
                      themes="puzzles", premise="A lighthouse mystery", reader_notes="Age 9",
                      provider="ollama", cover_choice=cover)
        if cover == "upload":
            fields["cover"] = (io.BytesIO(b"\x89PNG\r\n\x1a\n" + b"a" * 20), "cover.png")
        with patch.object(storytime, "generate_text", return_value="The moonlit harbour was quiet.\n\nRex found a key."):
            return self.client.post("/books", data=fields, follow_redirects=True)

    def test_ongoing_book_can_be_reopened_and_guided(self):
        response = self.create()
        self.assertIn(b"Moon Harbour", response.data)
        with storytime.db() as connection:
            book_id = connection.execute("SELECT id FROM books").fetchone()[0]
        self.assertIn(b"Write", self.client.get("/").data)
        self.assertIn(b"Write Chapter 2", self.client.get(f"/books/{book_id}/write").data)
        with patch.object(storytime, "generate_text", return_value="Mira opened the lighthouse door.") as generated:
            response = self.client.post(f"/books/{book_id}/chapters",
                                        data={"guidance": "Visit the lighthouse", "provider": "ollama"},
                                        follow_redirects=True)
        self.assertIn(b"Chapter 2", response.data)
        self.assertTrue(any("Visit the lighthouse" in call.args[1] for call in generated.call_args_list))
        with storytime.db() as connection:
            chapters = storytime.chapters_for(connection, book_id)
        self.assertEqual([c["number"] for c in chapters], [1, 2])
        self.assertEqual(chapters[1]["guidance"], "Visit the lighthouse")
        self.assertTrue(chapters[0]["summary"])
        self.assertTrue(chapters[1]["summary"])

    def test_existing_v2_database_gains_summary_column(self):
        old = sqlite3.connect(Path(self.temp.name, "storytime-v2.db"))
        old.execute("CREATE TABLE chapters (id TEXT PRIMARY KEY, book_id TEXT, number INTEGER, "
                    "title TEXT, content TEXT, guidance TEXT, provider TEXT, created_at TEXT)")
        old.commit()
        old.close()
        with storytime.db() as connection:
            names = {row[1] for row in connection.execute("PRAGMA table_info(chapters)")}
        self.assertIn("summary", names)

    def test_standalone_cannot_be_continued(self):
        self.create(format="standalone")
        with storytime.db() as connection:
            book_id = connection.execute("SELECT id FROM books").fetchone()[0]
        self.assertEqual(self.client.get(f"/books/{book_id}/write").status_code, 404)
        self.assertEqual(self.client.post(f"/books/{book_id}/chapters").status_code, 404)

    def test_uploaded_cover_is_served_and_story_is_escaped(self):
        self.create(cover="upload")
        with storytime.db() as connection:
            book = connection.execute("SELECT * FROM books").fetchone()
        self.assertTrue(Path(self.temp.name, "covers", book["cover_path"]).exists())
        response = self.client.get(f"/covers/{book['cover_path']}")
        self.assertEqual(response.status_code, 200)
        response.close()
        with storytime.app.app_context():
            self.assertNotIn("<script>", str(storytime.render_story("<script>alert(1)</script>")))

    def test_generated_cover_uses_supported_model_and_base64_image(self):
        image = SimpleNamespace(b64_json="iVBORw0KGgphYQ==", url=None)
        with patch.dict(storytime.os.environ, {"OPENAI_API_KEY": "test-key"}):
            with patch.object(storytime, "OpenAI") as client:
                client.return_value.images.generate.return_value.data = [image]
                book = dict(id="example", title="Moon Harbour", characters="Rex",
                            themes="puzzles", premise="a lighthouse")
                name = storytime.generate_cover(book)
                self.assertTrue(Path(self.temp.name, "covers", name).exists())
                self.assertEqual(client.return_value.images.generate.call_args.kwargs["model"],
                                 "gpt-image-2.5-flare")
                self.assertEqual(client.return_value.images.generate.call_args.kwargs["quality"], "medium")

    def test_twenty_chapter_prompt_keeps_early_summary_and_latest_text(self):
        self.create()
        with storytime.db() as connection:
            book = connection.execute("SELECT * FROM books").fetchone()
            for number in range(2, 21):
                connection.execute("INSERT INTO chapters (id,book_id,number,title,content,summary,provider) "
                                   "VALUES (?,?,?,?,?,?,?)",
                                   (f"chapter-{number}", book["id"], number, f"Chapter {number}",
                                    f"Full chapter {number}. " * 100, f"Summary of chapter {number}.", "ollama"))
            chapters = storytime.chapters_for(connection, book["id"])
        prompt = storytime.next_prompt(book, chapters, "Remember the chapter 2 key")
        self.assertIn("Summary of chapter 2", prompt)
        self.assertIn("Full chapter 20", prompt)
        self.assertIn("Remember the chapter 2 key", prompt)

    def test_summary_failure_preserves_chapter(self):
        self.create()
        with storytime.db() as connection:
            book = connection.execute("SELECT * FROM books").fetchone()
        with patch.object(storytime, "generate_text", side_effect=["A new chapter.", RuntimeError("summary offline")]):
            response = self.client.post(f"/books/{book['id']}/chapters",
                                        data={"provider": "ollama"}, follow_redirects=True)
        self.assertIn(b"A new chapter.", response.data)
        with storytime.db() as connection:
            chapters = storytime.chapters_for(connection, book["id"])
            updated_book = storytime.book_or_404(connection, book["id"])
        self.assertEqual(len(chapters), 2)
        self.assertEqual(chapters[1]["summary"], "")
        prompt = storytime.next_prompt(updated_book, chapters, "Continue")
        self.assertIn("Latest chapter (2): A new chapter.", prompt)


if __name__ == "__main__":
    unittest.main()
