"""Storytime V2: books with persistent, guided chapters."""
import base64
import os
import re
import sqlite3
import uuid
from contextlib import contextmanager
from pathlib import Path

import requests
from dotenv import load_dotenv
from flask import Flask, abort, flash, redirect, render_template, request, send_from_directory, url_for
from markupsafe import Markup, escape
from openai import OpenAI

from llm_providers import get_available_providers, get_provider

load_dotenv()
app = Flask(__name__)
app.secret_key = os.getenv("STORYTIME_SECRET_KEY") or os.urandom(32)
app.config.update(
    MAX_CONTENT_LENGTH=8 * 1024 * 1024,
    HOMEFRAME_URL=os.getenv("HOMEFRAME_URL", "https://homeframe.theplague.obsoletenerd.com/0.1").rstrip("/"),
    DATA_DIR=os.getenv("STORYTIME_DATA_DIR", str(Path(__file__).parent / "data")),
    CONTEXT_CHARS=30000,
)
SCHEMA = """
CREATE TABLE IF NOT EXISTS books (
 id TEXT PRIMARY KEY, title TEXT NOT NULL, format TEXT NOT NULL CHECK(format IN ('ongoing','standalone')),
 characters TEXT NOT NULL, themes TEXT NOT NULL, premise TEXT NOT NULL, reader_notes TEXT NOT NULL,
 preferred_provider TEXT NOT NULL, cover_path TEXT, cover_source TEXT, continuity_summary TEXT NOT NULL DEFAULT '',
 created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP, updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS chapters (
 id TEXT PRIMARY KEY, book_id TEXT NOT NULL REFERENCES books(id) ON DELETE CASCADE,
 number INTEGER NOT NULL, title TEXT NOT NULL, content TEXT NOT NULL,
 guidance TEXT NOT NULL DEFAULT '', provider TEXT NOT NULL, summary TEXT NOT NULL DEFAULT '',
 created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP, UNIQUE(book_id,number)
);
CREATE INDEX IF NOT EXISTS chapter_order ON chapters(book_id,number);
"""


def data_dir():
    path = Path(app.config["DATA_DIR"])
    path.mkdir(parents=True, exist_ok=True)
    return path


@contextmanager
def db():
    connection = sqlite3.connect(data_dir() / "storytime-v2.db", timeout=15)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys=ON")
    connection.executescript(SCHEMA)
    # Existing V2 installations have chapters without the memory field.
    columns = {row[1] for row in connection.execute("PRAGMA table_info(chapters)")}
    if "summary" not in columns:
        connection.execute("ALTER TABLE chapters ADD COLUMN summary TEXT NOT NULL DEFAULT ''")
    try:
        with connection:
            yield connection
    finally:
        connection.close()


def book_or_404(connection, book_id):
    book = connection.execute("SELECT * FROM books WHERE id=?", (book_id,)).fetchone()
    if book is None:
        abort(404)
    return book


def chapters_for(connection, book_id):
    return connection.execute("SELECT * FROM chapters WHERE book_id=? ORDER BY number", (book_id,)).fetchall()


def generate_text(provider, prompt):
    result = get_provider(provider).generate_text(prompt)
    if not result or result.lstrip().startswith("[Error"):
        raise RuntimeError(result or "The storyteller returned no text.")
    return result.strip()


def first_prompt(book):
    instruction = ("Write Chapter 1 of an ongoing bedtime book. Establish the world and characters, "
                   "give this chapter a satisfying small adventure, and leave room for later chapters."
                   if book["format"] == "ongoing" else
                   "Write a complete standalone bedtime story with a beginning, middle, and satisfying ending.")
    return (f"{instruction} Aim for 600-1000 words. Be gentle, vivid, and age appropriate. "
            "Return only prose, without a heading.\n\n"
            f"Title: {book['title']}\nCharacters: {book['characters']}\nThemes: {book['themes']}\n"
            f"Premise: {book['premise']}\nReader notes: {book['reader_notes']}")


def next_prompt(book, chapters, guidance):
    latest = chapters[-1]
    # Summaries preserve every earlier chapter, even when full text no longer fits.
    earlier = "\n\n".join(
        f"Chapter {c['number']}: " + (c["summary"] or
        "Summary unavailable; excerpt: " + c["content"][:1200])[:1200]
        for c in chapters[:-1]
    )
    # A user may call back to an older detail. Include one matching chapter in full.
    words = {word.lower() for word in re.findall(r"\b\w{4,}\b", guidance)}
    candidates = [(sum(word in c["content"].lower() for word in words), c)
                  for c in chapters[:-1]]
    relevant = max(candidates, key=lambda item: item[0], default=(0, None))
    recalled = (f"\nRelevant earlier chapter {relevant[1]['number']} in full: " +
                relevant[1]["content"][:6000]) if relevant[0] and relevant[1] else ""
    memory = (f"Established facts and open threads: {book['continuity_summary'][:4000]}\n\n"
              f"Chapter-by-chapter history: {earlier}{recalled}")
    memory = memory[:app.config["CONTEXT_CHARS"]]
    return ("Continue this children's bedtime book. Preserve personalities, events, and world rules. "
            "Write a gentle 600-1000 word chapter with a small arc and room for more. "
            "Follow the reader's direction where possible. Return only prose, no heading.\n\n"
            f"Book: {book['title']}\nCharacters: {book['characters']}\nThemes: {book['themes']}\n"
            f"Premise: {book['premise']}\nReader notes: {book['reader_notes']}\n"
            f"{memory}\n\nLatest chapter ({latest['number']}): {latest['content']}\n\n"
            f"Direction for Chapter {latest['number'] + 1}: {guidance or 'Continue naturally.'}")


def update_memory(book_id, provider, chapter_number=None):
    """Save a chapter synopsis and refresh the book's lasting facts."""
    with db() as connection:
        book = book_or_404(connection, book_id)
        chapters = chapters_for(connection, book_id)
        chapter = next((c for c in chapters if c["number"] == chapter_number), None) if chapter_number else chapters[-1]
        if chapter is None:
            raise ValueError("Chapter not found")
    try:
        summary = generate_text(provider,
            "Summarize this children's book chapter in at most 150 words. Preserve plot events, "
            "character decisions, new facts, locations, and unresolved threads. Do not invent details.\n\n"
            + chapter["content"])[:1200]
        with db() as connection:
            connection.execute("UPDATE chapters SET summary=? WHERE id=?", (summary, chapter["id"]))
    except Exception:
        app.logger.exception("Could not summarize chapter %s", chapter["number"])
        return
    try:
        bible = generate_text(provider,
            "Maintain a concise continuity bible for this children's book. Update the existing notes "
            "using the new chapter summary. Preserve established characters, relationships, world rules, "
            "locations, important objects, and unresolved threads. Remove resolved threads; do not invent "
            "facts. Use short labelled sections and stay under 500 words.\n\n"
            f"Existing notes:\n{book['continuity_summary'] or '(none yet)'}\n\n"
            f"Chapter {chapter['number']} summary:\n{summary}")[:4000]
        with db() as connection:
            connection.execute("UPDATE books SET continuity_summary=? WHERE id=?", (bible, book_id))
    except Exception:
        app.logger.exception("Could not update continuity bible for %s", book_id)


def upload_cover(file, book_id):
    if not file or not file.filename:
        raise ValueError("Choose an image to upload.")
    raw = file.read()
    if raw.startswith(b"\x89PNG\r\n\x1a\n"):
        extension = "png"
    elif raw.startswith(b"\xff\xd8\xff") and raw.endswith(b"\xff\xd9"):
        extension = "jpg"
    elif raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
        extension = "webp"
    else:
        raise ValueError("Upload a PNG, JPEG, or WebP image.")
    name = f"{book_id}-{uuid.uuid4().hex}.{extension}"
    covers = data_dir() / "covers"
    covers.mkdir(exist_ok=True)
    (covers / name).write_bytes(raw)
    return name


def generate_cover(book):
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("Cover generation requires an OpenAI API key. You can upload an image instead.")
    prompt = ("Portrait children's book cover illustration, no words or letters, gentle magical bedtime mood. "
              f"Title: {book['title']}. Characters: {book['characters']}. Themes: {book['themes']}. "
              f"Premise: {book['premise']}.")
    image = OpenAI(api_key=os.getenv("OPENAI_API_KEY")).images.generate(
        model=os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-2.5-flare"),
        prompt=prompt, size="1024x1536", quality="medium", n=1).data[0]
    if image.b64_json:
        raw = base64.b64decode(image.b64_json)
    elif image.url and image.url.startswith("https://"):
        response = requests.get(image.url, timeout=45)
        response.raise_for_status()
        raw = response.content
    else:
        raise ValueError("The image service returned no image.")
    if not raw.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError("The image service returned an invalid image.")
    name = f"{book['id']}-{uuid.uuid4().hex}.png"
    covers = data_dir() / "covers"
    covers.mkdir(exist_ok=True)
    (covers / name).write_bytes(raw)
    return name


@app.template_filter("story")
def render_story(content):
    blocks = re.split(r"\n\s*\n", content.strip())
    return Markup("\n".join(f"<p>{escape(block).replace(chr(10), Markup('<br>'))}</p>"
                            for block in blocks if block.strip()))


@app.route("/")
def index():
    with db() as connection:
        books = connection.execute("SELECT books.*, COUNT(chapters.id) AS chapter_count FROM books "
                                   "LEFT JOIN chapters ON chapters.book_id=books.id GROUP BY books.id "
                                   "ORDER BY books.updated_at DESC, books.created_at DESC").fetchall()
    return render_template("index.html", books=books)


@app.route("/books/new")
def new_book():
    return render_template("new_book.html", providers=get_available_providers())


@app.route("/books", methods=["POST"])
def create_book():
    details = {key: request.form.get(key, "").strip() for key in
               ("title", "format", "characters", "themes", "premise", "reader_notes", "provider", "cover_choice")}
    if details["format"] not in ("ongoing", "standalone") or not details["characters"] or not details["premise"]:
        abort(400)
    if details["cover_choice"] not in ("none", "upload", "generate"):
        abort(400)
    book_id = uuid.uuid4().hex
    try:
        content = generate_text(details["provider"], first_prompt(details))
        if not details["title"]:
            details["title"] = generate_text(details["provider"],
                "Give this children's story a short title. Reply with only the title.\n\n" + content[:5000]).strip(' "\'')[:100]
        cover = upload_cover(request.files.get("cover"), book_id) if details["cover_choice"] == "upload" else None
    except Exception as error:
        app.logger.exception("Book creation failed")
        flash(f"Could not create the book: {error}")
        return redirect(url_for("new_book"))
    with db() as connection:
        connection.execute("INSERT INTO books (id,title,format,characters,themes,premise,reader_notes,preferred_provider,cover_path,cover_source) "
                           "VALUES (?,?,?,?,?,?,?,?,?,?)",
                           (book_id, details["title"], details["format"], details["characters"], details["themes"],
                            details["premise"], details["reader_notes"], details["provider"], cover,
                            "uploaded" if cover else None))
        connection.execute("INSERT INTO chapters (id,book_id,number,title,content,provider) VALUES (?,?,?,?,?,?)",
                           (uuid.uuid4().hex, book_id, 1, "Chapter 1" if details["format"] == "ongoing" else "The Story",
                            content, details["provider"]))
    if details["format"] == "ongoing":
        update_memory(book_id, details["provider"])
    if details["cover_choice"] == "generate":
        try:
            with db() as connection:
                cover = generate_cover(book_or_404(connection, book_id))
                connection.execute("UPDATE books SET cover_path=?,cover_source='generated' WHERE id=?", (cover, book_id))
        except Exception:
            app.logger.exception("Cover generation failed")
            flash("Your book is ready, but its cover could not be generated. You can add one later.")
    return redirect(url_for("read_book", book_id=book_id))


@app.route("/books/<book_id>")
def book_detail(book_id):
    with db() as connection:
        book = book_or_404(connection, book_id)
        chapters = chapters_for(connection, book_id)
    return render_template("book.html", book=book, chapters=chapters)


@app.route("/books/<book_id>/read")
def read_book(book_id):
    with db() as connection:
        book = book_or_404(connection, book_id)
        chapters = chapters_for(connection, book_id)
    return render_template("reader.html", book=book, chapters=chapters)


@app.route("/books/<book_id>/write")
def write_book(book_id):
    with db() as connection:
        book = book_or_404(connection, book_id)
        chapters = chapters_for(connection, book_id)
    if book["format"] != "ongoing":
        abort(404)
    return render_template("write.html", book=book, latest=chapters[-1],
                           next_number=chapters[-1]["number"] + 1, providers=get_available_providers())


@app.route("/books/<book_id>/chapters", methods=["POST"])
def create_chapter(book_id):
    with db() as connection:
        book = book_or_404(connection, book_id)
        chapters = chapters_for(connection, book_id)
    if book["format"] != "ongoing":
        abort(404)
    guidance = request.form.get("guidance", "").strip()[:4000]
    provider = request.form.get("provider", book["preferred_provider"])
    # Books created by early V2 builds have no stored chapter summaries yet.
    for chapter in chapters:
        if not chapter["summary"]:
            update_memory(book_id, provider, chapter["number"])
    with db() as connection:
        book = book_or_404(connection, book_id)
        chapters = chapters_for(connection, book_id)
    try:
        content = generate_text(provider, next_prompt(book, chapters, guidance))
    except Exception as error:
        app.logger.exception("Chapter generation failed")
        flash(f"Could not write the next chapter: {error}")
        return redirect(url_for("write_book", book_id=book_id))
    number = chapters[-1]["number"] + 1
    try:
        with db() as connection:
            connection.execute("INSERT INTO chapters (id,book_id,number,title,content,guidance,provider) VALUES (?,?,?,?,?,?,?)",
                               (uuid.uuid4().hex, book_id, number, f"Chapter {number}", content, guidance, provider))
            connection.execute("UPDATE books SET updated_at=CURRENT_TIMESTAMP WHERE id=?", (book_id,))
    except sqlite3.IntegrityError:
        flash("Another chapter was added while this one was being written. Please review the book and try again.")
        return redirect(url_for("book_detail", book_id=book_id))
    update_memory(book_id, provider)
    return redirect(url_for("read_book", book_id=book_id, _anchor=f"chapter-{number}"))


@app.route("/books/<book_id>/cover", methods=["POST"])
def replace_cover(book_id):
    with db() as connection:
        book = book_or_404(connection, book_id)
    choice = request.form.get("cover_choice")
    try:
        if choice == "upload":
            cover = upload_cover(request.files.get("cover"), book_id)
        elif choice == "generate":
            cover = generate_cover(book)
        else:
            abort(400)
        with db() as connection:
            connection.execute("UPDATE books SET cover_path=?,cover_source=?,updated_at=CURRENT_TIMESTAMP WHERE id=?",
                               (cover, "uploaded" if choice == "upload" else "generated", book_id))
        flash("Cover updated.")
    except Exception as error:
        app.logger.exception("Cover update failed")
        flash(f"Could not update the cover: {error}")
    return redirect(url_for("book_detail", book_id=book_id))


@app.route("/covers/<filename>")
def cover_file(filename):
    return send_from_directory(data_dir() / "covers", filename)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=1337, debug=True)
