import os
import uuid
import json
import requests
from datetime import datetime
import re
from dotenv import load_dotenv

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    session,
    jsonify,
)
from openai import OpenAI
from llm_providers import get_available_providers, get_provider

app = Flask(__name__)
app.secret_key = os.urandom(24)  # <-- change for production

# Session storage directory
SESSIONS_DIR = "sessions"
if not os.path.exists(SESSIONS_DIR):
    os.makedirs(SESSIONS_DIR)

# Load .env file
load_dotenv()

# Homeframe is served as a versioned shared asset, keeping this app's templates
# free to reference the configured deployment location.
app.config["HOMEFRAME_URL"] = os.environ.get(
    "HOMEFRAME_URL",
    "https://homeframe.theplague.obsoletenerd.com/0.1",
).rstrip("/")

# OpenAI setup for image generation
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Master prompts
MASTER_PROMPT = """
You are a masterful storyteller creating an engaging bedtime story for a child.
If the age of the child is mentioned in the input, adapt the story to suit.
If no age is mentioned, assume the child is around 8 years old.

STORY REQUIREMENTS:
- Write a complete story that's 8-12 paragraphs long (aim for about 800-1200 words)
- The story should be gentle, positive, funny, engaging, and adventurous
- Include a clear beginning, middle with multiple events/challenges, and satisfying ending

STORY STRUCTURE - Choose ONE of these varied formats:
1. Quest Adventure: Character goes on a journey with 3-4 different challenges/discoveries
2. Mystery Story: Character discovers something puzzling and solves it through multiple clues
3. Friendship Tale: Character meets new friends and they have several adventures together
4. Magic Discovery: Character finds something magical that leads to multiple magical experiences
5. Problem-Solving Story: Character faces a problem that requires several creative attempts to solve
6. Exploration Adventure: Character explores a new place and has multiple discoveries/encounters

VARIETY ELEMENTS - Include 2-3 of these to make each story unique:
- Talking animals with distinct personalities
- A magical object or ability
- A helpful mentor figure
- A silly misunderstanding that gets resolved
- A creative invention or solution
- A celebration or festival
- Weather that affects the adventure
- A map, riddle, or puzzle to solve
- An unexpected ally
- A cozy hideout or special place

PACING:
- Spend 2-3 paragraphs setting up the character and situation
- Include 4-6 paragraphs of main adventure with multiple events
- Use 2-3 paragraphs for a satisfying conclusion that ties everything together
- Add sensory details (what characters see, hear, smell, feel) to make scenes vivid
- Include dialogue to bring characters to life

Here is what the user gave you to try and use in the story.
Don't feel like you need to use all of it, but weave in what you can naturally.

"""

CHAPTER_PROMPT = """
You are a masterful storyteller continuing an engaging bedtime story for a child.
You will be given an existing story and need to create the next chapter that continues the adventure.

CHAPTER REQUIREMENTS:
- Write a new chapter that's 6-10 paragraphs long (aim for about 600-1000 words)
- The chapter should feel like a natural continuation of the existing story
- Maintain the same tone, characters, and world established in the original story
- The chapter should be gentle, positive, engaging, and adventurous
- Include new events, discoveries, or challenges that build on what came before
- End with either resolution or a gentle cliffhanger that could lead to another chapter

CONTINUITY GUIDELINES:
- Keep the same main characters and their personalities
- Reference events from the previous story/chapters
- Maintain the same magical or realistic world rules established
- Keep the same age-appropriate tone and complexity
- Build on relationships and locations already introduced

CHAPTER STRUCTURE:
- Start by briefly connecting to where the previous story left off
- Introduce 2-3 new events, challenges, or discoveries
- Show character growth or new aspects of familiar characters
- Include sensory details and dialogue to bring scenes to life
- End with a satisfying conclusion or gentle transition to potential future adventures

Here is the existing story that you need to continue:

"""


IMAGE_SUMMARY_PROMPT = """
Based on the following story, create a brief description for an image that would be perfect to accompany this story.
The image should be child-friendly and capture the main scene or feeling of the story.
Keep the description under 200 characters and focus on visual elements that would delight a 6-12 year old (not toddler).
Do not include any text or words in the image.
Do not include any children in the picture, but instead focus on characters and elements of the story so they can imagine themselves in it.

Story:
"""

TITLE_EXTRACTION_PROMPT = """
Based on the following bedtime story, create a short, catchy title that captures the essence of the story.
The title should be:
- 3-8 words long
- Child-friendly and engaging
- Capture the main adventure or theme
- Suitable for a bedtime story

Only respond with the title, nothing else.

Story:
"""

# Helper functions
def generate_story(provider_name: str, prompt: str) -> str:
    """
    Generate a story using the specified LLM provider.
    """
    try:
        provider = get_provider(provider_name)
        return provider.generate_text(prompt)
    except Exception as e:
        return f"[Error with {provider_name}: {e}]"

def generate_image_prompt(provider_name: str, story: str) -> str:
    """
    Uses the selected LLM provider to create a concise image description based on the story.
    """
    full_prompt = IMAGE_SUMMARY_PROMPT + story
    return generate_story(provider_name, full_prompt).strip()

def extract_story_title(provider_name: str, story: str) -> str:
    """
    Uses the selected LLM provider to extract a concise title from the story.
    """
    full_prompt = TITLE_EXTRACTION_PROMPT + story
    title = generate_story(provider_name, full_prompt).strip()

    # Clean the title to make it filename-safe
    # Remove quotes if the AI wrapped the title in them
    title = title.strip('"\'')
    # Replace problematic characters with safe ones
    title = re.sub(r'[<>:"/\\|?*]', '', title)
    # Replace spaces with underscores and limit length
    title = title.replace(' ', '_')[:80]

    return title if title else "Untitled_Story"

def save_story_to_file(story: str, provider_name: str, existing_filename: str = "") -> str:
    """
    Saves the story to a markdown file with date and AI-generated title.
    If existing_filename is provided, updates that file instead of creating a new one.
    Returns the filename of the saved file.
    """
    # Create stories directory if it doesn't exist
    stories_dir = "stories"
    if not os.path.exists(stories_dir):
        os.makedirs(stories_dir)

    if existing_filename and existing_filename.strip():
        # Use existing filename for updates
        filename = existing_filename
        filepath = os.path.join(stories_dir, filename)

        # Extract title from existing file if it exists
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    existing_content = f.read()
                    lines = existing_content.split('\n')
                    title = "Untitled Story"
                    for line in lines:
                        if line.startswith('# '):
                            title = line[2:].strip()
                            break
            except Exception:
                title = "Updated Story"
        else:
            title = "Updated Story"
    else:
        # Create new filename
        date_str = datetime.now().strftime("%Y-%m-%d")

        # Check if this is a multi-chapter story
        is_multi_chapter = "## Chapter 2" in story or "---" in story

        # Extract title from story (use first part if it's multi-chapter)
        story_for_title = story.split("---")[0] if is_multi_chapter else story
        title = extract_story_title(provider_name, story_for_title)

        # Add chapter indicator to title if multi-chapter
        if is_multi_chapter:
            chapter_count = story.count("## Chapter") + 1  # +1 for the first chapter
            title += f"_Chapters_1-{chapter_count}"

        # Create filename
        filename = f"{date_str}_{title}.md"
        filepath = os.path.join(stories_dir, filename)

    # Create markdown content
    clean_title = title.replace('_', ' ').replace('Chapters 1-', 'Chapters 1-')
    markdown_content = f"# {clean_title}\n\n"
    markdown_content += f"*Generated on {datetime.now().strftime('%Y-%m-%d at %H:%M:%S')}*\n"
    markdown_content += f"*Using provider: {provider_name}*\n\n"
    markdown_content += "---\n\n"
    markdown_content += story

    # Save to file
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        print(f"Story saved to: {filepath}")
        return filename
    except Exception as e:
        print(f"Error saving story: {e}")
        return ""

def get_available_stories() -> dict:
    """
    Scans the stories directory and returns a dictionary of available stories.
    Returns dict with filename as key and display info as value.
    """
    stories_dir = "stories"
    available_stories = {}

    if not os.path.exists(stories_dir):
        return available_stories

    try:
        for filename in os.listdir(stories_dir):
            if filename.endswith('.md'):
                filepath = os.path.join(stories_dir, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()
                        # Extract title from markdown (first # line)
                        lines = content.split('\n')
                        title = "Untitled Story"
                        for line in lines:
                            if line.startswith('# '):
                                title = line[2:].strip()
                                break

                        # Get creation date from filename or file stats
                        date_match = filename[:10] if filename[:10].count('-') == 2 else None
                        if date_match:
                            display_name = f"{title} ({date_match})"
                        else:
                            mod_time = os.path.getmtime(filepath)
                            date_str = datetime.fromtimestamp(mod_time).strftime("%Y-%m-%d")
                            display_name = f"{title} ({date_str})"

                        available_stories[filename] = {
                            'display_name': display_name,
                            'title': title,
                            'filepath': filepath
                        }
                except Exception as e:
                    print(f"Error reading story file {filename}: {e}")
                    continue
    except Exception as e:
        print(f"Error scanning stories directory: {e}")

    return available_stories

def load_story_from_file(filename: str) -> tuple[str, str]:
    """
    Loads a story from a markdown file and returns (title, story_content).
    """
    stories_dir = "stories"
    filepath = os.path.join(stories_dir, filename)

    if not os.path.exists(filepath):
        return "Story Not Found", "The requested story could not be found."

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        # Split content to separate metadata from story
        lines = content.split('\n')
        title = "Untitled Story"
        story_start_idx = 0

        # Find title and story start
        in_metadata = True
        for i, line in enumerate(lines):
            if line.startswith('# '):
                title = line[2:].strip()
            elif line.strip() == '---' and in_metadata:
                # This is the metadata separator, story starts after this
                story_start_idx = i + 1
                in_metadata = False
                break

        # Extract story content (everything after the metadata separator)
        # This includes all chapters and any additional separators
        story_lines = lines[story_start_idx:]
        story_content = '\n'.join(story_lines).strip()

        return title, story_content

    except Exception as e:
        print(f"Error loading story from {filename}: {e}")
        return "Error", f"Could not load the story: {e}"

def get_session_file_path(session_id: str) -> str:
    """Get the file path for a session."""
    return os.path.join(SESSIONS_DIR, f"{session_id}.json")

def save_session_data(session_id: str, data: dict):
    """Save session data to a file."""
    try:
        filepath = get_session_file_path(session_id)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Error saving session data: {e}")

def load_session_data(session_id: str) -> dict:
    """Load session data from a file."""
    try:
        filepath = get_session_file_path(session_id)
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        print(f"Error loading session data: {e}")
    return {}

def get_or_create_session_id() -> str:
    """Get existing session ID or create a new one."""
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    return session['session_id']

def get_session_value(key: str, default=None):
    """Get a value from file-based session storage."""
    session_id = get_or_create_session_id()
    data = load_session_data(session_id)
    return data.get(key, default)

def set_session_value(key: str, value):
    """Set a value in file-based session storage."""
    session_id = get_or_create_session_id()
    data = load_session_data(session_id)
    data[key] = value
    save_session_data(session_id, data)

def convert_basic_markdown_to_html(text: str) -> str:
    """
    Converts basic markdown elements to HTML without external dependencies.
    """
    html = text

    # Convert headers
    html = re.sub(r'^## (.*?)$', r'<h2>\1</h2>', html, flags=re.MULTILINE)
    html = re.sub(r'^# (.*?)$', r'<h1>\1</h1>', html, flags=re.MULTILINE)

    # Convert images with proper Flask static URLs
    def replace_image(match):
        alt_text = match.group(1)
        image_path = match.group(2)
        # Convert static/images/filename.png to proper Flask URL
        if image_path.startswith('static/images/'):
            filename = image_path.replace('static/images/', '')
            flask_url = url_for('static', filename=f'images/{filename}')
            return f'<img src="{flask_url}" alt="{alt_text}" />'
        else:
            return f'<img src="{image_path}" alt="{alt_text}" />'

    html = re.sub(r'!\[(.*?)\]\((.*?)\)', replace_image, html)

    # Convert horizontal rules
    html = re.sub(r'^---$', r'<hr>', html, flags=re.MULTILINE)

    # Convert paragraphs (split by double newlines)
    paragraphs = html.split('\n\n')
    formatted_paragraphs = []

    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if paragraph:
            # Don't wrap headers or images or hrs in paragraph tags
            if not (paragraph.startswith('<h') or paragraph.startswith('<img') or paragraph.startswith('<hr')):
                # Convert single newlines to spaces within paragraphs
                paragraph = paragraph.replace('\n', ' ')
                paragraph = f'<p>{paragraph}</p>'
            formatted_paragraphs.append(paragraph)

    return '\n'.join(formatted_paragraphs)

def cleanup_old_sessions():
    """Remove session files older than 24 hours."""
    try:
        current_time = datetime.now().timestamp()
        for filename in os.listdir(SESSIONS_DIR):
            if filename.endswith('.json'):
                filepath = os.path.join(SESSIONS_DIR, filename)
                file_time = os.path.getmtime(filepath)
                # Remove files older than 24 hours (86400 seconds)
                if current_time - file_time > 86400:
                    os.remove(filepath)
                    print(f"Removed old session file: {filename}")
    except Exception as e:
        print(f"Error cleaning up old sessions: {e}")

def download_and_save_image(image_url: str, story_filename: str, chapter_num: int = 1) -> str:
    """
    Downloads an image from URL and saves it locally.
    Returns the local filename or empty string if failed.
    """
    if not image_url:
        return ""

    try:
        # Create images directory if it doesn't exist
        images_dir = os.path.join("static", "images")
        if not os.path.exists(images_dir):
            os.makedirs(images_dir)

        # Generate local filename based on story filename and chapter
        base_name = story_filename.replace('.md', '')
        if chapter_num > 1:
            image_filename = f"{base_name}_chapter_{chapter_num}.png"
        else:
            image_filename = f"{base_name}.png"

        image_path = os.path.join(images_dir, image_filename)

        # Download the image
        response = requests.get(image_url, timeout=30)
        response.raise_for_status()

        # Save the image
        with open(image_path, 'wb') as f:
            f.write(response.content)

        print(f"Image saved to: {image_path}")
        return image_filename

    except Exception as e:
        print(f"Error downloading/saving image: {e}")
        return ""

def embed_image_in_story(story_content: str, image_filename: str) -> str:
    """
    Embeds an image at the end of each chapter in the story content.
    """
    if not image_filename:
        return story_content

    # Split story into chapters
    chapters = story_content.split("---")

    if len(chapters) == 1:
        # Single chapter story - add image at the end
        image_markdown = f"\n\n![Story Illustration](static/images/{image_filename})"
        return story_content + image_markdown

    else:
        # Multi-chapter story - add image to each chapter
        updated_chapters = []

        for i, chapter in enumerate(chapters):
            chapter = chapter.strip()
            if chapter:
                # Determine chapter number for image filename
                if i == 0:
                    # First chapter
                    chapter_image = image_filename
                else:
                    # Subsequent chapters
                    base_name = image_filename.replace('.png', '')
                    chapter_image = f"{base_name}_chapter_{i + 1}.png"

                image_markdown = f"\n\n![Chapter {i + 1} Illustration](static/images/{chapter_image})"
                updated_chapters.append(chapter + image_markdown)
            else:
                updated_chapters.append(chapter)

        return "\n\n---\n\n".join(updated_chapters)

def generate_image_with_openai(prompt: str) -> str:
    """
    Generates an image using OpenAI's DALL-E API and returns the image URL.
    """
    try:
        response = client.images.generate(
            model="dall-e-3",
            prompt=f"Children's book illustration style: {prompt}. Soft, warm colors, gentle and magical atmosphere, perfect for bedtime.",
            size="1024x1024",
            quality="standard",
            n=1,
        )
        if response and response.data and len(response.data) > 0 and response.data[0].url:
            return response.data[0].url
        else:
            return ""
    except Exception as e:
        print(f"Error generating image: {e}")
        return ""

# Routes
@app.route("/", methods=["GET"])
def index():
    # Clean up old session files periodically
    cleanup_old_sessions()

    available_providers = get_available_providers()
    available_stories = get_available_stories()
    print(available_providers)
    return render_template("index.html", available_providers=available_providers, available_stories=available_stories)


@app.route("/start", methods=["POST"])
def start():
    """
    Stores the user data in the session and redirects to the wait page.
    """
    set_session_value("name_and_friends", request.form.get("names", ""))
    set_session_value("favourite_things", request.form.get("things", ""))
    set_session_value("topic", request.form.get("topic", ""))
    set_session_value("llm_provider", request.form.get("llm_provider", "ollama"))
    # Store whether user wants image generation (checkbox returns "on" if checked, None if not)
    set_session_value("generate_image", request.form.get("generate_image") == "on")
    return redirect(url_for("wait"))

@app.route("/wait")
def wait():
    """
    Shows the "Please wait…" page. The page triggers the story generation.
    """
    return render_template("wait.html")

@app.route("/generate", methods=["POST"])
def generate():
    """
    Generates the story using the selected LLM provider and stores it in the session.
    Also saves the story to a local file.
    """
    user_prompt = (
        f"Names: {get_session_value('name_and_friends', '')}\n"
        f"Favourite things: {get_session_value('favourite_things', '')}\n"
        f"Topic: {get_session_value('topic', '')}\n"
    )
    full_prompt = MASTER_PROMPT + user_prompt

    # Generate the story using the selected provider
    provider_name = get_session_value("llm_provider", "ollama") or "ollama"
    story = generate_story(provider_name, full_prompt)
    set_session_value("story", story)

    # Extract and store the title
    title = extract_story_title(provider_name, story)
    set_session_value("story_title", title.replace('_', ' '))
    set_session_value("is_loaded_story", False)

    # Save the story to a file
    saved_filename = save_story_to_file(story, provider_name)
    if saved_filename:
        set_session_value("saved_filename", saved_filename)

    return jsonify({"status": "ok"})

@app.route("/result")
def result():
    """
    Display the generated story with optional image generation capability.
    """
    story = get_session_value("story", "No story generated.")
    title = get_session_value("story_title", "")
    should_generate_image = get_session_value("generate_image", False)
    is_loaded_story = get_session_value("is_loaded_story", False)

    # Convert basic markdown to HTML for proper display
    story_text = story or "No story generated."
    story_html = convert_basic_markdown_to_html(story_text)

    return render_template("result.html", story=story_html, title=title, generate_image=should_generate_image, is_loaded_story=is_loaded_story)

@app.route("/generate_image", methods=["POST"])
def generate_image():
    """
    Generates an image based on the story and returns the image URL.
    """
    story = get_session_value("story", "")
    if not story:
        return jsonify({"error": "No story found"}), 400

    # Generate image description using the same LLM provider that created the story
    provider_name = get_session_value("llm_provider", "ollama") or "ollama"
    image_description = generate_image_prompt(provider_name, story)

    # Generate the actual image
    image_url = generate_image_with_openai(image_description)

    if image_url:
        set_session_value("image_url", image_url)

        # Download and save the image
        saved_filename = get_session_value("saved_filename", "")
        if saved_filename:
            # Determine current chapter count for image naming
            current_story = get_session_value("story", "") or ""
            chapter_count = current_story.count("## Chapter") + 1

            image_filename = download_and_save_image(image_url, saved_filename, chapter_count)

            if image_filename:
                # Embed image in story content and update both session and file
                updated_story = embed_image_in_story(current_story, image_filename)
                set_session_value("story", updated_story)

                # Re-save the story file with embedded image
                provider_name = get_session_value("llm_provider", "ollama") or "ollama"
                save_story_to_file(updated_story, provider_name, saved_filename)

        return jsonify({"status": "success", "image_url": image_url})
    else:
        return jsonify({"error": "Failed to generate image"}), 500

@app.route("/check_image")
def check_image():
    """
    Check if an image has been generated for the current story.
    """
    image_url = get_session_value("image_url")
    if image_url:
        return jsonify({"status": "ready", "image_url": image_url})
    else:
        return jsonify({"status": "generating"})

@app.route("/generate_chapter", methods=["POST"])
def generate_chapter():
    """
    Generates a new chapter based on the existing story and appends it.
    """
    current_story = get_session_value("story", "")
    if not current_story:
        return jsonify({"error": "No existing story found"}), 400

    # Create the chapter prompt with the existing story
    full_prompt = CHAPTER_PROMPT + current_story

    # Generate the new chapter using the same provider
    provider_name = get_session_value("llm_provider", "ollama") or "ollama"
    new_chapter = generate_story(provider_name, full_prompt)

    # Determine the next chapter number
    existing_chapters = current_story.count("## Chapter")
    next_chapter_num = existing_chapters + 2  # +2 because first story is Chapter 1, then we add 1 more

    # Combine the original story with the new chapter
    combined_story = current_story + f"\n\n---\n\n## Chapter {next_chapter_num}\n\n" + new_chapter
    set_session_value("story", combined_story)

    # Generate image for the new chapter if image generation was originally enabled
    should_generate_image = get_session_value("generate_image", False)
    if should_generate_image:
        # Generate image description for the new chapter
        image_description = generate_image_prompt(provider_name, new_chapter)
        image_url = generate_image_with_openai(image_description)

        if image_url:
            # Download and save the image for this specific chapter
            saved_filename = get_session_value("saved_filename", "")
            if saved_filename:
                image_filename = download_and_save_image(image_url, saved_filename, next_chapter_num)

                if image_filename:
                    # Embed the image at the end of just the new chapter
                    image_markdown = f"\n\n![Chapter {next_chapter_num} Illustration](static/images/{image_filename})"
                    combined_story += image_markdown
                    set_session_value("story", combined_story)

    # Save the updated story to a file
    existing_filename = get_session_value("saved_filename", "") or ""
    saved_filename = save_story_to_file(combined_story, provider_name, existing_filename)
    if saved_filename:
        set_session_value("saved_filename", saved_filename)

    return jsonify({"status": "success", "redirect": url_for("result")})

@app.route("/read", methods=["POST"])
def read():
    """
    Loads an existing story from file and displays it in the result template.
    """
    selected_story = request.form.get("story_selector")
    if not selected_story:
        return redirect(url_for("index"))

    # Load the story from file
    title, story_content = load_story_from_file(selected_story)

    # Store in session so it can be continued with new chapters
    set_session_value("story", story_content)
    set_session_value("story_title", title)
    set_session_value("is_loaded_story", True)
    # Set a default provider for chapter generation if needed
    set_session_value("llm_provider", get_session_value("llm_provider", "ollama"))
    # Disable image generation for loaded stories by default
    set_session_value("generate_image", False)

    return redirect(url_for("result"))

# Run the app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=1337, debug=True)
