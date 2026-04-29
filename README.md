# PKM Automation System

Personal Knowledge Management pipeline: raw sources → structured Obsidian notes.

---

## Setup

### 1. Install Ollama
```bash
# macOS
brew install ollama

# Start the server
ollama serve

# Pull the recommended model (in a new terminal)
ollama pull qwen2.5:14b
```

### 2. Install Python dependencies
```bash
# Python 3.11+ required
pip install -r requirements.txt
```

### 3. Configure your vault path
Edit `config.yaml`:
```yaml
vault:
  path: "~/path/to/your/ObsidianVault"
```

### 4. (First run) Rebuild index from existing vault
```bash
python main.py sync
```
This scans your existing Concepts folder and builds the index files.

---

## Usage

```bash
# Process a plain text file
python main.py process --input "notes.txt" --type text --source "Article: Deep Work"

# Process a PDF
python main.py process --input "book.pdf" --type pdf --source "Book: Thinking Fast and Slow"

# Process a YouTube video
python main.py process --input "https://youtube.com/..." --type youtube --source "Talk: Andrew Huberman on Sleep"

# Provide a pre-made transcript
python main.py process --input "transcript.txt" --type text --source "Video: XYZ"

# Rebuild vault index from existing notes (run after manual Obsidian edits)
python main.py sync
```

---

## Output Structure (inside your Obsidian vault)
```
/Concepts/          ← atomic concept .md files
/Sources/           ← one summary note per processed source
/Review/            ← merge suggestions (review manually)
/_index/            ← vault_index.json, tag_index.json (system files)
```

---

## Workflow
1. Run `process` on a source
2. Open Obsidian — new notes appear in `/Concepts/`
3. Check `/Review/` for merge suggestions
4. Adjust tags or links directly in Obsidian as needed
5. Run `sync` after manual edits to keep indexes up to date
