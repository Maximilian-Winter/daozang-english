# Daoist Translation Database

A SQLAlchemy-based database system for storing and searching translated Daoist texts.
Supports searching in both Chinese (original) and English (translated) text.

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Load your JSON files into the database

```bash
# Load all JSON files from a folder (recursively)
python daoist_db.py load ./translated_json

# Non-recursive (only top-level folder)
python daoist_db.py load /path/to/folder --no-recursive
```

### 2. Search the database

```bash
# Search both Chinese and English text
python daoist_db.py search "tiger" "dragon"

# Search only Chinese text
python daoist_db.py search --chinese "道"

# Search only English text
python daoist_db.py search --english "immortal"

# Search with category filter
python daoist_db.py search --english --category "正一部" "Divine Tiger"

# Require all terms to match
python daoist_db.py search --match-all "tiger" "dragon"
```

### 3. Get a specific text

```bash
python daoist_db.py get "洞真太上神虎隐文"

# Export in different formats
python daoist_db.py get "洞真太上神虎隐文" --format json
python daoist_db.py get "洞真太上神虎隐文" --format txt
python daoist_db.py get "洞真太上神虎隐文" --format md
```

### 4. View statistics

```bash
python daoist_db.py stats
python daoist_db.py categories
```

## Interactive Mode

For a more exploratory experience, use the interactive search interface:

```bash
python interactive_search.py
```

Commands in interactive mode:
- `search <terms>` or `s <terms>` - Search both Chinese and English
- `chinese <terms>` or `c <terms>` - Search Chinese only
- `english <terms>` or `e <terms>` - Search English only
- `get <source_id>` or `g <id>` - Get full text
- `view <number>` or `v <n>` - View result from last search
- `stats` - Show database statistics
- `categories` or `cat` - List all categories
- `export <id> <format>` - Export text to file
- `help` or `?` - Show help
- `quit` or `q` - Exit

## Python API

```python
from daoist_db import DaoistDatabase

# Initialize database
db = DaoistDatabase("daoist_texts.db")

# Load files
db.load_folder("/path/to/translations")

# Search
results = db.search(["tiger", "dragon"])  # Both languages
results = db.search_chinese(["道", "德"])  # Chinese only
results = db.search_english(["immortal"])  # English only

# Search with options
results = db.search(
    terms=["tiger"],
    search_chinese=True,
    search_english=True,
    category="正一部",
    match_all=False,  # Any term matches (use True for all terms)
    limit=100
)

# Get full text
text = db.get_text("洞真太上神虎隐文")

# Get statistics
stats = db.get_stats()
categories = db.list_categories()

# Export
output = db.export_text("洞真太上神虎隐文", format="md")
```

## Database Schema

### DaoistText (main table)
- `source_id` - Unique identifier (Chinese title)
- `source_file` - Original JSON file path
- `category` - Category (e.g., "正统道藏正一部")
- `total_chunks` - Number of chunks
- `total_original_chars` - Character count in Chinese
- `total_translated_chars` - Character count in English
- `model_used` - Translation model
- `full_translation` - Complete English translation

### TextChunk (chunks table)
- `chunk_index` - Position in text
- `original_text` - Chinese text
- `translated_text` - English translation
- `start_char`, `end_char` - Position markers
- `translation_time` - Time taken to translate

## JSON Input Format

The expected JSON format matches your translation output:

```json
{
  "source_id": "洞真太上神虎隐文",
  "source_file": "chunked_daozang/洞真太上神虎隐文.json",
  "total_chunks": 2,
  "total_original_chars": 2304,
  "total_translated_chars": 15792,
  "model_used": "mistralai/mistral-large-2512",
  "metadata": {
    "source_file": "dao/正统道藏正一部/洞真太上神虎隐文.txt",
    "category": "正统道藏正一部"
  },
  "full_translation": "...",
  "chunks": [
    {
      "chunk_index": 0,
      "original_text": "...",
      "translated_text": "...",
      "start_char": 0,
      "end_char": 1295,
      "translation_time": 48.6
    }
  ]
}
```

## Tips

1. **Use quotes for multi-word searches**: `search "Divine Tiger"`
2. **Mix Chinese and English terms**: `search 道 "Great Dao"`
3. **Filter by category**: `--category 正一部`
4. **View full context**: Use `view <n>` after search to see complete text
5. **Export for reading**: `export <id> md` creates a readable Markdown file
