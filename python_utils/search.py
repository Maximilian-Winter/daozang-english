import json

from daoist_db import DaoistDatabase

# Initialize database
db = DaoistDatabase("daoist_texts.db")

# Search with options
results = db.search(
    terms=["九天玄女", "西王母"],
    search_chinese=True,
    search_english=True,
    #category="正一部",
    match_all=True,  # Any term matches (use True for all terms)
    limit=10
)
for result in results:
    print(f"---\n\n\nTitle: {result["source_id"]}")
    print(f"Category: {result["category"]}\n\n")
    print(result["translated_text"])