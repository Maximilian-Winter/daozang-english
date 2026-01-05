#!/usr/bin/env python3
"""
Interactive Search Interface for Daoist Translation Database
============================================================
A REPL-style interface for exploring the database.
"""

import readline  # For command history
import shlex
from daoist_db import DaoistDatabase


class InteractiveSearch:
    """Interactive search interface."""
    
    def __init__(self, db_path: str = "daoist_texts.db"):
        self.db = DaoistDatabase(db_path)
        self.last_results = []
        self.commands = {
            'help': self.cmd_help,
            '?': self.cmd_help,
            'search': self.cmd_search,
            's': self.cmd_search,
            'chinese': self.cmd_chinese,
            'c': self.cmd_chinese,
            'english': self.cmd_english,
            'e': self.cmd_english,
            'get': self.cmd_get,
            'g': self.cmd_get,
            'view': self.cmd_view,
            'v': self.cmd_view,
            'stats': self.cmd_stats,
            'categories': self.cmd_categories,
            'cat': self.cmd_categories,
            'export': self.cmd_export,
            'quit': self.cmd_quit,
            'exit': self.cmd_quit,
            'q': self.cmd_quit,
        }
    
    def cmd_help(self, args):
        """Show help message."""
        print("""
╔═══════════════════════════════════════════════════════════════════════╗
║            Daoist Translation Database - Interactive Search           ║
╠═══════════════════════════════════════════════════════════════════════╣
║ Commands:                                                             ║
║                                                                       ║
║  search <terms>   Search both Chinese and English (alias: s)          ║
║  chinese <terms>  Search only Chinese text (alias: c)                 ║
║  english <terms>  Search only English text (alias: e)                 ║
║                                                                       ║
║  get <source_id>  Get full text by source_id (alias: g)               ║
║  view <number>    View full result from last search (alias: v)        ║
║                                                                       ║
║  stats            Show database statistics                            ║
║  categories       List all categories (alias: cat)                    ║
║  export <id> <fmt> Export text (formats: json, txt, md)               ║
║                                                                       ║
║  help             Show this help (alias: ?)                           ║
║  quit             Exit (aliases: exit, q)                             ║
║                                                                       ║
║ Search options:                                                       ║
║  --category <cat>  Filter by category                                 ║
║  --all             Require all terms to match                         ║
║  --limit <n>       Limit results (default: 20)                        ║
║                                                                       ║
║ Examples:                                                             ║
║  > search 道 tiger                                                    ║
║  > chinese 神虎                                                       ║
║  > english "Divine Tiger" --category 正一部                           ║
║  > get 洞真太上神虎隐文                                               ║
╚═══════════════════════════════════════════════════════════════════════╝
        """)
    
    def parse_search_args(self, args):
        """Parse search arguments including options."""
        terms = []
        category = None
        match_all = False
        limit = 20
        
        i = 0
        while i < len(args):
            arg = args[i]
            if arg == '--category' and i + 1 < len(args):
                category = args[i + 1]
                i += 2
            elif arg == '--all':
                match_all = True
                i += 1
            elif arg == '--limit' and i + 1 < len(args):
                try:
                    limit = int(args[i + 1])
                except ValueError:
                    pass
                i += 2
            else:
                terms.append(arg)
                i += 1
        
        return terms, category, match_all, limit
    
    def display_results(self, results, show_full=False):
        """Display search results."""
        if not results:
            print("No results found.")
            return
        
        print(f"\n{'═' * 70}")
        print(f"Found {len(results)} results")
        print('═' * 70)
        
        for i, r in enumerate(results, 1):
            print(f"\n[{i}] {r['source_id']}")
            print(f"    Category: {r['category']}")
            print(f"    Matched: {', '.join(r['matched_terms'])}")
            
            if r['chunk_index'] is not None:
                print(f"    Chunk: {r['chunk_index']}")
            
            if show_full:
                print(f"\n    ── Original (Chinese) ──")
                print(f"    {r['original_text']}")
                print(f"\n    ── Translation (English) ──")
                print(f"    {r['translated_text']}")
            else:
                # Show preview
                if r['original_text']:
                    preview = r['original_text'][:100].replace('\n', ' ')
                    if len(r['original_text']) > 100:
                        preview += "..."
                    print(f"    Preview: {preview}")
        
        print(f"\n{'─' * 70}")
        print("Use 'view <number>' to see full text of a result")
    
    def cmd_search(self, args):
        """Search both Chinese and English."""
        if not args:
            print("Usage: search <terms> [--category <cat>] [--all] [--limit <n>]")
            return
        
        terms, category, match_all, limit = self.parse_search_args(args)
        if not terms:
            print("Please provide search terms.")
            return
        
        print(f"Searching for: {terms}")
        self.last_results = self.db.search(
            terms,
            search_chinese=True,
            search_english=True,
            category=category,
            match_all=match_all,
            limit=limit
        )
        self.display_results(self.last_results)
    
    def cmd_chinese(self, args):
        """Search Chinese text only."""
        if not args:
            print("Usage: chinese <terms> [--category <cat>] [--all] [--limit <n>]")
            return
        
        terms, category, match_all, limit = self.parse_search_args(args)
        if not terms:
            print("Please provide search terms.")
            return
        
        print(f"Searching Chinese for: {terms}")
        self.last_results = self.db.search_chinese(
            terms,
            category=category,
            match_all=match_all,
            limit=limit
        )
        self.display_results(self.last_results)
    
    def cmd_english(self, args):
        """Search English text only."""
        if not args:
            print("Usage: english <terms> [--category <cat>] [--all] [--limit <n>]")
            return
        
        terms, category, match_all, limit = self.parse_search_args(args)
        if not terms:
            print("Please provide search terms.")
            return
        
        print(f"Searching English for: {terms}")
        self.last_results = self.db.search_english(
            terms,
            category=category,
            match_all=match_all,
            limit=limit
        )
        self.display_results(self.last_results)
    
    def cmd_get(self, args):
        """Get a specific text."""
        if not args:
            print("Usage: get <source_id>")
            return
        
        source_id = ' '.join(args)
        text_data = self.db.get_text(source_id)
        
        if not text_data:
            print(f"Text not found: {source_id}")
            return
        
        print(f"\n{'═' * 70}")
        print(f"Title: {text_data['source_id']}")
        print(f"Category: {text_data['category']}")
        print(f"Chunks: {text_data['total_chunks']}")
        print(f"Original chars: {text_data['total_original_chars']:,}")
        print(f"Translated chars: {text_data['total_translated_chars']:,}")
        print(f"Model: {text_data['model_used']}")
        print('═' * 70)
        print("\n── Full Translation ──\n")
        print(text_data['full_translation'])
    
    def cmd_view(self, args):
        """View a specific result from last search."""
        if not args:
            print("Usage: view <number>")
            return
        
        if not self.last_results:
            print("No previous search results. Run a search first.")
            return
        
        try:
            idx = int(args[0]) - 1
            if 0 <= idx < len(self.last_results):
                result = self.last_results[idx]
                print(f"\n{'═' * 70}")
                print(f"Source: {result['source_id']}")
                print(f"Category: {result['category']}")
                if result['chunk_index'] is not None:
                    print(f"Chunk: {result['chunk_index']}")
                print('═' * 70)
                
                print("\n── Original (Chinese) ──\n")
                print(result['original_text'] or "(No original text in this result)")
                
                print("\n── Translation (English) ──\n")
                print(result['translated_text'] or "(No translation in this result)")
            else:
                print(f"Invalid result number. Use 1-{len(self.last_results)}")
        except ValueError:
            print("Please provide a valid number.")
    
    def cmd_stats(self, args):
        """Show database statistics."""
        stats = self.db.get_stats()
        
        print(f"\n{'═' * 50}")
        print("        Database Statistics")
        print('═' * 50)
        print(f"  Total texts:             {stats['total_texts']:,}")
        print(f"  Total chunks:            {stats['total_chunks']:,}")
        print(f"  Total original chars:    {stats['total_original_chars']:,}")
        print(f"  Total translated chars:  {stats['total_translated_chars']:,}")
        print(f"  Categories:              {len(stats['categories'])}")
        print('═' * 50)
    
    def cmd_categories(self, args):
        """List all categories."""
        categories = self.db.list_categories()
        
        print(f"\n{'═' * 50}")
        print(f"        Categories ({len(categories)})")
        print('═' * 50)
        
        for cat in sorted(categories, key=lambda x: x['count'], reverse=True):
            print(f"  {cat['category']}: {cat['count']} texts")
        
        print('═' * 50)
    
    def cmd_export(self, args):
        """Export a text to file."""
        if len(args) < 1:
            print("Usage: export <source_id> [format]")
            print("Formats: json, txt, md (default: md)")
            return
        
        source_id = args[0]
        fmt = args[1] if len(args) > 1 else 'md'
        
        output = self.db.export_text(source_id, fmt)
        if output:
            filename = f"{source_id}.{fmt}"
            # Sanitize filename
            filename = filename.replace('/', '_').replace('\\', '_')
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(output)
            print(f"Exported to: {filename}")
        else:
            print(f"Text not found: {source_id}")
    
    def cmd_quit(self, args):
        """Exit the program."""
        print("\nGoodbye! 道可道，非常道。")
        return True
    
    def run(self):
        """Run the interactive loop."""
        print("\n" + "═" * 70)
        print("     Daoist Translation Database - Interactive Search")
        print("═" * 70)
        print("Type 'help' for commands, 'quit' to exit.\n")
        
        # Show initial stats
        stats = self.db.get_stats()
        print(f"Database: {stats['total_texts']} texts, {stats['total_chunks']} chunks")
        print(f"          {stats['total_original_chars']:,} Chinese chars → {stats['total_translated_chars']:,} English chars")
        print()
        
        while True:
            try:
                line = input("道藏> ").strip()
                if not line:
                    continue
                
                # Parse command and arguments
                try:
                    parts = shlex.split(line)
                except ValueError:
                    parts = line.split()
                
                if not parts:
                    continue
                
                cmd = parts[0].lower()
                args = parts[1:]
                
                if cmd in self.commands:
                    result = self.commands[cmd](args)
                    if result:  # quit command returns True
                        break
                else:
                    # Treat as search if not a command
                    self.cmd_search(parts)
                    
            except KeyboardInterrupt:
                print("\n(Use 'quit' to exit)")
            except EOFError:
                break


def main():
    import sys
    db_path = sys.argv[1] if len(sys.argv) > 1 else "daoist_texts.db"
    
    search = InteractiveSearch(db_path)
    search.run()


if __name__ == '__main__':
    main()
