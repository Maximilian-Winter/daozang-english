"""
Daoist Translation Database
===========================
A SQLAlchemy-based database for storing and searching translated Daoist texts.
Supports searching in both Chinese (original) and English (translated) text.
"""

import json
import os
import re
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Any, Union

from sqlalchemy import (
    create_engine, Column, Integer, String, Text, Float, 
    DateTime, ForeignKey, Index, and_, or_, func
)
from sqlalchemy.orm import (
    declarative_base, relationship, sessionmaker, Session
)

Base = declarative_base()


class DaoistText(Base):
    """Main table for Daoist texts (one per JSON file)."""
    __tablename__ = 'daoist_texts'
    
    id = Column(Integer, primary_key=True)
    source_id = Column(String(500), unique=True, nullable=False, index=True)
    source_file = Column(String(1000))
    category = Column(String(500), index=True)
    total_chunks = Column(Integer)
    total_original_chars = Column(Integer)
    total_translated_chars = Column(Integer)
    total_translation_time = Column(Float)
    model_used = Column(String(200))
    full_translation = Column(Text)
    
    # Metadata fields
    metadata_source_file = Column(String(1000))
    metadata_category = Column(String(500))
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationship to chunks
    chunks = relationship("TextChunk", back_populates="text", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<DaoistText(source_id='{self.source_id}', category='{self.category}')>"


class TextChunk(Base):
    """Individual chunks of translated text."""
    __tablename__ = 'text_chunks'
    
    id = Column(Integer, primary_key=True)
    text_id = Column(Integer, ForeignKey('daoist_texts.id'), nullable=False)
    chunk_index = Column(Integer, nullable=False)
    
    # Content
    original_text = Column(Text, nullable=False)  # Chinese
    translated_text = Column(Text, nullable=False)  # English
    overlap_context = Column(Text)
    
    # Position info
    start_char = Column(Integer)
    end_char = Column(Integer)
    start_sentence = Column(Integer)
    end_sentence = Column(Integer)
    
    # Translation metadata
    model_used = Column(String(200))
    translation_time = Column(Float)
    
    # Relationship
    text = relationship("DaoistText", back_populates="chunks")
    
    # Indexes for faster searching
    __table_args__ = (
        Index('idx_chunk_original', 'original_text', mysql_length=255),
        Index('idx_chunk_translated', 'translated_text', mysql_length=255),
        Index('idx_text_chunk', 'text_id', 'chunk_index'),
    )
    
    def __repr__(self):
        return f"<TextChunk(text_id={self.text_id}, chunk_index={self.chunk_index})>"


class DaoistDatabase:
    """Main database interface for Daoist translations."""
    
    def __init__(self, db_path: str = "daoist_texts.db"):
        """
        Initialize database connection.
        
        Args:
            db_path: Path to SQLite database file (or SQLAlchemy connection string)
        """
        if "://" not in db_path:
            db_path = f"sqlite:///{db_path}"
        
        self.engine = create_engine(db_path, echo=False)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
    
    def get_session(self) -> Session:
        """Get a new database session."""
        return self.Session()
    
    def load_json_file(self, filepath: str, session: Optional[Session] = None) -> Optional[DaoistText]:
        """
        Load a single JSON file into the database.
        
        Args:
            filepath: Path to the JSON file
            session: Optional existing session (creates new if not provided)
        
        Returns:
            DaoistText object if successful, None if failed
        """
        close_session = session is None
        if session is None:
            session = self.get_session()
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Check if already exists
            existing = session.query(DaoistText).filter_by(
                source_id=data.get('source_id')
            ).first()
            
            if existing:
                print(f"  Skipping (already exists): {data.get('source_id')}")
                return existing
            
            # Extract metadata
            metadata = data.get('metadata', {})
            
            # Create main text record
            text = DaoistText(
                source_id=data.get('source_id'),
                source_file=data.get('source_file'),
                category=metadata.get('category'),
                total_chunks=data.get('total_chunks'),
                total_original_chars=data.get('total_original_chars'),
                total_translated_chars=data.get('total_translated_chars'),
                total_translation_time=data.get('total_translation_time'),
                model_used=data.get('model_used'),
                full_translation=data.get('full_translation'),
                metadata_source_file=metadata.get('source_file'),
                metadata_category=metadata.get('category'),
            )
            
            session.add(text)
            session.flush()  # Get the ID
            
            # Add chunks
            for chunk_data in data.get('chunks', []):
                chunk = TextChunk(
                    text_id=text.id,
                    chunk_index=chunk_data.get('chunk_index'),
                    original_text=chunk_data.get('original_text', ''),
                    translated_text=chunk_data.get('translated_text', ''),
                    overlap_context=chunk_data.get('overlap_context', ''),
                    start_char=chunk_data.get('start_char'),
                    end_char=chunk_data.get('end_char'),
                    start_sentence=chunk_data.get('start_sentence'),
                    end_sentence=chunk_data.get('end_sentence'),
                    model_used=chunk_data.get('model_used'),
                    translation_time=chunk_data.get('translation_time'),
                )
                session.add(chunk)
            
            session.commit()
            print(f"  Loaded: {text.source_id} ({text.total_chunks} chunks)")
            return text
            
        except Exception as e:
            session.rollback()
            print(f"  Error loading {filepath}: {e}")
            return None
        finally:
            if close_session:
                session.close()
    
    def load_folder(self, folder_path: str, recursive: bool = True) -> Dict[str, int]:
        """
        Load all JSON files from a folder into the database.
        
        Args:
            folder_path: Path to folder containing JSON files
            recursive: Whether to search subdirectories
        
        Returns:
            Dictionary with counts of loaded, skipped, and failed files
        """
        folder = Path(folder_path)
        pattern = "**/*.json" if recursive else "*.json"
        json_files = list(folder.glob(pattern))
        
        stats = {'loaded': 0, 'skipped': 0, 'failed': 0}
        
        print(f"Found {len(json_files)} JSON files in {folder_path}")
        
        session = self.get_session()
        try:
            for i, filepath in enumerate(json_files, 1):
                print(f"[{i}/{len(json_files)}] Processing: {filepath.name}")
                result = self.load_json_file(str(filepath), session)
                if result:
                    stats['loaded'] += 1
                else:
                    stats['failed'] += 1
        finally:
            session.close()
        
        print(f"\nSummary: {stats['loaded']} loaded, {stats['failed']} failed")
        return stats
    
    def search(
        self,
        terms: Union[str, List[str]],
        search_chinese: bool = True,
        search_english: bool = True,
        search_full_translation: bool = False,
        category: Optional[str] = None,
        match_all: bool = False,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Search for terms in the database.
        
        Args:
            terms: Single term or list of terms to search for
            search_chinese: Search in original Chinese text
            search_english: Search in English translation
            search_full_translation: Also search in full_translation field
            category: Filter by category (optional)
            match_all: If True, all terms must match; if False, any term matches
            limit: Maximum number of results
        
        Returns:
            List of dictionaries with search results
        """
        if isinstance(terms, str):
            terms = [terms]
        
        session = self.get_session()
        try:
            # Build search conditions for each term
            term_conditions = []
            for term in terms:
                term_conds = []
                pattern = f"%{term}%"
                
                if search_chinese:
                    term_conds.append(TextChunk.original_text.like(pattern))
                if search_english:
                    term_conds.append(TextChunk.translated_text.like(pattern))
                
                if term_conds:
                    term_conditions.append(or_(*term_conds))
            
            # Combine term conditions
            if match_all:
                search_filter = and_(*term_conditions) if term_conditions else True
            else:
                search_filter = or_(*term_conditions) if term_conditions else True
            
            # Build query
            query = session.query(TextChunk).join(DaoistText)
            
            if category:
                query = query.filter(DaoistText.category.like(f"%{category}%"))
            
            query = query.filter(search_filter).limit(limit)
            
            results = []
            for chunk in query.all():
                results.append({
                    'source_id': chunk.text.source_id,
                    'category': chunk.text.category,
                    'chunk_index': chunk.chunk_index,
                    'original_text': chunk.original_text,
                    'translated_text': chunk.translated_text,
                    'matched_terms': [t for t in terms if t in chunk.original_text or t in chunk.translated_text]
                })
            
            # Also search full translations if requested
            if search_full_translation:
                full_conditions = []
                for term in terms:
                    full_conditions.append(DaoistText.full_translation.like(f"%{term}%"))
                
                if match_all:
                    full_filter = and_(*full_conditions) if full_conditions else True
                else:
                    full_filter = or_(*full_conditions) if full_conditions else True
                
                full_query = session.query(DaoistText).filter(full_filter)
                if category:
                    full_query = full_query.filter(DaoistText.category.like(f"%{category}%"))
                
                for text in full_query.limit(limit).all():
                    # Only add if not already in results
                    if not any(r['source_id'] == text.source_id for r in results):
                        results.append({
                            'source_id': text.source_id,
                            'category': text.category,
                            'chunk_index': None,
                            'original_text': None,
                            'translated_text': text.full_translation[:500] + "..." if len(text.full_translation or '') > 500 else text.full_translation,
                            'matched_terms': [t for t in terms if t in (text.full_translation or '')]
                        })
            
            return results
            
        finally:
            session.close()
    
    def search_chinese(self, terms: Union[str, List[str]], **kwargs) -> List[Dict[str, Any]]:
        """Search only in Chinese original text."""
        return self.search(terms, search_chinese=True, search_english=False, **kwargs)
    
    def search_english(self, terms: Union[str, List[str]], **kwargs) -> List[Dict[str, Any]]:
        """Search only in English translation."""
        return self.search(terms, search_chinese=False, search_english=True, **kwargs)
    
    def get_text(self, source_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a complete text by its source_id.
        
        Args:
            source_id: The unique source identifier
        
        Returns:
            Dictionary with full text data or None if not found
        """
        session = self.get_session()
        try:
            text = session.query(DaoistText).filter_by(source_id=source_id).first()
            if not text:
                return None
            
            return {
                'source_id': text.source_id,
                'source_file': text.source_file,
                'category': text.category,
                'total_chunks': text.total_chunks,
                'total_original_chars': text.total_original_chars,
                'total_translated_chars': text.total_translated_chars,
                'model_used': text.model_used,
                'full_translation': text.full_translation,
                'chunks': [
                    {
                        'chunk_index': c.chunk_index,
                        'original_text': c.original_text,
                        'translated_text': c.translated_text,
                        'translation_time': c.translation_time,
                    }
                    for c in sorted(text.chunks, key=lambda x: x.chunk_index)
                ]
            }
        finally:
            session.close()
    
    def list_categories(self) -> List[Dict[str, int]]:
        """List all categories with text counts."""
        session = self.get_session()
        try:
            results = session.query(
                DaoistText.category,
                func.count(DaoistText.id).label('count')
            ).group_by(DaoistText.category).all()
            
            return [{'category': r[0], 'count': r[1]} for r in results]
        finally:
            session.close()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        session = self.get_session()
        try:
            text_count = session.query(DaoistText).count()
            chunk_count = session.query(TextChunk).count()
            
            total_chars = session.query(
                func.sum(DaoistText.total_original_chars)
            ).scalar() or 0
            
            total_translated = session.query(
                func.sum(DaoistText.total_translated_chars)
            ).scalar() or 0
            
            return {
                'total_texts': text_count,
                'total_chunks': chunk_count,
                'total_original_chars': total_chars,
                'total_translated_chars': total_translated,
                'categories': self.list_categories()
            }
        finally:
            session.close()
    
    def export_text(self, source_id: str, output_format: str = 'json') -> Optional[str]:
        """
        Export a text in various formats.
        
        Args:
            source_id: The source identifier
            output_format: 'json', 'txt', or 'md' (markdown)
        
        Returns:
            Formatted string or None if not found
        """
        text_data = self.get_text(source_id)
        if not text_data:
            return None
        
        if output_format == 'json':
            return json.dumps(text_data, ensure_ascii=False, indent=2)
        
        elif output_format == 'txt':
            output = f"Title: {text_data['source_id']}\n"
            output += f"Category: {text_data['category']}\n"
            output += "=" * 60 + "\n\n"
            output += text_data['full_translation'] or ''
            return output
        
        elif output_format == 'md':
            output = f"# {text_data['source_id']}\n\n"
            output += f"**Category:** {text_data['category']}\n\n"
            output += "---\n\n"
            output += text_data['full_translation'] or ''
            return output
        
        return None


def main():
    """Command-line interface for the database."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Daoist Translation Database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Load all JSON files from a folder
  python daoist_db.py load /path/to/translations

  # Search for terms (Chinese or English)
  python daoist_db.py search "tiger" "dragon"
  
  # Search only Chinese
  python daoist_db.py search --chinese "道"
  
  # Search only English with category filter
  python daoist_db.py search --english --category "正一部" "immortal"
  
  # Get a specific text
  python daoist_db.py get "洞真太上神虎隐文"
  
  # Show statistics
  python daoist_db.py stats
  
  # List all categories
  python daoist_db.py categories
        """
    )
    
    parser.add_argument('--db', default='daoist_texts.db',
                        help='Database file path (default: daoist_texts.db)')
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Load command
    load_parser = subparsers.add_parser('load', help='Load JSON files into database')
    load_parser.add_argument('folder', help='Folder containing JSON files')
    load_parser.add_argument('--no-recursive', action='store_true',
                            help='Do not search subdirectories')
    
    # Search command
    search_parser = subparsers.add_parser('search', help='Search for terms')
    search_parser.add_argument('terms', nargs='+', help='Search terms')
    search_parser.add_argument('--chinese', action='store_true',
                              help='Search only Chinese text')
    search_parser.add_argument('--english', action='store_true',
                              help='Search only English text')
    search_parser.add_argument('--category', help='Filter by category')
    search_parser.add_argument('--match-all', action='store_true',
                              help='Require all terms to match')
    search_parser.add_argument('--limit', type=int, default=20,
                              help='Maximum results (default: 20)')
    
    # Get command
    get_parser = subparsers.add_parser('get', help='Get a specific text')
    get_parser.add_argument('source_id', help='Source ID of the text')
    get_parser.add_argument('--format', choices=['json', 'txt', 'md'],
                           default='md', help='Output format')
    
    # Stats command
    subparsers.add_parser('stats', help='Show database statistics')
    
    # Categories command
    subparsers.add_parser('categories', help='List all categories')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    db = DaoistDatabase(args.db)
    
    if args.command == 'load':
        db.load_folder(args.folder, recursive=not args.no_recursive)
    
    elif args.command == 'search':
        search_chinese = not args.english or args.chinese
        search_english = not args.chinese or args.english
        
        results = db.search(
            args.terms,
            search_chinese=search_chinese,
            search_english=search_english,
            category=args.category,
            match_all=args.match_all,
            limit=args.limit
        )
        
        print(f"\nFound {len(results)} results:\n")
        for i, r in enumerate(results, 1):
            print(f"--- Result {i} ---")
            print(f"Source: {r['source_id']}")
            print(f"Category: {r['category']}")
            print(f"Matched: {', '.join(r['matched_terms'])}")
            if r['chunk_index'] is not None:
                print(f"Chunk: {r['chunk_index']}")
            print(f"\nOriginal (Chinese):")
            if r['original_text']:
                print(r['original_text'][:300] + "..." if len(r['original_text'] or '') > 300 else r['original_text'])
            print(f"\nTranslation (English):")
            if r['translated_text']:
                print(r['translated_text'][:500] + "..." if len(r['translated_text'] or '') > 500 else r['translated_text'])
            print()
    
    elif args.command == 'get':
        output = db.export_text(args.source_id, args.format)
        if output:
            print(output)
        else:
            print(f"Text not found: {args.source_id}")
    
    elif args.command == 'stats':
        stats = db.get_stats()
        print(f"\n=== Database Statistics ===")
        print(f"Total texts: {stats['total_texts']}")
        print(f"Total chunks: {stats['total_chunks']}")
        print(f"Total original characters: {stats['total_original_chars']:,}")
        print(f"Total translated characters: {stats['total_translated_chars']:,}")
        print(f"\nCategories: {len(stats['categories'])}")
    
    elif args.command == 'categories':
        categories = db.list_categories()
        print(f"\n=== Categories ({len(categories)}) ===")
        for cat in sorted(categories, key=lambda x: x['count'], reverse=True):
            print(f"  {cat['category']}: {cat['count']} texts")


if __name__ == '__main__':
    main()
