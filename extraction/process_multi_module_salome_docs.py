"""
Local Multi-Module SALOME Documentation Processor
Process SHAPER, SMESH, and GUI documentation from local HTML files
Handles both dev docs (Doxygen) and user docs (Sphinx)
"""

import json
import re
import gc
from pathlib import Path
from typing import List, Dict, Optional, Iterator
from bs4 import BeautifulSoup
from dataclasses import dataclass, asdict
import os

# For token-aware chunking
try:
    from transformers import AutoTokenizer
    TOKENIZER_AVAILABLE = True
except ImportError:
    TOKENIZER_AVAILABLE = False
    print("WARNING: transformers not installed - using character-based chunking")
    print("   Install: pip install transformers")

@dataclass
class DocumentChunk:
    """Represents a chunk of documentation"""
    title: str
    content: str
    url: str
    doc_type: str
    hierarchy: str
    chunk_id: int
    module: str  # SHAPER, SMESH, GUI
    doc_category: str  # 'dev' or 'user'
    metadata: Dict  # Extended metadata including code blocks, quality score, parent doc

class LocalMultiModuleProcessor:
    """Process multiple SALOME modules from local HTML directories"""

    # Default module descriptions and URLs (can be overridden by config)
    DEFAULT_MODULE_INFO = {
        'SHAPER': {
            'description': 'CAD modeling and geometry creation',
            'url': 'https://docs.salome-platform.org/latest/tui/SHAPER'
        },
        'SMESH': {
            'description': 'Mesh generation and manipulation',
            'url': 'https://docs.salome-platform.org/latest/tui/SMESH'
        },
        'GUI': {
            'description': 'Graphical user interface components',
            'url': 'https://docs.salome-platform.org/latest/tui/GUI'
        }
    }

    @classmethod
    def from_config_file(cls, config_path: str):
        """Create processor from JSON config file"""
        with open(config_path, 'r') as f:
            config = json.load(f)

        # Create processor with config settings
        output_dir = config.get('output_dir', './salome_docs_extracted')
        use_token_chunking = config.get('use_token_chunking', True)
        processor = cls(output_dir, use_token_chunking)

        # Override chunking parameters if specified
        if 'chunking' in config:
            processor.max_tokens = config['chunking'].get('max_tokens', 384)
            processor.overlap_tokens = config['chunking'].get('overlap_tokens', 50)
            processor.char_chunk_size = config['chunking'].get('char_chunk_size', 1000)
            processor.char_overlap = config['chunking'].get('char_overlap', 200)

        # Override quality parameters if specified
        if 'quality' in config:
            processor.quality_threshold = config['quality'].get('min_score', 0.3)
            processor.min_word_count = config['quality'].get('min_word_count', 50)
            processor.substantial_word_count = config['quality'].get('substantial_word_count', 100)

        # Load modules from config
        if 'modules' in config:
            for module_name, module_config in config['modules'].items():
                # Update MODULE_INFO with config data
                if module_name not in processor.MODULE_INFO:
                    processor.MODULE_INFO[module_name] = {}

                if 'description' in module_config:
                    processor.MODULE_INFO[module_name]['description'] = module_config['description']
                if 'url' in module_config:
                    processor.MODULE_INFO[module_name]['url'] = module_config['url']

                # Add module paths
                dev_path = module_config.get('dev_path')
                user_path = module_config.get('user_path')
                if dev_path or user_path:
                    processor.add_module(module_name, dev_path, user_path)

        return processor
    
    def __init__(self, output_dir: str = "./salome_docs_extracted", use_token_chunking: bool = False):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.modules = {}  # {module_name: {'dev': path, 'user': path}}

        # Initialize MODULE_INFO from defaults
        self.MODULE_INFO = self.DEFAULT_MODULE_INFO.copy()

        # Configurable parameters (can be overridden from config file)
        self.max_tokens = 384
        self.overlap_tokens = 50
        self.char_chunk_size = 1000
        self.char_overlap = 200
        self.quality_threshold = 0.3
        self.min_word_count = 50
        self.substantial_word_count = 100

        # Initialize tokenizer for token-aware chunking
        self.use_token_chunking = use_token_chunking and TOKENIZER_AVAILABLE
        self.tokenizer = None
        if self.use_token_chunking:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    "sentence-transformers/all-MiniLM-L6-v2",
                    model_max_length=512  # Explicitly set to suppress warning
                )
                print("Token-aware chunking enabled (all-MiniLM-L6-v2, 512 token limit)")
            except Exception as e:
                print(f"Failed to load tokenizer: {e}")
                self.use_token_chunking = False
    
    def add_module(self, module_name: str, dev_path: str = None, user_path: str = None):
        """Add a module with dev and/or user docs"""
        if module_name not in self.modules:
            self.modules[module_name] = {}
        
        if dev_path:
            path = Path(dev_path)
            if path.exists():
                self.modules[module_name]['dev'] = path
                print(f"OK: Added {module_name} dev docs: {dev_path}")
            else:
                print(f"Warning: Dev path not found for {module_name}: {dev_path}")
        
        if user_path:
            path = Path(user_path)
            if path.exists():
                self.modules[module_name]['user'] = path
                print(f"OK: Added {module_name} user docs: {user_path}")
            else:
                print(f"Warning: User path not found for {module_name}: {user_path}")
        
        return bool(self.modules[module_name])
    
    def find_html_files(self, doc_path: Path) -> List[Path]:
        """Find all HTML files in a directory"""
        html_files = list(doc_path.rglob("*.html"))
        return html_files
    
    def extract_content_doxygen(self, soup: BeautifulSoup) -> Dict:
        """Extract content from Doxygen HTML"""
        content_dict = {'title': '', 'content': ''}
        
        title = soup.find('title')
        if title:
            content_dict['title'] = title.get_text(strip=True)
        
        # Doxygen uses 'contents' class
        contents = soup.find('div', class_='contents')
        if contents:
            for tag in contents.find_all(['script', 'style', 'noscript']):
                tag.decompose()
            content_dict['content'] = contents.get_text(separator='\n', strip=True)
        else:
            body = soup.find('body')
            if body:
                content_dict['content'] = body.get_text(separator='\n', strip=True)
        
        return content_dict
    
    def extract_content_sphinx(self, soup: BeautifulSoup) -> Dict:
        """Extract content from Sphinx HTML"""
        content_dict = {'title': '', 'content': ''}
        
        title = soup.find('title')
        if title:
            content_dict['title'] = title.get_text(strip=True)
        
        # Sphinx uses different classes
        # Try common Sphinx content containers
        content_areas = [
            soup.find('div', role='main'),
            soup.find('div', class_='document'),
            soup.find('div', class_='body'),
            soup.find('div', class_='section'),
            soup.find('article'),
        ]
        
        for area in content_areas:
            if area:
                # Remove navigation, sidebars
                for tag in area.find_all(['script', 'style', 'noscript', 'nav']):
                    tag.decompose()
                for cls in ['sphinxsidebar', 'related', 'footer']:
                    for elem in area.find_all(class_=cls):
                        elem.decompose()
                
                content_dict['content'] = area.get_text(separator='\n', strip=True)
                break
        
        # Fallback
        if not content_dict['content']:
            body = soup.find('body')
            if body:
                content_dict['content'] = body.get_text(separator='\n', strip=True)
        
        return content_dict
    
    def determine_doc_type(self, filepath: Path, doc_category: str) -> str:
        """Determine doc type from filename and category"""
        name = filepath.name.lower()
        
        if doc_category == 'dev':
            # Doxygen structure
            if name.startswith('class') or name.startswith('struct'):
                return 'class'
            elif name.startswith('namespace'):
                return 'namespace'
            elif 'modules' in name or 'group' in name:
                return 'module'
            else:
                return 'page'
        else:
            # Sphinx structure (user docs)
            if 'tutorial' in name:
                return 'tutorial'
            elif 'guide' in name or 'howto' in name:
                return 'guide'
            elif 'api' in name or 'reference' in name:
                return 'reference'
            else:
                return 'page'
    
    def chunk_text_by_tokens(self, text: str, max_tokens: int = 384, overlap_tokens: int = 50) -> List[str]:
        """Split text into chunks based on token count"""
        if not self.tokenizer:
            # Fallback to character-based
            return self.chunk_text_by_chars(text)

        # Tokenize
        tokens = self.tokenizer.encode(text, add_special_tokens=False)

        if len(tokens) <= max_tokens:
            return [text]

        chunks = []
        start = 0

        while start < len(tokens):
            end = min(start + max_tokens, len(tokens))
            chunk_tokens = tokens[start:end]

            # Decode back to text
            chunk_text = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)

            # Try to find semantic boundary in last part of chunk
            if end < len(tokens):
                # Look for sentence/paragraph breaks in decoded text
                last_100_chars = chunk_text[-100:] if len(chunk_text) > 100 else chunk_text
                last_period = last_100_chars.rfind('.')
                last_newline = last_100_chars.rfind('\n')

                if last_period > 0 or last_newline > 0:
                    break_point = max(last_period, last_newline)
                    # Re-encode to find token position
                    trimmed_text = chunk_text[:-(100 - break_point)]
                    trimmed_tokens = self.tokenizer.encode(trimmed_text, add_special_tokens=False)
                    end = start + len(trimmed_tokens)
                    chunk_text = trimmed_text

            chunks.append(chunk_text.strip())
            start = end - overlap_tokens

        return chunks

    def chunk_text_by_chars(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Split text into chunks (character-based fallback)"""
        if len(text) <= chunk_size:
            return [text]

        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            if end < len(text):
                last_period = text.rfind('.', end - 100, end)
                last_newline = text.rfind('\n', end - 100, end)
                break_point = max(last_period, last_newline)
                if break_point > start:
                    end = break_point + 1

            chunks.append(text[start:end].strip())
            start = end - overlap

        return chunks

    def chunk_text(self, text: str) -> List[str]:
        """Split text into chunks (uses token-aware if available)"""
        if self.use_token_chunking:
            return self.chunk_text_by_tokens(text, max_tokens=self.max_tokens, overlap_tokens=self.overlap_tokens)
        else:
            return self.chunk_text_by_chars(text, chunk_size=self.char_chunk_size, overlap=self.char_overlap)

    def extract_code_blocks(self, soup: BeautifulSoup) -> List[str]:
        """Extract code blocks from HTML"""
        code_blocks = []

        # Find all code/pre tags
        for tag in soup.find_all(['pre', 'code']):
            code_text = tag.get_text(strip=True)
            if len(code_text) > 10:  # Skip trivial snippets
                code_blocks.append(code_text)

        # Deduplicate (nested tags)
        unique_blocks = []
        for block in code_blocks:
            if not any(block in existing for existing in unique_blocks):
                unique_blocks.append(block)

        return unique_blocks[:5]  # Keep top 5 code blocks

    def calculate_quality_score(self, content: str, title: str) -> float:
        """Calculate content quality score (0.0 - 1.0)"""
        if not content or not content.strip():
            return 0.0

        # Count meaningful words (alphanumeric)
        words = re.findall(r'\b\w+\b', content)
        word_count = len(words)

        # Heuristics for quality
        has_title = len(title.strip()) > 0
        has_content = word_count > self.min_word_count
        substantial_content = word_count > self.substantial_word_count

        # Calculate score
        base_score = min(word_count / 200.0, 1.0)  # Normalize by expected length

        # Bonuses
        if has_title:
            base_score += 0.1
        if has_content:
            base_score += 0.1
        if substantial_content:
            base_score += 0.1

        return min(base_score, 1.0)

    def validate_chunk_token_length(self, content: str) -> bool:
        """
        Validate that chunk content doesn't exceed model's token limit
        Returns True if valid, False and logs warning if too long
        """
        if self.tokenizer is None:
            return True  # Can't validate without tokenizer

        try:
            tokens = self.tokenizer.encode(content, add_special_tokens=True, truncation=False)
            token_count = len(tokens)

            if token_count > 512:  # all-MiniLM-L6-v2 limit
                return False
            return True
        except:
            return True  # If encoding fails, assume it's OK

    def process_file(self, filepath: Path, module_name: str, doc_category: str) -> List[DocumentChunk]:
        """Process a single HTML file"""
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                soup = BeautifulSoup(f.read(), 'html.parser')
        except Exception as e:
            return []

        # Extract content based on doc type
        if doc_category == 'dev':
            content_dict = self.extract_content_doxygen(soup)
        else:
            content_dict = self.extract_content_sphinx(soup)

        doc_type = self.determine_doc_type(filepath, doc_category)

        # Generate URL and parent doc ID
        relative_path = Path(*filepath.parts[2:])
        url = self.MODULE_INFO[module_name]['url'] + f"/{relative_path}"
        parent_doc_id = self.MODULE_INFO[module_name]['url']

        if (doc_category == "user") :
            url = url.replace("tui","gui")
            parent_doc_id = parent_doc_id.replace("tui","gui")

        full_text = f"{content_dict['title']}\n\n{content_dict['content']}"

        # Calculate quality score
        quality_score = self.calculate_quality_score(content_dict['content'], content_dict['title'])

        # Filter low-quality content
        if len(full_text.strip()) < 50 or quality_score < self.quality_threshold:
            soup.decompose()  # Free BeautifulSoup memory
            del soup
            return []

        # Extract code blocks
        code_blocks = self.extract_code_blocks(soup)

        # Free BeautifulSoup memory immediately
        soup.decompose()
        del soup

        # Chunk the text
        text_chunks = self.chunk_text(full_text)

        doc_chunks = []
        skipped_chunks = 0

        for i, chunk_text in enumerate(text_chunks):
            # Validate chunk length before creating
            if not self.validate_chunk_token_length(chunk_text):
                skipped_chunks += 1
                continue  # Skip chunks that exceed token limit

            chunk = DocumentChunk(
                title=content_dict['title'],
                content=chunk_text,
                url=url,
                doc_type=doc_type,
                hierarchy=content_dict['title'],
                chunk_id=i,
                module=module_name,
                doc_category=doc_category,
                metadata={
                    'total_chunks': len(text_chunks),
                    'chunk_position': f"{i+1}/{len(text_chunks)}",
                    'parent_doc_id': parent_doc_id,
                    'source': f'SALOME {module_name} {doc_category.title()} Documentation',
                    'file': str(filepath.name),
                    'module_description': self.MODULE_INFO.get(module_name, {}).get('description', ''),
                    'quality_score': round(quality_score, 2),
                    'has_code': len(code_blocks) > 0,
                    'code_blocks': code_blocks if len(code_blocks) > 0 else []
                }
            )
            doc_chunks.append(chunk)

        # Warning if chunks were skipped
        if skipped_chunks > 0:
            print(f"    Warning: Skipped {skipped_chunks} chunks (>512 tokens) in {filepath.name}")
        
        return doc_chunks
    
    def process_doc_category_generator(self, module_name: str, doc_path: Path, doc_category: str) -> Iterator[DocumentChunk]:
        """
        Process all files for a specific doc category (dev or user) - memory efficient version
        Yields chunks as they're created instead of accumulating in memory
        """
        html_files = self.find_html_files(doc_path)

        if not html_files:
            print(f"    Warning: No HTML files found")
            return

        print(f"    Found {len(html_files)} files")

        chunk_count = 0
        for i, filepath in enumerate(html_files, 1):
            chunks = self.process_file(filepath, module_name, doc_category)
            for chunk in chunks:
                yield chunk
                chunk_count += 1

            # Progress and periodic garbage collection
            if i % 100 == 0:
                gc.collect()
                print(f"    Processed {i}/{len(html_files)} files ({chunk_count} chunks so far)")

        print(f"    OK: Extracted {chunk_count} chunks")

    def process_doc_category(self, module_name: str, doc_path: Path, doc_category: str) -> List[DocumentChunk]:
        """Process all files for a specific doc category (dev or user) - returns list"""
        print(f"  Processing {doc_category} docs...")
        return list(self.process_doc_category_generator(module_name, doc_path, doc_category))
    
    def process_module(self, module_name: str, module_paths: Dict) -> List[DocumentChunk]:
        """Process all doc categories for a module"""
        print(f"\n{'='*70}")
        print(f"Processing {module_name} - {self.MODULE_INFO[module_name]}")
        print(f"{'='*70}")
        
        all_chunks = []
        
        if 'dev' in module_paths:
            chunks = self.process_doc_category(module_name, module_paths['dev'], 'dev')
            all_chunks.extend(chunks)
        
        if 'user' in module_paths:
            chunks = self.process_doc_category(module_name, module_paths['user'], 'user')
            all_chunks.extend(chunks)
        
        print(f"Total for {module_name}: {len(all_chunks)} chunks")
        return all_chunks
    
    def process_all(self) -> List[DocumentChunk]:
        """Process all modules and doc categories"""
        all_chunks = []
        
        for module_name, module_paths in self.modules.items():
            chunks = self.process_module(module_name, module_paths)
            all_chunks.extend(chunks)
        
        return all_chunks
    
    def save_chunks(self, chunks: List[DocumentChunk]):
        """Save chunks to JSON files"""
        # JSONL
        jsonl_file = self.output_dir / 'salome_docs.jsonl'
        with open(jsonl_file, 'w', encoding='utf-8') as f:
            for chunk in chunks:
                f.write(json.dumps(asdict(chunk), ensure_ascii=False) + '\n')
        print(f"\nOK: Saved to {jsonl_file}")
        
        # JSON
        json_file = self.output_dir / 'salome_docs.json'
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump([asdict(chunk) for chunk in chunks], f, ensure_ascii=False, indent=2)
        print(f"OK: Saved to {json_file}")
        
        # Statistics
        stats = {
            'total_chunks': len(chunks),
            'modules': {}
        }
        
        for module_name in self.modules.keys():
            module_chunks = [c for c in chunks if c.module == module_name]
            dev_chunks = [c for c in module_chunks if c.doc_category == 'dev']
            user_chunks = [c for c in module_chunks if c.doc_category == 'user']
            
            stats['modules'][module_name] = {
                'total_chunks': len(module_chunks),
                'dev_chunks': len(dev_chunks),
                'user_chunks': len(user_chunks),
                'description': self.MODULE_INFO.get(module_name, '')
            }
        
        stats_file = self.output_dir / 'statistics.json'
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2)
        print(f"OK: Statistics: {stats_file}")
    
    def create_chromadb(self, chunks: List[DocumentChunk]):
        """Create ChromaDB index with explicit embedding model - memory efficient version"""
        try:
            import chromadb
            from chromadb.config import Settings
            from chromadb.utils import embedding_functions

            print("\nCreating ChromaDB index...")

            # Create embedding function (same as chatbot uses)
            embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            print("  Using all-MiniLM-L6-v2 embeddings")

            client = chromadb.PersistentClient(
                path=str(self.output_dir / 'chromadb'),
                settings=Settings(anonymized_telemetry=False)
            )

            # Delete existing
            try:
                client.delete_collection("salome_documentation")
            except:
                pass

            # Create collection with explicit embedding and cosine similarity
            collection = client.create_collection(
                name="salome_documentation",
                embedding_function=embedding_function,
                metadata={
                    "description": "SALOME Multi-Module Documentation (Dev + User)",
                    "hnsw:space": "cosine"  # Use cosine similarity
                }
            )

            # Process in batches WITHOUT creating huge lists in memory
            batch_size = 100
            total_chunks = len(chunks)
            batch_num = 0

            for i in range(0, total_chunks, batch_size):
                batch_chunks = chunks[i:i + batch_size]

                # Build batch data on-the-fly
                batch_docs = []
                batch_metas = []
                batch_ids = []

                for idx, chunk in enumerate(batch_chunks):
                    batch_docs.append(chunk.content)
                    batch_metas.append({
                        'title': chunk.title,
                        'url': chunk.url,
                        'doc_type': chunk.doc_type,
                        'hierarchy': chunk.hierarchy,
                        'chunk_id': chunk.chunk_id,
                        'module': chunk.module,
                        'doc_category': chunk.doc_category,
                        'parent_doc_id': chunk.metadata.get('parent_doc_id', ''),
                        'chunk_position': chunk.metadata.get('chunk_position', ''),
                        'quality_score': chunk.metadata.get('quality_score', 0.0),
                        'has_code': chunk.metadata.get('has_code', False)
                    })
                    batch_ids.append(f"{i + idx:06d}")

                # Add batch to collection
                collection.add(
                    documents=batch_docs,
                    metadatas=batch_metas,
                    ids=batch_ids
                )

                batch_num += 1
                print(f"  Batch {batch_num}/{(total_chunks + batch_size - 1) // batch_size}")

                # Clear batch data and collect garbage
                del batch_docs, batch_metas, batch_ids, batch_chunks
                gc.collect()

            print(f"OK: ChromaDB created with {total_chunks} chunks")

        except ImportError:
            print("\nWarning: ChromaDB not installed: pip install chromadb")
        except Exception as e:
            print(f"\nWarning: ChromaDB error: {e}")


def main():
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Process local SALOME documentation (dev + user docs)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use config file (recommended for multiple modules)
  python %(prog)s --config config.json

  # Default: auto-detect from current directory
  python %(prog)s

  # Custom base directories
  python %(prog)s \\
    --shaper-dir ./shaper_docs_extracted \\
    --smesh-dir ./smesh_docs_extracted \\
    --gui-dir ./gui_docs_extracted

  # Manual paths for more control
  python %(prog)s \\
    --shaper-dev shaper_docs_extracted/html \\
    --shaper-user shaper_docs_extracted/html_gui
        """
    )

    # Config file option
    parser.add_argument('--config',
                       help='Path to JSON config file (defines modules, paths, and parameters)')
    
    # Auto-detect options (easier)
    parser.add_argument('--shaper-dir', 
                       default='./shaper_docs_extracted',
                       help='SHAPER base directory (default: ./shaper_docs_extracted)')
    parser.add_argument('--smesh-dir',
                       default='./smesh_docs_extracted', 
                       help='SMESH base directory (default: ./smesh_docs_extracted)')
    parser.add_argument('--gui-dir',
                       default='./gui_docs_extracted',
                       help='GUI base directory (default: ./gui_docs_extracted)')
    
    # Manual options (more control)
    parser.add_argument('--shaper-dev', help='SHAPER dev docs (html)')
    parser.add_argument('--shaper-user', help='SHAPER user docs (html_gui)')
    parser.add_argument('--smesh-dev', help='SMESH dev docs (html)')
    parser.add_argument('--smesh-user', help='SMESH user docs (html_gui)')
    parser.add_argument('--gui-dev', help='GUI dev docs (html)')
    parser.add_argument('--gui-user', help='GUI user docs (html_gui)')
    
    parser.add_argument('--output', default='./salome_docs_extracted',
                       help='Output directory (default: ./salome_docs_extracted)')
    
    parser.add_argument('--skip-missing', action='store_true',
                       help='Skip missing directories instead of failing')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("SALOME Multi-Module Documentation Processor")
    print("(Developer + User Documentation)")
    print("=" * 70)

    # Create processor from config file or defaults
    if args.config:
        print(f"\nLoading configuration from: {args.config}")
        try:
            processor = LocalMultiModuleProcessor.from_config_file(args.config)
            modules_added = len(processor.modules)
            print(f"Loaded {modules_added} module(s) from config")
        except FileNotFoundError:
            print(f"Config file not found: {args.config}")
            sys.exit(1)
        except json.JSONDecodeError as e:
            print(f"Invalid JSON in config file: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"Error loading config: {e}")
            sys.exit(1)
    else:
        # Use CLI arguments or defaults
        processor = LocalMultiModuleProcessor(args.output)

        # Add modules
        print("\nDetecting documentation directories...")
        modules_added = 0

        # SHAPER
        if args.shaper_dev or args.shaper_user:
            # Manual paths provided
            if processor.add_module('SHAPER',
                                   dev_path=args.shaper_dev,
                                   user_path=args.shaper_user):
                modules_added += 1
        else:
            # Auto-detect
            base = Path(args.shaper_dir)
            if base.exists():
                if processor.add_module('SHAPER',
                                   dev_path=str(base / 'html'),
                                   user_path=str(base / 'html_gui')):
                    modules_added += 1
            elif not args.skip_missing:
                print(f"Warning: SHAPER directory not found: {args.shaper_dir}")
                if not args.skip_missing:
                    print("  Use --skip-missing to continue without it")

        # SMESH
        if args.smesh_dev or args.smesh_user:
            # Manual paths
            if processor.add_module('SMESH',
                                   dev_path=args.smesh_dev,
                                   user_path=args.smesh_user):
                modules_added += 1
        else:
            # Auto-detect
            base = Path(args.smesh_dir)
            if base.exists():
                if processor.add_module('SMESH',
                                   dev_path=str(base / 'html'),
                                   user_path=str(base / 'html_gui')):
                    modules_added += 1
            elif not args.skip_missing:
                print(f"Warning: SMESH directory not found: {args.smesh_dir}")

        # GUI
        if args.gui_dev or args.gui_user:
            # Manual paths
            if processor.add_module('GUI',
                                   dev_path=args.gui_dev,
                                   user_path=args.gui_user):
                modules_added += 1
        else:
            # Auto-detect
            base = Path(args.gui_dir)
            if base.exists():
                if processor.add_module('GUI',
                                   dev_path=str(base / 'html'),
                                   user_path=str(base / 'html_gui')):
                    modules_added += 1
            elif not args.skip_missing:
                print(f"Warning: GUI directory not found: {args.gui_dir}")
    
    if modules_added == 0:
        print("\n" + "=" * 70)
        print("Error: No documentation directories found!")
        print("=" * 70)
        print("\nExpected structure:")
        print("  ./shaper_docs_extracted/html/")
        print("  ./shaper_docs_extracted/html_gui/")
        print("  ./smesh_docs_extracted/html/")
        print("  ./smesh_docs_extracted/html_gui/")
        print("  ./gui_docs_extracted/html/")
        print("  ./gui_docs_extracted/html_gui/")
        print("\nCurrent directory:", Path.cwd())
        print("\nUse --help for more options")
        sys.exit(1)
    
    print(f"\nOK: Found {modules_added} module(s) to process")
    
    # Process
    print("\nProcessing documentation...")
    chunks = processor.process_all()
    
    if not chunks:
        print("\nError: No content extracted")
        sys.exit(1)
    
    print(f"\n{'='*70}")
    print(f"TOTAL: {len(chunks)} chunks extracted")
    print(f"{'='*70}")
    
    # Save
    print("\nSaving files...")
    processor.save_chunks(chunks)
    processor.create_chromadb(chunks)
    
    print("\n" + "=" * 70)
    print("OK: Processing complete!")
    print("=" * 70)
    print(f"\nOutput: {args.output}")
    print("\nBreakdown:")
    for module_name in sorted(processor.modules.keys()):
        module_chunks = [c for c in chunks if c.module == module_name]
        dev_chunks = [c for c in module_chunks if c.doc_category == 'dev']
        user_chunks = [c for c in module_chunks if c.doc_category == 'user']
        print(f"  {module_name}:")
        print(f"    Dev:  {len(dev_chunks):5} chunks")
        print(f"    User: {len(user_chunks):5} chunks")
        print(f"    Total: {len(module_chunks):5} chunks")
    print("\nNext: Run the chatbot")
    print(f"  python salome_chatbot.py --chromadb {args.output}/chromadb")
    print("=" * 70)


if __name__ == "__main__":
    main()
