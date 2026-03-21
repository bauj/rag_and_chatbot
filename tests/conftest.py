# tests/conftest.py
import sys
from pathlib import Path

# Make chatbot and extraction importable in all test files
sys.path.insert(0, str(Path(__file__).parent.parent / "chatbot"))
sys.path.insert(0, str(Path(__file__).parent.parent / "extraction"))
