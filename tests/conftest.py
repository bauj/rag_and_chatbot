# tests/conftest.py
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
# Make chatbot and extraction importable in all test files
sys.path.insert(0, str(ROOT / "chatbot"))      # chatbot takes priority
sys.path.append(str(ROOT / "extraction"))       # extraction appended (lower priority)
