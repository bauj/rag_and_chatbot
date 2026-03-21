import sys
from pathlib import Path

# Add chatbot/core to sys.path for imports
CHATBOT_DIR = Path(__file__).parent.parent
if str(CHATBOT_DIR) not in sys.path:
    sys.path.insert(0, str(CHATBOT_DIR))
