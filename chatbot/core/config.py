"""
Configuration management for SALOME Documentation Chatbot
Supports defaults, JSON config files, and CLI overrides
"""

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional


@dataclass
class ChatbotConfig:
    """Configuration for SALOME Chatbot with hybrid JSON/code support"""

    # Paths
    chromadb_path: str = "../extraction/salome_docs_extracted/chromadb"

    # LLM settings
    base_url: str = "http://localhost:8080/v1"
    model_name: str = "mistral"
    api_key: str = "dummy"  # API key for LLM endpoint (use "dummy" for local models)
    ssl_cert_file: str = ""  # Path to SSL certificate bundle (empty = use system default)

    # Embedding settings
    embedding_model: str = "all-MiniLM-L6-v2"

    # Retrieval parameters
    k_standard: int = 40
    k_deep_dive: int = 60
    deep_dive_batch_size: int = 10

    # LLM parameters
    temperature: float = 0.0
    max_tokens: int = 2000

    # Response style presets (for UI)
    RESPONSE_STYLES = {
        "Precise (Recommended)": {
            "temperature": 0.0,
            "k": 40,
            "deep_dive": False
        },
        "Balanced": {
            "temperature": 0.2,
            "k": 50,
            "deep_dive": False
        },
        "Comprehensive": {
            "temperature": 0.1,
            "k": 60,
            "deep_dive": True
        }
    }

    @classmethod
    def from_json(cls, path: str) -> 'ChatbotConfig':
        """
        Load configuration from JSON file

        Args:
            path: Path to JSON config file

        Returns:
            ChatbotConfig instance

        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If JSON is malformed or has invalid values
        """
        config_path = Path(path)

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        try:
            with open(config_path) as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in config file: {e}")

        # Filter out comment keys (keys starting with _comment)
        config_data = {k: v for k, v in data.items() if not k.startswith('_comment')}

        # Validate that all keys are valid config parameters
        valid_keys = set(cls.__annotations__.keys())
        invalid_keys = set(config_data.keys()) - valid_keys

        if invalid_keys:
            raise ValueError(f"Invalid config keys: {invalid_keys}")

        return cls(**config_data)

    @classmethod
    def load(cls, config_file: Optional[str] = None) -> 'ChatbotConfig':
        """
        Load configuration with priority: JSON > defaults

        Args:
            config_file: Optional path to JSON config file

        Returns:
            ChatbotConfig instance
        """
        if config_file:
            return cls.from_json(config_file)

        # Check for default config.json in current directory
        default_config = Path("config.json")
        if default_config.exists():
            return cls.from_json(str(default_config))

        # Use defaults
        return cls()

    def to_json(self, path: str) -> None:
        """
        Save configuration to JSON file

        Args:
            path: Path where to save config
        """
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)

    def __str__(self) -> str:
        """Pretty print configuration"""
        lines = ["ChatbotConfig:"]
        for key, value in asdict(self).items():
            lines.append(f"  {key}: {value}")
        return "\n".join(lines)
