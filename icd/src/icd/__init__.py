"""
ICD - ICR Daemon

The data plane component of the Intelligent Code Retrieval system.
Handles file watching, indexing, embedding, retrieval, and pack compilation.
"""

__version__ = "0.1.0"
__all__ = [
    "Config",
    "ICDService",
]

from icd.config import Config
from icd.main import ICDService
