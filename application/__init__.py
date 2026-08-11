"""rag-multimodal application package (FastAPI + React Web UI + LangGraph agent)."""

import os
import sys

_APP_DIR = os.path.dirname(os.path.abspath(__file__))
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)
