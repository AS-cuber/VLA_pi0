"""Model tools shared with the GPU server."""
from pathlib import Path
import sys

_model_dir = str(Path(__file__).resolve().parent)
if _model_dir not in sys.path:
    sys.path.insert(0, _model_dir)
