"""Lista modelos disponibles (JSON stdout)."""
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from model_registry import list_models

if __name__ == "__main__":
    print(json.dumps(list_models(), ensure_ascii=False))
