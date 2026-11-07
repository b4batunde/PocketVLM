"""SMOKE TESTS ONLY. Runs llama.cpp's convert_hf_to_gguf.py / convert_lora_to_gguf.py with the
pre-tokenizer check pinned to "qwen2". The real Qwen tokenizer is recognized by its hash; our tiny
locally-trained tokenizer (same regex, different vocab) is not. Usage: convert_shim.py <script> args...
"""
import runpy
import sys
from pathlib import Path

script = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(script.parent))
sys.path.insert(1, str(script.parent / "gguf-py"))
from conversion.base import TextModel  # noqa: E402

TextModel.get_vocab_base_pre = lambda self, tokenizer: "qwen2"
sys.argv = [str(script)] + sys.argv[2:]
runpy.run_path(str(script), run_name="__main__")
