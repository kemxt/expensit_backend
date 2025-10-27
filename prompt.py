from pathlib import Path

PROMPT_PATH = Path(__file__).parent / "prompt.txt"

def load_prompt() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8")