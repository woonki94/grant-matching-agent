from pathlib import Path

def build_prompt(prompt_path: Path, corpus: str, max_keywords: int = 30) -> str:
    template = prompt_path.read_text(encoding="utf-8")
    # Only replace the two placeholders; leave all other braces alone
    return (template
            .replace("{corpus}", corpus)
            .replace("{max_keywords}", str(max_keywords)))