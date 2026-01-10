"""Functions for creating description prompts from SMT files."""

from pathlib import Path


def create_prompt_from_smt_file(smt_file_path: str | Path) -> str:
    """
    Read an SMT file and create a prompt for generating a description.

    Args:
        smt_file_path: Path to the SMT file (.smt2)

    Returns:
        Prompt string for the LLM

    Raises:
        FileNotFoundError: If the SMT file doesn't exist
        ValueError: If the SMT file is empty
    """
    smt_path = Path(smt_file_path)
    if not smt_path.exists():
        raise FileNotFoundError(f"SMT file not found: {smt_path}")

    # Read SMT file content
    with open(smt_path, "r", encoding="utf-8") as f:
        smt_content = f.read().strip()

    if not smt_content:
        raise ValueError(f"SMT file is empty: {smt_path}")

    # Create prompt
    prompt = f"""Analyze the following SMT-LIB instance and provide a concise description of what it represents. 
Focus on:
- The logic/theory used
- The main problem structure
- Key constraints or properties being checked
- Any notable characteristics

SMT-LIB instance:
```
{smt_content}
```

"""
    return prompt
