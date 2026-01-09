"""Generate descriptions for SMT instances using GPT-5-mini."""

import argparse
import sys
from pathlib import Path

from openai import OpenAI


def generate_description(
    smt_file_path: str | Path, api_key: str | None = None, model: str = "gpt-5-mini"
) -> str:
    """
    Generate a description of an SMT instance using GPT-5-mini.

    Args:
        smt_file_path: Path to the SMT file (.smt2 or .smt)
        api_key: OpenAI API key. If None, uses OPENAI_API_KEY environment variable
        model: Model name to use (default: "gpt-5-mini")

    Returns:
        Generated description string

    Raises:
        FileNotFoundError: If the SMT file doesn't exist
        ValueError: If the SMT file is empty
        Exception: If API call fails
    """
    smt_path = Path(smt_file_path)
    if not smt_path.exists():
        raise FileNotFoundError(f"SMT file not found: {smt_path}")

    # Read SMT file content
    with open(smt_path, "r", encoding="utf-8") as f:
        smt_content = f.read().strip()

    if not smt_content:
        raise ValueError(f"SMT file is empty: {smt_path}")

    # Initialize OpenAI client
    client = OpenAI(api_key=api_key)

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

Description:"""

    try:
        # Call OpenAI API
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert in Satisfiability Modulo Theories (SMT).",
                },
                {"role": "user", "content": prompt},
            ],
        )

        description = response.choices[0].message.content.strip()
        return description

    except Exception as e:
        raise Exception(f"Failed to generate description: {e}") from e


def main():
    """Command-line interface for generate_description."""
    parser = argparse.ArgumentParser(
        description="Generate a description of an SMT instance using GPT-5-mini.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (requires OPENAI_API_KEY environment variable)
  python -m src.generate_desc path/to/instance.smt2

  # Specify API key explicitly
  python -m src.generate_desc path/to/instance.smt2 --api-key sk-...

  # Use different model and adjust max tokens
  python -m src.generate_desc path/to/instance.smt2 --model gpt-5-mini --max-tokens 1000
        """,
    )
    parser.add_argument(
        "smt_file_path",
        type=str,
        help="Path to the SMT file (.smt2 or .smt)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="OpenAI API key. If not provided, uses OPENAI_API_KEY environment variable",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5-mini",
        help="Model name to use (default: gpt-5-mini)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=500,
        help="Maximum tokens for the generated description (default: 500)",
    )

    args = parser.parse_args()

    try:
        description = generate_description(
            smt_file_path=args.smt_file_path, api_key=args.api_key, model=args.model
        )
        print(description)
        return 0
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
