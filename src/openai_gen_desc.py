"""Generate descriptions for SMT instances using GPT-5-mini."""

import argparse
import json
import sys
from pathlib import Path

from openai import OpenAI

from .prompt import create_prompt_from_smt_file


def openai_gen_desc(
    smt_file_path: str | Path,
    api_key: str | None = None,
    model: str = "gpt-5-mini",
    reasoning_effort: str | None = None,
):
    """
    Generate a description of an SMT instance using GPT-5-mini.

    Args:
        smt_file_path: Path to the SMT file (.smt2)
        api_key: OpenAI API key. If None, uses OPENAI_API_KEY environment variable
        model: Model name to use (default: "gpt-5-mini")
        reasoning_effort: Reasoning effort level.
                         Options may be model specific. For gpt-5-mini: "minimal", "low", "medium", "high".
                         If None, uses API default for the model.

    Returns:
        Response object from OpenAI API

    Raises:
        FileNotFoundError: If the SMT file doesn't exist
        ValueError: If the SMT file is empty
        Exception: If API call fails
    """
    # Create prompt from SMT file
    prompt = create_prompt_from_smt_file(smt_file_path)

    # Initialize OpenAI client
    client = OpenAI(api_key=api_key)

    try:
        # Call OpenAI Responses API
        api_params = {
            "model": model,
            "instructions": "You are an expert in Satisfiability Modulo Theories (SMT), who helps generate descriptions of SMT instances to support understanding and algorithm selection, including insights into factors that may affect solving difficulty.",
            "input": prompt,
        }

        # Add reasoning effort if specified
        if reasoning_effort is not None:
            api_params["reasoning"] = {"effort": reasoning_effort}

        response = client.responses.create(**api_params)
        return response

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

  # Use different model
  python -m src.generate_desc path/to/instance.smt2 --model gpt-5-mini

  # Specify reasoning effort
  python -m src.generate_desc path/to/instance.smt2 --reasoning-effort minimal

  # Save full response to JSON file
  python -m src.generate_desc path/to/instance.smt2 --output-json response.json
        """,
    )
    parser.add_argument(
        "smt_file_path",
        type=str,
        help="Path to the SMT file (.smt2)",
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
        "--reasoning-effort",
        type=str,
        default=None,
        choices=["minimal", "low", "medium", "high"],
        help="Reasoning effort level (model-specific, e.g., for gpt-5-mini). "
        "Options: minimal, low (default), medium, high. "
        "If not specified, uses API default for the model. May not be supported by all models.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional path to save full response object as JSON file",
    )

    args = parser.parse_args()

    try:
        response = openai_gen_desc(
            smt_file_path=args.smt_file_path,
            api_key=args.api_key,
            model=args.model,
            reasoning_effort=args.reasoning_effort,
        )
        print(response.output_text)

        # Save response to JSON if output path specified
        if args.output_json:
            output_path = Path(args.output_json)
            response_dict = response.model_dump()

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(response_dict, f, indent=2, ensure_ascii=False, default=str)
            print(f"Response saved to: {output_path}", file=sys.stderr)

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
