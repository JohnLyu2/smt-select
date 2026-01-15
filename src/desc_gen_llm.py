"""Generate descriptions for SMT instances using LLM API."""

import argparse
import json
import sys
from pathlib import Path

from openai import OpenAI

from .prompt import create_prompt_from_smt_file


def openai_gen_desc(
    smt_file_path: str | Path,
    api_key: str | None = None,
    model: str = "openai-gpt-oss-120b",
    base_url: str | None = None,
    reasoning_effort: str | None = None,
    verbosity: str | None = None,
    char_limit: int = 200000,
    prompt_only: bool = False,
) -> tuple[object, bool]:
    """
    Generate a description of an SMT instance using LLM API.

    Args:
        smt_file_path: Path to the SMT file (.smt2)
        api_key: OpenAI API key. If None, uses OPENAI_API_KEY environment variable
        model: Model name to use (default: "openai-gpt-oss-120b")
        base_url: Base URL for the API endpoint. If None, uses OpenAI's default endpoint.
        reasoning_effort: Reasoning effort level.
                         Options may be model specific. For gpt-5-mini: "minimal", "low", "medium", "high".
                         If None, uses API default for the model.
        verbosity: Verbosity level for the response. Options: "low", "medium" (default), "high".
                   Controls the level of detail in the model's response.
                   If None, uses API default for the model.
        char_limit: Maximum number of characters to include from SMT content (default: 20000).
                   Content exceeding this limit will be truncated.
        prompt_only: If True, only return the prompt string without calling the model (default: False).

    Returns:
        Tuple of (response_or_prompt, is_truncated).
        response_or_prompt is a Response object from OpenAI API, or prompt string if
        prompt_only=True.
    """
    # Create prompt from SMT file
    prompt, is_truncated = create_prompt_from_smt_file(
        smt_file_path, char_limit=char_limit
    )

    # If prompt_only is True, just return the prompt
    if prompt_only:
        return prompt, is_truncated

    # Initialize OpenAI client
    client_kwargs = {"api_key": api_key}
    if base_url is not None:
        client_kwargs["base_url"] = base_url
    client = OpenAI(**client_kwargs)

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

        # Add verbosity if specified (within text object for Responses API)
        if verbosity is not None:
            api_params["text"] = {"verbosity": verbosity}

        response = client.responses.create(**api_params)
        return response, is_truncated

    except Exception as e:
        raise Exception(f"Failed to generate description: {e}") from e


def main():
    """Command-line interface for generate_description."""
    parser = argparse.ArgumentParser(
        description="Generate a description of an SMT instance using LLM API.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (requires OPENAI_API_KEY environment variable)
  python -m src.desc_gen_llm path/to/instance.smt2

  # Just print the prompt without calling the model
  python -m src.desc_gen_llm path/to/instance.smt2 --prompt

  # Specify API key explicitly
  python -m src.desc_gen_llm path/to/instance.smt2 --api-key sk-...

  # Use different model
  python -m src.desc_gen_llm path/to/instance.smt2 --model gpt-5-mini

  # Use custom API endpoint
  python -m src.desc_gen_llm path/to/instance.smt2 --base-url https://api.example.com/v1

  # Specify reasoning effort
  python -m src.desc_gen_llm path/to/instance.smt2 --reasoning-effort minimal

  # Specify verbosity level
  python -m src.desc_gen_llm path/to/instance.smt2 --verbosity high

  # Save full response to JSON file
  python -m src.desc_gen_llm path/to/instance.smt2 --output-json response.json
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
        "--base-url",
        type=str,
        default=None,
        help="Base URL for the API endpoint. If not provided, uses OpenAI's default endpoint.",
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
        "--verbosity",
        type=str,
        default=None,
        choices=["low", "medium", "high"],
        help="Verbosity level for the response. Options: low, medium (default), high. "
        "Controls the level of detail in the model's response. "
        "If not specified, uses API default for the model.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional path to save full response object as JSON file",
    )
    parser.add_argument(
        "--char-limit",
        type=int,
        default=200000,
        help="Maximum number of characters to include from SMT content (default: 20000). "
        "Content exceeding this limit will be truncated.",
    )
    parser.add_argument(
        "--prompt",
        action="store_true",
        help="If specified, only print the prompt without calling the model",
    )

    args = parser.parse_args()

    try:
        result, is_truncated = openai_gen_desc(
            smt_file_path=args.smt_file_path,
            api_key=args.api_key,
            model=args.model,
            base_url=args.base_url,
            reasoning_effort=args.reasoning_effort,
            verbosity=args.verbosity,
            char_limit=args.char_limit,
            prompt_only=args.prompt,
        )

        # If prompt_only, result is a string (the prompt)
        if args.prompt:
            print(result)
            return 0

        # Otherwise, result is a response object
        print(result.output_text)

        # Save response to JSON if output path specified
        if args.output_json:
            output_path = Path(args.output_json)
            response_dict = result.model_dump()

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
