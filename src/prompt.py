"""Functions for creating description prompts from SMT files."""

import json
from pathlib import Path

BASIC_INFO_FIELDS = [
    "smtlib_path",
    "logic",
    "category",
    "family",
    "size",
    "asserts_count",
    "declare_fun_count",
    "declare_const_count",
    "declare_sort_count",
    "define_fun_count",
    "define_fun_rec_count",
    "constant_fun_count",
    "define_sort_count",
    "declare_datatype_count",
    "max_term_depth",
]


def get_basic_info_from_json(
    smtlib_path: str, json_path: str | Path = "data/raw_jsons/BV.json"
) -> tuple[dict, dict]:
    """
    Get basic information for an SMT benchmark from JSON.

    Args:
        smtlib_path: SMT-LIB path to match (e.g., "BV/2017-Preiner-scholl-smt08/RND/RND_3_15.smt2")
        json_path: Path to the JSON file containing benchmark data (default: "data/raw_jsons/BV.json")

    Returns:
        Tuple of (basic_info, symbol_counts):
        - basic_info: Dictionary of basic info fields from BASIC_INFO_FIELDS
        - symbol_counts: Dictionary of symbol counts (only non-zero symbols)

    Raises:
        FileNotFoundError: If the JSON file doesn't exist
        ValueError: If the benchmark is not found in the JSON file
    """
    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")

    # Load JSON data
    with open(json_path, "r", encoding="utf-8") as f:
        benchmarks = json.load(f)

    # Find matching benchmark by smtlib_path
    matched_benchmark = None
    for benchmark in benchmarks:
        if "smtlib_path" in benchmark and benchmark["smtlib_path"] == smtlib_path:
            matched_benchmark = benchmark
            break

    if matched_benchmark is None:
        raise ValueError(f"Benchmark not found in JSON for smtlib_path: {smtlib_path}")

    # Extract basic info fields
    basic_info = {}
    for field in BASIC_INFO_FIELDS:
        basic_info[field] = matched_benchmark[field]

    # Extract symbol counts
    symbol_counts = {}
    if "symbol_counts" in matched_benchmark and matched_benchmark["symbol_counts"]:
        symbol_counts = {
            symbol: count
            for symbol, count in matched_benchmark["symbol_counts"].items()
            if count > 0
        }

    return basic_info, symbol_counts


def format_basic_info_text(basic_info: dict, symbol_counts: dict) -> str:
    """
    Format basic info and symbol counts into a text string for the prompt.

    Args:
        basic_info: Dictionary of basic info field defined in BASIC_INFO_FIELDS
        symbol_counts: Dictionary of symbol counts

    Returns:
        Formatted text string with basic information
    """
    text = "\n\nBasic Information about the instance:\n"
    if basic_info:
        # Core identification fields (in order of BASIC_INFO_FIELDS)
        if "smtlib_path" in basic_info:
            text += f"smtlib_path: {basic_info['smtlib_path']}\n"

        if "logic" in basic_info:
            text += f"logic: {basic_info['logic']}\n"

        if "category" in basic_info:
            text += f"category: {basic_info['category']}\n"

        if "family" in basic_info:
            text += f"family: {basic_info['family']}\n"

        if "size" in basic_info:
            text += f"size: {basic_info['size']}\n"

        # Count fields (skip zero values)
        count_fields = [
            "asserts_count",
            "declare_fun_count",
            "declare_const_count",
            "declare_sort_count",
            "define_fun_count",
            "define_fun_rec_count",
            "constant_fun_count",
            "define_sort_count",
            "declare_datatype_count",
        ]
        for field in count_fields:
            if field in basic_info and basic_info[field] > 0:
                display_name = field.replace("_count", " count").replace("_", "-")
                if display_name == "asserts count":
                    display_name = "assert count"
                text += f"{display_name}: {basic_info[field]}\n"

        # Last line: max_term_depth
        if "max_term_depth" in basic_info:
            text += f"max_term_depth: {basic_info['max_term_depth']}\n"

    if symbol_counts:
        # Show all symbols sorted by count
        sorted_symbols = sorted(symbol_counts.items(), key=lambda x: x[1], reverse=True)
        symbols_str = ", ".join(f"{k}: {v}" for k, v in sorted_symbols)
        text += f"\nSymbol counts: {symbols_str}\n"
    return text


def get_smt_content_from_file(
    smt_file_path: str | Path, char_limit: int = 20000
) -> str:
    """
    Read SMT file content and return it formatted with label and code fences.
    Content is optionally truncated to char_limit, with truncation notice if needed.

    Args:
        smt_file_path: Path to the SMT file (.smt2)
        char_limit: Maximum number of characters to include from SMT content (default: 20000).
                   Content exceeding this limit will be truncated.

    Returns:
        Formatted string with "SMT-LIB instance:" label, code fences, and content.
        If truncated, includes a truncation notice.
    """
    smt_path = Path(smt_file_path)
    if not smt_path.exists():
        raise FileNotFoundError(f"SMT file not found: {smt_path}")

    # Read SMT file content
    with open(smt_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Filter out lines starting with "(set-info :status"
    filtered_lines = [
        line for line in lines if not line.strip().startswith("(set-info :status")
    ]
    smt_content = "".join(filtered_lines).strip()

    if not smt_content:
        raise ValueError(f"SMT file is empty: {smt_path}")

    original_length = len(smt_content)
    was_truncated = original_length > char_limit

    # Truncate if exceeds char_limit
    truncation_notice = ""
    if was_truncated:
        smt_content = smt_content[:char_limit]
        truncated_chars = original_length - char_limit
        # Add ellipsis to indicate truncation
        smt_content += "\n..."
        truncation_notice = f"\n[Note: Content truncated. Original size: {original_length:,} characters. Truncated by: {truncated_chars:,} characters ({truncated_chars / original_length * 100:.1f}% removed). Only the first {char_limit:,} characters are shown above.]"

    # Format with label and code fences
    formatted_content = f"""SMT-LIB instance:
```
{smt_content}
```{truncation_notice}"""
    return formatted_content


def create_prompt_from_smt_file(
    smt_file_path: str | Path, char_limit: int = 20000
) -> str:
    """
    Read an SMT file and create a prompt for generating a description.

    Args:
        smt_file_path: Path to the SMT file (.smt2)
        char_limit: Maximum number of characters to include from SMT content (default: 20000).
                   Content exceeding this limit will be truncated.

    Returns:
        Prompt string for the LLM
    """
    # Extract smtlib_path by keeping only the part after "non-incremental/"
    smt_path_str = str(smt_file_path)
    if "non-incremental/" not in smt_path_str:
        raise ValueError(f"Path must contain 'non-incremental/': {smt_file_path}")
    smtlib_path = smt_path_str.split("non-incremental/", 1)[1]

    # Get basic info from JSON
    basic_info_dict, symbol_counts = get_basic_info_from_json(smtlib_path)

    # Get SMT file content (includes truncation notice if truncated)
    smt_content = get_smt_content_from_file(smt_file_path, char_limit)

    # Build basic info string
    basic_info_str = format_basic_info_text(basic_info_dict, symbol_counts)

    # Create prompt
    prompt = f"""Based on the following SMT-LIB instance and its metadata, provide a concise description of what it encodes. 

Limit the description to 4–6 sentences.

Focus on:
- Basic information, such as logic/theory, size, source, etc.
- The main problem structure
- Key constraints or properties being checked
- Any notable characteristics

{smt_content}
{basic_info_str}
"""
    return prompt
