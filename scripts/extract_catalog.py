#!/usr/bin/env python3
"""
Find non-incremental benchmarks for a specific logic in a specific evaluation.

Usage:
    # List available evaluations
    python extract_catalog.py --list-evaluations

    # Find benchmarks for a logic - uses latest evaluation by default
    python extract_catalog.py --logic QF_BV

    # Use a specific evaluation ID
    python extract_catalog.py --logic QF_BV --evaluation 20

    # Export to JSON
    python extract_catalog.py --logic QF_BV --output results.json
"""

import sqlite3
import argparse
import json
import sys
from typing import List, Dict, Optional


def get_latest_evaluation_id(db_path: str) -> Optional[int]:
    """Get the ID of the most recent evaluation."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM Evaluations ORDER BY date DESC, id DESC LIMIT 1")
    result = cursor.fetchone()
    conn.close()
    return result[0] if result else None


def find_benchmarks(
    db_path: str,
    logic: str,
    evaluation_id: Optional[int] = None,
    output_file: Optional[str] = None,
) -> List[Dict]:
    """
    Find non-incremental benchmarks for a logic.

    Args:
        db_path: Path to SQLite database
        logic: Logic string to filter by
        evaluation_id: Optional specific evaluation ID to filter by
        output_file: Optional CSV file to write results to

    Returns:
        List of dictionaries with benchmark information
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row  # Enable column access by name
    cursor = conn.cursor()

    # Build the query
    if evaluation_id is not None:
        # Filter by evaluation_id - this ensures we only get benchmarks that were actually evaluated
        query = """
            SELECT DISTINCT
                b.id AS benchmark_id,
                b.name AS benchmark_name,
                b.logic,
                b.category,
                b.size,
                b.queryCount,
                b.description,
                q.id AS query_id,
                q.idx AS query_index,
                ev.id AS evaluation_id,
                ev.name AS evaluation_name,
                ev.date AS evaluation_date,
                f.id AS family_id,
                f.name AS family_name,
                f.folderName AS family_folderName,
                q.assertsCount,
                q.declareFunCount,
                q.declareConstCount,
                q.declareSortCount,
                q.defineFunCount,
                q.defineFunRecCount,
                q.constantFunCount,
                q.defineSortCount,
                q.declareDatatypeCount,
                q.maxTermDepth
            FROM Benchmarks b
            INNER JOIN Queries q ON b.id = q.benchmark
            INNER JOIN Ratings r ON q.id = r.query
            INNER JOIN Evaluations ev ON r.evaluation = ev.id
            LEFT JOIN Families f ON b.family = f.id
            WHERE b.isIncremental = 0
              AND b.logic = ?
              AND ev.id = ?
        """
        params = [logic, evaluation_id]
    else:
        # No evaluation filter - get all benchmarks for the logic in the database
        query = """
            SELECT DISTINCT
                b.id AS benchmark_id,
                b.name AS benchmark_name,
                b.logic,
                b.category,
                b.size,
                b.queryCount,
                b.description,
                q.id AS query_id,
                q.idx AS query_index,
                NULL AS evaluation_id,
                NULL AS evaluation_name,
                NULL AS evaluation_date,
                f.id AS family_id,
                f.name AS family_name,
                f.folderName AS family_folderName,
                q.assertsCount,
                q.declareFunCount,
                q.declareConstCount,
                q.declareSortCount,
                q.defineFunCount,
                q.defineFunRecCount,
                q.constantFunCount,
                q.defineSortCount,
                q.declareDatatypeCount,
                q.maxTermDepth
            FROM Benchmarks b
            INNER JOIN Queries q ON b.id = q.benchmark
            LEFT JOIN Families f ON b.family = f.id
            WHERE b.isIncremental = 0
              AND b.logic = ?
        """
        params = [logic]

    query += " ORDER BY b.id, q.idx"

    cursor.execute(query, params)
    results = cursor.fetchall()

    # Convert to list of dictionaries
    benchmarks = []
    for row in results:
        # Construct SMT-LIB path: {logic}/{folderName}/{name}
        folder_name = row["family_folderName"]
        smtlib_path = None
        if folder_name and row["benchmark_name"]:
            smtlib_path = f"{row['logic']}/{folder_name}/{row['benchmark_name']}"

        # Get symbol counts for this query
        query_id = row["query_id"]
        symbol_query = """
            SELECT s.name, sc.count
            FROM SymbolCounts sc
            JOIN Symbols s ON sc.symbol = s.id
            WHERE sc.query = ?
        """
        cursor.execute(symbol_query, (query_id,))
        symbol_counts = {name: count for name, count in cursor.fetchall()}

        benchmark_dict = {
            "benchmark_id": row["benchmark_id"],
            "benchmark_name": row["benchmark_name"],
            "logic": row["logic"],
            "category": row["category"],
            "size": row["size"],
            "evaluation": row["evaluation_name"],
            "family": row["family_name"],
            "smtlib_path": smtlib_path,
            "asserts_count": row["assertsCount"] or 0,
            "declare_fun_count": row["declareFunCount"] or 0,
            "declare_const_count": row["declareConstCount"] or 0,
            "declare_sort_count": row["declareSortCount"] or 0,
            "define_fun_count": row["defineFunCount"] or 0,
            "define_fun_rec_count": row["defineFunRecCount"] or 0,
            "constant_fun_count": row["constantFunCount"] or 0,
            "define_sort_count": row["defineSortCount"] or 0,
            "declare_datatype_count": row["declareDatatypeCount"] or 0,
            "max_term_depth": row["maxTermDepth"] or 0,
            "symbol_counts": symbol_counts,
        }

        # Only include description if it exists and is not empty
        if row["description"]:
            benchmark_dict["description"] = row["description"]

        benchmarks.append(benchmark_dict)

    conn.close()
    return benchmarks


def print_results(
    benchmarks: List[Dict],
    db_path: str,
    logic: str,
    evaluation_id: Optional[int] = None,
):
    """Print summary to console."""
    if not benchmarks:
        print("No benchmarks found matching the criteria.")
        return

    unique_benchmark_count = len(set(b["benchmark_id"] for b in benchmarks))
    total_query_count = len(benchmarks)

    # Get total count of non-incremental benchmarks in database for this logic
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    if evaluation_id is not None:
        # Get number of unique benchmarks in the evaluation for this logic
        cursor.execute(
            """
            SELECT COUNT(DISTINCT b.id)
            FROM Benchmarks b
            INNER JOIN Queries q ON b.id = q.benchmark
            INNER JOIN Ratings r ON q.id = r.query
            WHERE b.logic = ?
              AND b.isIncremental = 0
              AND r.evaluation = ?
            """,
            (logic, evaluation_id),
        )
        context_benchmark_count = cursor.fetchone()[0]
        context_name = "evaluation"
    else:
        # Get total number of unique non-incremental benchmarks in the database for this logic
        cursor.execute(
            """
            SELECT COUNT(DISTINCT id)
            FROM Benchmarks
            WHERE logic = ?
              AND isIncremental = 0
            """,
            (logic,),
        )
        context_benchmark_count = cursor.fetchone()[0]
        context_name = "database"
    
    conn.close()

    # Get unique families
    unique_families = set()
    for b in benchmarks:
        if b.get("family"):
            unique_families.add(b["family"])

    # Count benchmarks with and without descriptions
    unique_benchmarks_with_desc = set()
    unique_benchmarks_without_desc = set()
    for b in benchmarks:
        bench_id = b["benchmark_id"]
        if "description" in b and b["description"]:
            unique_benchmarks_with_desc.add(bench_id)
        else:
            unique_benchmarks_without_desc.add(bench_id)

    print("\nSummary:")
    print(
        f"  Found {unique_benchmark_count} unique benchmark(s) ({total_query_count} total queries)"
    )
    print(f"  Out of {context_benchmark_count} benchmark(s) in {context_name}")
    print(f"  Across {len(unique_families)} families")
    print(f"  With description: {len(unique_benchmarks_with_desc)} benchmark(s)")
    print(f"  Without description: {len(unique_benchmarks_without_desc)} benchmark(s)")

    # Show a few example benchmarks
    if benchmarks:
        print("\nExample benchmarks (first 5):")
        unique_benchmarks = {}
        for b in benchmarks:
            bench_id = b["benchmark_id"]
            if bench_id not in unique_benchmarks:
                unique_benchmarks[bench_id] = {
                    "name": b["benchmark_name"],
                    "family_name": b.get("family"),
                    "smtlib_path": b.get("smtlib_path"),
                }

        displayed = 0
        for bench_id, info in sorted(unique_benchmarks.items()):
            if displayed >= 5:
                break
            smtlib_path = info.get("smtlib_path") or "N/A"
            print(f"  {bench_id}: {smtlib_path}")
            displayed += 1

        if len(unique_benchmarks) > 5:
            print(f"  ... and {len(unique_benchmarks) - 5} more benchmarks")


def write_json(benchmarks: List[Dict], output_file: str):
    """Write results to JSON file."""
    if not benchmarks:
        print("No benchmarks to write.")
        return

    # Prepare JSON data - keep all fields
    json_data = []
    for benchmark in benchmarks:
        json_benchmark = dict(benchmark)
        # Keep family_name as is for JSON (more descriptive)
        json_data.append(json_benchmark)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)

    print(f"\nResults written to {output_file}")


def list_available_logics(db_path: str):
    """List all available logics in the database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT DISTINCT logic FROM Benchmarks WHERE isIncremental = 0 ORDER BY logic"
    )
    logics = [row[0] for row in cursor.fetchall()]

    print("Available non-incremental logics:")
    for logic in logics:
        cursor.execute(
            "SELECT COUNT(*) FROM Benchmarks WHERE logic = ? AND isIncremental = 0",
            (logic,),
        )
        count = cursor.fetchone()[0]
        print(f"  {logic} ({count} benchmarks)")
    conn.close()


def list_available_evaluations(db_path: str):
    """List all available evaluations in the database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT id, name, date, link, 
               (SELECT COUNT(*) FROM Ratings WHERE evaluation = Evaluations.id) as rating_count
        FROM Evaluations 
        ORDER BY date DESC, id DESC
        """
    )
    evaluations = cursor.fetchall()

    print("Available evaluations:")
    print(f"{'ID':<6} {'Date':<12} {'Name':<40} {'Ratings':<10}")
    print("-" * 80)
    for ev_id, name, date, link, rating_count in evaluations:
        name_display = (
            (name[:37] + "...") if name and len(name) > 40 else (name or "N/A")
        )
        date_display = date or "N/A"
        print(f"{ev_id:<6} {date_display:<12} {name_display:<40} {rating_count:<10}")
    conn.close()


def main():
    parser = argparse.ArgumentParser(
        description="Find non-incremental benchmarks for a logic in a specific evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--db",
        default="smtlib2025.sqlite",
        help="Path to SQLite database (default: smtlib2025.sqlite)",
    )
    parser.add_argument(
        "--logic", help="Logic string to filter by (e.g., QF_BV, QF_LIA)"
    )
    parser.add_argument(
        "--evaluation",
        type=int,
        help="Evaluation ID to filter by. If not specified, uses the most recent evaluation by default.",
    )
    parser.add_argument(
        "--latest-evaluation",
        action="store_true",
        help="Explicitly use the most recent evaluation (this is the default behavior if --evaluation is not specified)",
    )
    parser.add_argument(
        "--all-benchmarks",
        action="store_true",
        help="Find benchmarks across all evaluations/database (overrides --evaluation)",
    )
    parser.add_argument("--output", "-o", help="Optional: Output JSON file path")
    parser.add_argument(
        "--list-logics",
        action="store_true",
        help="List all available non-incremental logics and exit",
    )
    parser.add_argument(
        "--list-evaluations",
        action="store_true",
        help="List all available evaluations and exit",
    )

    args = parser.parse_args()

    if args.list_logics:
        list_available_logics(args.db)
        return

    if args.list_evaluations:
        list_available_evaluations(args.db)
        return

    if not args.logic:
        parser.error(
            "--logic is required (unless using --list-logics or --list-evaluations)"
        )

    # Determine evaluation ID
    evaluation_id = None
    if not args.all_benchmarks:
        evaluation_id = args.evaluation
        if evaluation_id is None or args.latest_evaluation:
            evaluation_id = get_latest_evaluation_id(args.db)
            if evaluation_id is None:
                parser.error("No evaluations found in database")
            if args.latest_evaluation or args.evaluation is None:
                print(f"Using most recent evaluation (ID: {evaluation_id})")

    # Get evaluation info for display
    eval_name = "All Benchmarks"
    eval_date = "N/A"
    if evaluation_id is not None:
        conn = sqlite3.connect(args.db)
        cursor = conn.cursor()
        cursor.execute("SELECT name, date FROM Evaluations WHERE id = ?", (evaluation_id,))
        eval_info = cursor.fetchone()
        eval_name = eval_info[0] if eval_info else f"ID {evaluation_id}"
        eval_date = eval_info[1] if eval_info and eval_info[1] else "N/A"
        
        # Get number of unique benchmarks in the evaluation for this logic
        cursor.execute(
            """
            SELECT COUNT(DISTINCT b.id)
            FROM Benchmarks b
            INNER JOIN Queries q ON b.id = q.benchmark
            INNER JOIN Ratings r ON q.id = r.query
            WHERE b.logic = ?
              AND b.isIncremental = 0
              AND r.evaluation = ?
            """,
            (args.logic, evaluation_id),
        )
        context_benchmark_count = cursor.fetchone()[0]
        conn.close()
    else:
        # Get total count of non-incremental benchmarks for this logic
        conn = sqlite3.connect(args.db)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT COUNT(*) FROM Benchmarks WHERE logic = ? AND isIncremental = 0",
            (args.logic,),
        )
        context_benchmark_count = cursor.fetchone()[0]
        conn.close()

    print("Searching for non-incremental benchmarks:")
    print(f"  Logic: {args.logic}")
    if evaluation_id is not None:
        print(f"  Evaluation benchmark size: {context_benchmark_count}")
        print(f"  Evaluation: {eval_name} (ID: {evaluation_id}, Date: {eval_date})")
    else:
        print(f"  Total database benchmarks for logic: {context_benchmark_count}")
        print(f"  Scope: {eval_name}")
    print()

    try:
        benchmarks = find_benchmarks(
            args.db,
            args.logic,
            evaluation_id,
            args.output,
        )

        print_results(benchmarks, args.db, args.logic, evaluation_id)

        if args.output:
            write_json(benchmarks, args.output)

    except sqlite3.Error as e:
        print(f"Database error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
