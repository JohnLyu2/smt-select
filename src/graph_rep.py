"""Build a graph representation of an SMT instance from its formula AST using Z3."""

import gc
import logging
import multiprocessing
import os
import signal
import sys
from pathlib import Path

from tqdm import tqdm

from grakel import Graph
from z3 import (
    AstVector,
    Z3_OP_UNINTERPRETED,
    is_app,
    is_bv_value,
    is_false,
    is_int_value,
    is_quantifier,
    is_rational_value,
    is_true,
    is_var,
    parse_smt2_file,
)


def smt_to_graph(smt_path: str | Path) -> dict:
    """
    Parse an SMT-LIB2 file and return a directed graph of the SMT instance.

    The graph represents the formula as a term DAG: each node is a subexpression
    (application, quantifier, variable, or constant), and each edge connects a
    parent to a child subterm (with an optional argument index).

    Parsing is done via Z3's parse_smt2_file.

    Args:
        smt_path: Path to the .smt2 file.

    Returns:
        A dict with:
          - "nodes": dict[int, dict] — node_id -> {"label": str, "sort": str, "type": str}
            (label is the operator name or a short form of the term; sort is
            the Z3 sort string). type: for built-in operator applications the
            exact operator name (e.g. "bvmul", "forall"); "udf" for
            user-defined (uninterpreted) function applications (name in label);
            "const" for numerals/literals; "free_var" for 0-ary symbols;
            "bound_var" for bound variables; "lambda" for lambda expressions.
          - "edges": list[tuple[int, int, int]] — (parent_id, child_id, arg_index).
          - "roots": list[int] — node ids of top-level assertions (roots of the DAG).

    Raises:
        FileNotFoundError: If smt_path does not exist.
        z3.Z3Exception: If the file is not valid SMT-LIB2 or contains unsupported commands.
        ValueError: If a quantifier is not forall, exists, or lambda; or if an
            expression is not quantifier, app, or bound variable.
    """
    path = Path(smt_path)
    if not path.exists():
        raise FileNotFoundError(f"SMT file not found: {smt_path}")

    ast_vector: AstVector = parse_smt2_file(str(path.resolve()))
    nodes: dict[int, dict] = {}
    edges: list[tuple[int, int, int]] = []
    roots: list[int] = []
    _node_counter = 0
    _expr_to_id: dict[int, int] = {}  # z3_expr.get_id() -> our node_id

    def _node_id(expr) -> int:
        nonlocal _node_counter
        key = expr.get_id()  # Z3's stable AST node ID
        if key not in _expr_to_id:
            _expr_to_id[key] = _node_counter
            _node_counter += 1
        return _expr_to_id[key]

    def _add_node(nid: int, label: str, sort: str, node_type: str) -> None:
        nodes[nid] = {"label": label, "sort": sort, "type": node_type}

    def _is_numeral(expr) -> bool:
        """True if expr is a real constant (numeral/literal): BV, Int, Real, Bool."""
        return (
            is_bv_value(expr)
            or is_int_value(expr)
            or is_rational_value(expr)
            or is_true(expr)
            or is_false(expr)
        )

    def _walk(expr) -> int:
        nid = _node_id(expr)
        if nid in nodes:
            return nid
        sort_str = str(expr.sort())
        if is_quantifier(expr):
            if expr.is_lambda():
                op = "lambda"
            elif expr.is_forall():
                op = "forall"
            elif expr.is_exists():
                op = "exists"
            else:
                raise ValueError(f"Unknown quantifier kind: {expr}")
            _add_node(nid, op, sort_str, op)
            body = expr.body()
            cid = _walk(body)
            edges.append((nid, cid, 0))
        elif is_app(expr):
            decl = expr.decl()
            name = decl.name()
            if expr.num_args() == 0:
                if _is_numeral(expr):
                    _add_node(nid, str(expr), sort_str, "const")
                else:
                    _add_node(nid, name, sort_str, "free_var")
            else:
                node_type = "udf" if decl.kind() == Z3_OP_UNINTERPRETED else name
                _add_node(nid, name, sort_str, node_type)
            for i, child in enumerate(expr.children()):
                cid = _walk(child)
                edges.append((nid, cid, i))
        elif is_var(expr):
            _add_node(nid, str(expr), sort_str, "bound_var")
        else:
            raise ValueError(
                f"Unknown expression kind (not quantifier, app, or var): {expr}"
            )
        return nid

    for i in range(len(ast_vector)):
        assertion = ast_vector[i]
        rid = _walk(assertion)
        roots.append(rid)

    return {"nodes": nodes, "edges": edges, "roots": roots}


def smt_graph_to_grakel(graph_dict: dict) -> Graph:
    """
    Convert the internal graph dict from smt_to_graph() to a GraKel Graph.

    Node labels are taken from nodes[nid]["type"] (e.g. "bvmul", "forall",
    "const", "bound_var"). Edges are made undirected by including both
    (u, v) and (v, u); the argument index from the internal format is dropped.

    Args:
        graph_dict: Dict returned by smt_to_graph() with keys "nodes", "edges".

    Returns:
        A grakel.Graph suitable for Weisfeiler-Lehman and other graph kernels.
    """
    nodes = graph_dict["nodes"]
    raw_edges = graph_dict["edges"]
    node_labels = {nid: nodes[nid]["type"] for nid in nodes}
    undirected_edges: list[tuple[int, int]] = []
    for u, v, _ in raw_edges:
        undirected_edges.append((u, v))
        undirected_edges.append((v, u))
    return Graph(undirected_edges, node_labels=node_labels)


def smt_graph_to_gin(graph_dict: dict) -> dict:
    """
    Convert the internal graph dict from smt_to_graph() to a GIN-ready structure.

    Node labels are taken from nodes[nid]["type"] (e.g. "bvmul", "forall",
    "const", "bound_var"). Edges are made undirected by including both
    (u, v) and (v, u); the argument index from the internal format is dropped.

    Args:
        graph_dict: Dict returned by smt_to_graph() with keys "nodes", "edges".

    Returns:
        A dict with:
          - "node_types": list[str] — node_types[i] is the type of node id i
            (nodes are ordered by id 0..n-1).
          - "edges": list[tuple[int, int]] — undirected edges (u, v) and (v, u).
    """
    nodes = graph_dict["nodes"]
    raw_edges = graph_dict["edges"]
    node_ids = sorted(nodes.keys())
    # Remap node ids to 0..n-1 so GIN scatter never sees out-of-bounds indices.
    old_id_to_idx = {old_id: i for i, old_id in enumerate(node_ids)}
    node_types = [nodes[nid]["type"] for nid in node_ids]
    undirected_edges: list[tuple[int, int]] = []
    for u, v, _ in raw_edges:
        ui, vi = old_id_to_idx[u], old_id_to_idx[v]
        undirected_edges.append((ui, vi))
        undirected_edges.append((vi, ui))
    return {"node_types": node_types, "edges": undirected_edges}


def _graph_timeout_handler(_signum: int, _frame: object) -> None:
    signal.alarm(0)
    raise TimeoutError("Graph build timed out")


def _suppress_z3_destructor_noise() -> None:
    """Run GC with stderr suppressed to avoid z3 AstRef.__del__ messages after timeout."""
    devnull = open(os.devnull, "w")
    old = sys.stderr
    try:
        sys.stderr = devnull
        gc.collect()
    finally:
        sys.stderr = old
        devnull.close()


def build_smt_graph_dict_timeout(smt_path: str | Path, timeout_sec: int) -> dict | None:
    """Build raw graph dict (smt_to_graph) with timeout. Returns None on timeout/error."""
    path = Path(smt_path)
    signal.signal(signal.SIGALRM, _graph_timeout_handler)
    signal.alarm(timeout_sec)
    try:
        graph_dict = smt_to_graph(path)
        signal.alarm(0)
        return graph_dict
    except TimeoutError:
        logging.debug("Timeout (%ds) while building graph for %s", timeout_sec, smt_path)
        _suppress_z3_destructor_noise()
        return None
    except RecursionError:
        logging.debug("Recursion limit exceeded while building graph for %s", smt_path)
        _suppress_z3_destructor_noise()
        return None
    except Exception as e:
        if "recursion" in str(e).lower():
            logging.debug("Recursion limit exceeded while building graph for %s", smt_path)
        else:
            logging.debug("Error building graph for %s: %s", smt_path, e)
        _suppress_z3_destructor_noise()
        return None
    finally:
        signal.alarm(0)


def _build_graph_dict_worker(args: tuple[str, int]) -> tuple[str, dict | None]:
    """Top-level worker for process pool: (path, timeout_sec) -> (path, graph_dict | None)."""
    path, timeout_sec = args
    result = build_smt_graph_dict_timeout(path, timeout_sec)
    return (path, result)


def generate_graph_dicts(
    instance_paths: list[str], timeout_sec: int
) -> tuple[dict[str, dict], list[str]]:
    """Build graph dicts for each instance. Returns (path -> graph_dict, failed_paths)."""
    graph_by_path: dict[str, dict] = {}
    failed_list: list[str] = []
    for p in instance_paths:
        g = build_smt_graph_dict_timeout(p, timeout_sec)
        if g is not None:
            graph_by_path[p] = g
        else:
            failed_list.append(p)
            _suppress_z3_destructor_noise()
    logging.info(
        "Graphs: %d built, %d failed (of %d instances)",
        len(graph_by_path),
        len(failed_list),
        len(instance_paths),
    )
    return graph_by_path, failed_list


def generate_graph_dicts_parallel(
    instance_paths: list[str],
    timeout_sec: int,
    n_workers: int,
    result_timeout_buffer: int = 10,
) -> tuple[dict[str, dict], list[str]]:
    """Build graph dicts in parallel. Returns (path -> graph_dict, failed_paths).
    Uses sequential path when n_workers <= 1 or instance_paths is empty.

    Each result is collected with a timeout of (timeout_sec + result_timeout_buffer) so that
    a stuck or dead worker cannot block the main process indefinitely.
    """
    if not instance_paths:
        return {}, []
    if n_workers <= 1:
        return generate_graph_dicts(instance_paths, timeout_sec)
    n_workers = min(n_workers, len(instance_paths))
    args = [(p, timeout_sec) for p in instance_paths]
    graph_by_path: dict[str, dict] = {}
    failed_list: list[str] = []
    get_timeout = timeout_sec + result_timeout_buffer
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(n_workers) as pool:
        async_results = [
            (instance_paths[i], pool.apply_async(_build_graph_dict_worker, (args[i],)))
            for i in range(len(args))
        ]
        for path, ar in tqdm(
            async_results,
            total=len(async_results),
            desc="Building graphs",
            unit="instance",
        ):
            try:
                _, result = ar.get(timeout=get_timeout)
            except (TimeoutError, multiprocessing.TimeoutError):
                logging.warning(
                    "Graph build result timeout (%ds) for %s — worker may be stuck; marking failed",
                    get_timeout,
                    path,
                )
                failed_list.append(path)
                continue
            except Exception as e:
                logging.debug("Graph build failed for %s: %s", path, e)
                failed_list.append(path)
                continue
            if result is not None:
                graph_by_path[path] = result
            else:
                failed_list.append(path)
    logging.info(
        "Graphs: %d built, %d failed (of %d instances)",
        len(graph_by_path),
        len(failed_list),
        len(instance_paths),
    )
    return graph_by_path, failed_list
