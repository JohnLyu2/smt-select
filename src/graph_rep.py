"""Build a graph representation of an SMT instance from its formula AST using Z3."""

from pathlib import Path

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
