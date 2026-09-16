import warnings

from ..lineage_tree import LineageTree

try:
    import motile
except ImportError:
    warnings.warn(
        "No motile installed therefore you will not be able to produce links with motile.",
        stacklevel=2,
    )


def to_motile(
    lT: LineageTree, crop: int | None = None, max_dist=200, max_skip_frames=1
):
    """Build a motile candidate graph from a lineage tree.

    Parameters
    ----------
    lT : LineageTree
        The lineage tree to convert.
    crop : int, optional
        The last time point (exclusive) to include. If None, ``lT.t_e`` is used.
    max_dist : float, default=200
        Maximum spatial distance allowed for candidate edges.
    max_skip_frames : int, default=1
        Maximum number of time points an edge is allowed to skip.

    Returns
    -------
    networkx.DiGraph
        A directed graph with candidate edges added by motile.

    Raises
    ------
    Warning
        If networkx is not installed.
    """
    try:
        import networkx as nx
    except ImportError:
        raise Warning("Please install networkx")  # noqa: B904

    fmt = nx.DiGraph()
    if not crop:
        crop = lT.t_e
    for time in range(crop):
        for time_node in lT.time_nodes[time]:
            fmt.add_node(
                time_node,
                t=lT.time[time_node],
                pos=lT.pos[time_node],
                score=1,
            )

    motile.add_cand_edges(fmt, max_dist, max_skip_frames=max_skip_frames)

    return fmt
