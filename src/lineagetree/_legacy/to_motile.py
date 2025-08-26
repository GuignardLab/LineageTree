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
