import numpy as np

from lineagetree import LineageTree

from lineagetree.approximation import (
    FullTree,
    SimpleTreeTimed,
    SimpleTreeGeneral,
    DownsampledTree,
    ResampledTree,
    delta_difference,
    delta_binary,
    delta_normalized_difference,
)

## Test lineage trees:


def create_chain(tree, length, current_id, start=None):
    id_ = current_id
    if start is not None:
        tree.setdefault(start, []).append(id_)
    for _ in range(length - 1):
        tree[id_] = [id_ + 1]
        id_ += 1

    return id_, id_ + 1


tree1 = {}
id_ = 0
end_chain_1, id_ = create_chain(tree1, 50, id_)
end_chain_2, id_ = create_chain(tree1, 30, id_, end_chain_1)
end_chain_3, id_ = create_chain(tree1, 40, id_, end_chain_1)
_, id_ = create_chain(tree1, 20, id_, end_chain_2)
_, id_ = create_chain(tree1, 20, id_, end_chain_2)
scalar_property = {n: 1 for n in range(id_)}
vector_property = {n: np.array([1, 1]) for n in range(id_)}

lT1 = LineageTree(
    successor=tree1,
    scalar_property=scalar_property,
    vector_property=vector_property,
)

tree2 = {}
id_ = 0
end_chain_1, id_ = create_chain(tree2, 40, id_)
end_chain_2, id_ = create_chain(tree2, 30, id_, end_chain_1)
end_chain_3, id_ = create_chain(tree2, 40, id_, end_chain_1)
_, id_ = create_chain(tree2, 20, id_, end_chain_2)
_, id_ = create_chain(tree2, 20, id_, end_chain_2)
scalar_property = {n: 2 for n in range(id_)}
vector_property = {n: np.array([2, 2]) for n in range(id_)}

lT2 = LineageTree(
    successor=tree2,
    scalar_property=scalar_property,
    vector_property=vector_property,
)

tree3 = {}
id_ = 0
end_chain_1, id_ = create_chain(tree3, 50, id_)
end_chain_2, id_ = create_chain(tree3, 30, id_, end_chain_1)
end_chain_3, id_ = create_chain(tree3, 40, id_, end_chain_1)
scalar_property = {n: 3 for n in range(id_)}
vector_property = {n: np.array([3, 3]) for n in range(id_)}

lT3 = LineageTree(
    successor=tree3,
    scalar_property=scalar_property,
    vector_property=vector_property,
)


def test_FullTree():
    fulltree = FullTree()

    approx_1 = fulltree.build_approximated_tree(lT1, 0)
    approx_2 = fulltree.build_approximated_tree(lT2, 0)
    approx_3 = fulltree.build_approximated_tree(lT3, 0)

    dist12 = fulltree.compute_uted_distance(approx_1, approx_2)
    assert dist12 == 10

    dist13 = fulltree.compute_uted_distance(approx_1, approx_3)
    assert dist13 == 40

    dist23 = fulltree.compute_uted_distance(approx_2, approx_3)
    assert dist23 == 50


def test_FullTree_delta():
    fulltree_delta = FullTree(delta=delta_difference)

    approx_1 = fulltree_delta.build_approximated_tree(
        lT1, 0, property_dictionary=lT1.scalar_property
    )
    approx_2 = fulltree_delta.build_approximated_tree(
        lT2, 0, property_dictionary=lT2.scalar_property
    )
    approx_3 = fulltree_delta.build_approximated_tree(
        lT3, 0, property_dictionary=lT3.scalar_property
    )

    dist12 = fulltree_delta.compute_uted_distance(approx_1, approx_2)
    assert dist12 == 160

    dist13 = fulltree_delta.compute_uted_distance(approx_1, approx_3)
    assert dist13 == 280

    dist23 = fulltree_delta.compute_uted_distance(approx_2, approx_3)
    assert dist23 == 220


def test_SimpleTreeTimed():
    simple_tree_timed = SimpleTreeTimed(delta=delta_difference)

    approx_1 = simple_tree_timed.build_approximated_tree(lT1, 0)
    approx_2 = simple_tree_timed.build_approximated_tree(lT2, 0)
    approx_3 = simple_tree_timed.build_approximated_tree(lT3, 0)

    dist12 = simple_tree_timed.compute_uted_distance(approx_1, approx_2)
    assert dist12 == 10

    dist13 = simple_tree_timed.compute_uted_distance(approx_1, approx_3)
    assert dist13 == 40

    dist23 = simple_tree_timed.compute_uted_distance(approx_2, approx_3)
    assert dist23 == 50


def test_SimpleTreeTimed_normed():
    simple_tree_timed_normed = SimpleTreeTimed()

    approx_1 = simple_tree_timed_normed.build_approximated_tree(lT1, 0)
    approx_2 = simple_tree_timed_normed.build_approximated_tree(lT2, 0)
    approx_3 = simple_tree_timed_normed.build_approximated_tree(lT3, 0)

    dist12 = simple_tree_timed_normed.compute_uted_distance(approx_1, approx_2)
    assert dist12 == 0.2

    dist13 = simple_tree_timed_normed.compute_uted_distance(approx_1, approx_3)
    assert dist13 == 2

    dist23 = simple_tree_timed_normed.compute_uted_distance(approx_2, approx_3)
    assert dist23 == 2.2


def test_SimpleTreeGeneral_scalar():
    simple_tree_general = SimpleTreeGeneral()

    approx_1 = simple_tree_general.build_approximated_tree(
        lT1, 0, properties=lT1.scalar_property
    )
    approx_2 = simple_tree_general.build_approximated_tree(
        lT2, 0, properties=lT2.scalar_property
    )
    approx_3 = simple_tree_general.build_approximated_tree(
        lT3, 0, properties=lT3.scalar_property
    )

    dist12 = simple_tree_general.compute_uted_distance(approx_1, approx_2)
    assert dist12 == 5

    dist13 = simple_tree_general.compute_uted_distance(approx_1, approx_3)
    assert dist13 == 8

    dist23 = simple_tree_general.compute_uted_distance(approx_2, approx_3)
    assert dist23 == 7


def test_SimpleTreeGeneral_vector():
    simple_tree_general = SimpleTreeGeneral()
    approx_1 = simple_tree_general.build_approximated_tree(
        lT1, 0, properties=lT1.vector_property
    )
    approx_2 = simple_tree_general.build_approximated_tree(
        lT2, 0, properties=lT2.vector_property
    )
    approx_3 = simple_tree_general.build_approximated_tree(
        lT3, 0, properties=lT3.vector_property
    )

    dist12 = simple_tree_general.compute_uted_distance(approx_1, approx_2)
    assert np.isclose(dist12, 5 * 2**0.5)

    dist13 = simple_tree_general.compute_uted_distance(approx_1, approx_3)
    assert np.isclose(dist13, 8**0.5 * 3 + 2**0.5 * 2)

    dist23 = simple_tree_general.compute_uted_distance(approx_2, approx_3)
    assert np.isclose(dist23, 2**0.5 * 3 + 8**0.5 * 2)


def test_DownsampledTree():
    downsampled_tree = DownsampledTree()

    approx_1 = downsampled_tree.build_approximated_tree(lT1, 0, downsample=2)

    approx_2 = downsampled_tree.build_approximated_tree(lT2, 0, downsample=2)

    approx_3 = downsampled_tree.build_approximated_tree(lT3, 0, downsample=2)

    dist12 = downsampled_tree.compute_uted_distance(approx_1, approx_2)
    assert dist12 == 5

    dist13 = downsampled_tree.compute_uted_distance(approx_1, approx_3)
    assert dist13 == 20

    dist23 = downsampled_tree.compute_uted_distance(approx_2, approx_3)
    assert dist23 == 25


def test_ResampledTree_simple():
    resampled_tree = ResampledTree(
        target_time_resolution=5, delta=delta_binary
    )

    approx_1 = resampled_tree.build_approximated_tree(
        lT1, 0, time_resolution=1
    )

    approx_2 = resampled_tree.build_approximated_tree(
        lT2, 0, time_resolution=2
    )

    approx_3 = resampled_tree.build_approximated_tree(
        lT3, 0, time_resolution=10
    )

    dist12 = resampled_tree.compute_uted_distance(approx_1, approx_2)
    assert dist12 == 28

    # I don't get that score
    dist13 = resampled_tree.compute_uted_distance(approx_1, approx_3)
    assert dist13 == 216

    # I don't get that score
    dist23 = resampled_tree.compute_uted_distance(approx_2, approx_3)
    assert dist23 == 196


def test_ResampledTree():
    resampled_tree = ResampledTree(target_time_resolution=5)

    approx_1 = resampled_tree.build_approximated_tree(
        lT1, 0, time_resolution=1, sampling_property=lT1.scalar_property
    )

    approx_2 = resampled_tree.build_approximated_tree(
        lT2, 0, time_resolution=2, sampling_property=lT2.scalar_property
    )

    approx_3 = resampled_tree.build_approximated_tree(
        lT3, 0, time_resolution=10, sampling_property=lT3.scalar_property
    )

    # I didn't verify these scores :/
    dist12 = resampled_tree.compute_uted_distance(approx_1, approx_2)
    assert dist12 == 88

    dist13 = resampled_tree.compute_uted_distance(approx_1, approx_3)
    assert dist13 == 696

    dist23 = resampled_tree.compute_uted_distance(approx_2, approx_3)
    assert dist23 == 632


def test_norm():
    fulltree = FullTree()

    approx_1 = fulltree.build_approximated_tree(lT1, 0)
    approx_2 = fulltree.build_approximated_tree(lT2, 0)

    norm12 = fulltree.get_norm(approx_1, approx_2)
    assert norm12 == 160
    norm12 = fulltree.get_norm(approx_1, approx_2, norm_type="sum")
    assert norm12 == 310

    simple_tree = SimpleTreeGeneral(delta=delta_normalized_difference)

    approx_1 = simple_tree.build_approximated_tree(
        lT1, 0, properties=lT1.scalar_property
    )
    approx_3 = simple_tree.build_approximated_tree(
        lT3, 0, properties=lT3.scalar_property
    )

    norm13 = simple_tree.get_norm(approx_1, approx_3)
    assert norm13 == 5
