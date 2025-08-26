from __future__ import annotations

import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.spatial import distance
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..lineage_tree import LineageTree


def __calculate_diag_line(dist_mat: np.ndarray) -> tuple[float, float]:
    """
    Calculate the line that centers the band w.

    Parameters
    ----------
    dist_mat : np.ndarray
        distance matrix obtained by the function calculate_dtw

    Returns
    -------
    float
        The slope of the curve
    float
        The intercept of the curve
    """
    i, j = dist_mat.shape
    x1 = max(0, i - j) / 2
    x2 = (i + min(i, j)) / 2
    y1 = max(0, j - i) / 2
    y2 = (j + min(i, j)) / 2
    slope = (y1 - y2) / (x1 - x2)
    intercept = y1 - slope * x1
    return slope, intercept


# Reference: https://github.com/kamperh/lecture_dtw_notebook/blob/main/dtw.ipynb
def __dp(
    dist_mat: np.ndarray,
    start_d: int = 0,
    back_d: int = 0,
    fast: bool = False,
    w: int = 0,
    centered_band: bool = True,
) -> tuple[list[int], np.ndarray, float]:
    """
    Find DTW minimum cost between two series using dynamic programming.

    Parameters
    ----------
    dist_mat : np.ndarray
        distance matrix obtained by the function calculate_dtw
    start_d : int, default=0
        start delay
    back_d : int, default=0
        end delay
    fast : bool, default=False
        if `True`, the algorithm will use a faster version but might not find the optimal alignment
    w : int, default=0
        window constrain
    centered_band : bool, default=True
        if `True`, the band will be centered around the diagonal

    Returns
    -------
    tuple of tuples of int
        Aligment path
    np.ndarray
        cost matrix
    float
        optimal cost
    """
    N, M = dist_mat.shape
    w_limit = max(w, abs(N - M))  # Calculate the Sakoe-Chiba band width

    if centered_band:
        slope, intercept = __calculate_diag_line(dist_mat)
        square_root = np.sqrt((slope**2) + 1)

    # Initialize the cost matrix
    cost_mat = np.full((N + 1, M + 1), np.inf)
    cost_mat[0, 0] = 0

    # Fill the cost matrix while keeping traceback information
    traceback_mat = np.zeros((N, M))

    cost_mat[: start_d + 1, 0] = 0
    cost_mat[0, : start_d + 1] = 0

    cost_mat[N - back_d :, M] = 0
    cost_mat[N, M - back_d :] = 0

    for i in range(N):
        for j in range(M):
            if fast and not centered_band:
                condition = abs(i - j) <= w_limit
            elif fast:
                condition = (
                    abs(slope * i - j + intercept) / square_root <= w_limit
                )
            else:
                condition = True

            if condition:
                penalty = [
                    cost_mat[i, j],  # match (0)
                    cost_mat[i, j + 1],  # insertion (1)
                    cost_mat[i + 1, j],  # deletion (2)
                ]
                i_penalty = np.argmin(penalty)
                cost_mat[i + 1, j + 1] = dist_mat[i, j] + penalty[i_penalty]
                traceback_mat[i, j] = i_penalty

    min_index1 = np.argmin(cost_mat[N - back_d :, M])
    min_index2 = np.argmin(cost_mat[N, M - back_d :])

    if (
        cost_mat[N, M - back_d + min_index2]
        < cost_mat[N - back_d + min_index1, M]
    ):
        i = N - 1
        j = M - back_d + min_index2 - 1
        final_cost = cost_mat[i + 1, j + 1]
    else:
        i = N - back_d + min_index1 - 1
        j = M - 1
        final_cost = cost_mat[i + 1, j + 1]

    path = [(i, j)]

    while (
        start_d != 0 and ((start_d < i and j > 0) or (i > 0 and start_d < j))
    ) or (start_d == 0 and (i > 0 or j > 0)):
        tb_type = traceback_mat[i, j]
        if tb_type == 0:
            # Match
            i -= 1
            j -= 1
        elif tb_type == 1:
            # Insertion
            i -= 1
        elif tb_type == 2:
            # Deletion
            j -= 1

        path.append((i, j))

    # Strip infinity edges from cost_mat before returning
    cost_mat = cost_mat[1:, 1:]
    return path[::-1], cost_mat, final_cost


# Reference: https://github.com/nghiaho12/rigid_transform_3D
def __rigid_transform_3D(A, B):
    assert A.shape == B.shape

    num_rows, num_cols = A.shape
    if num_rows != 3:
        raise Exception(f"matrix A is not 3xN, it is {num_rows}x{num_cols}")

    num_rows, num_cols = B.shape
    if num_rows != 3:
        raise Exception(f"matrix B is not 3xN, it is {num_rows}x{num_cols}")

    # find mean column wise
    centroid_A = np.mean(A, axis=1)
    centroid_B = np.mean(B, axis=1)

    # ensure centroids are 3x1
    centroid_A = centroid_A.reshape(-1, 1)
    centroid_B = centroid_B.reshape(-1, 1)

    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    H = Am @ np.transpose(Bm)

    # find rotation
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # special reflection case
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    t = -R @ centroid_A + centroid_B

    return R, t


def __interpolate(
    lT: LineageTree, chain1: list, chain2: list, threshold: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Interpolate two series that have different lengths

    Parameters
    ----------
    lT : LineageTree
        The LineageTree instance.
    chain1 : list of int
        list of nodes of the first chain to compare
    chain2 : list of int
        list of nodes of the second chain to compare
    threshold : int
        set a maximum number of points a chain can have

    Returns
    -------
    list of np.ndarray
        `x`, `y`, `z` postions for `chain1`
    list of np.ndarray
        `x`, `y`, `z` postions for `chain2`
    """
    inter1_pos = []
    inter2_pos = []

    chain1_pos = np.array([lT.pos[c_id] for c_id in chain1])
    chain2_pos = np.array([lT.pos[c_id] for c_id in chain2])

    # Both chains have the same length and size below the threshold - nothing is done
    if len(chain1) == len(chain2) and (
        len(chain1) <= threshold or len(chain2) <= threshold
    ):
        return chain1_pos, chain2_pos
    # Both chains have the same length but one or more sizes are above the threshold
    elif len(chain1) > threshold or len(chain2) > threshold:
        sampling = threshold
    # chains have different lengths and the sizes are below the threshold
    else:
        sampling = max(len(chain1), len(chain2))

    for pos in range(3):
        chain1_interp = InterpolatedUnivariateSpline(
            np.linspace(0, 1, len(chain1_pos[:, pos])),
            chain1_pos[:, pos],
            k=1,
        )
        inter1_pos.append(chain1_interp(np.linspace(0, 1, sampling)))

        chain2_interp = InterpolatedUnivariateSpline(
            np.linspace(0, 1, len(chain2_pos[:, pos])),
            chain2_pos[:, pos],
            k=1,
        )
        inter2_pos.append(chain2_interp(np.linspace(0, 1, sampling)))

    return np.column_stack(inter1_pos), np.column_stack(inter2_pos)


def calculate_dtw(
    lT: LineageTree,
    nodes1: int,
    nodes2: int,
    threshold: int = 1000,
    regist: bool = True,
    start_d: int = 0,
    back_d: int = 0,
    fast: bool = False,
    w: int = 0,
    centered_band: bool = True,
    cost_mat_p: bool = False,
) -> (
    tuple[float, tuple, np.ndarray, np.ndarray, np.ndarray]
    | tuple[float, tuple]
):
    """
    Calculate DTW distance between two chains

    Parameters
    ----------
    lT : LineageTree
        The LineageTree instance.
    nodes1 : int
        node to compare distance
    nodes2 : int
        node to compare distance
    threshold : int, default=1000
        set a maximum number of points a chain can have
    regist : bool, default=True
        Rotate and translate trajectories
    start_d : int, default=0
        start delay
    back_d : int, default=0
        end delay
    fast : bool, default=False
        if `True`, the algorithm will use a faster version but might not find the optimal alignment
    w : int, default=0
        window size
    centered_band : bool, default=True
        when running the fast algorithm, `True` if the windown is centered
    cost_mat_p : bool, default=False
        True if print the not normalized cost matrix

    Returns
    -------
    float
        DTW distance
    tuple of tuples
        Aligment path
    matrix
        Cost matrix
    list of lists
        rotated and translated trajectories positions
    list of lists
        rotated and translated trajectories positions
    """
    nodes1_chain = lT.get_chain_of_node(nodes1)
    nodes2_chain = lT.get_chain_of_node(nodes2)

    interp_chain1, interp_chain2 = __interpolate(
        lT, nodes1_chain, nodes2_chain, threshold
    )

    pos_chain1 = np.array([lT.pos[c_id] for c_id in nodes1_chain])
    pos_chain2 = np.array([lT.pos[c_id] for c_id in nodes2_chain])

    if regist:
        R, t = __rigid_transform_3D(
            np.transpose(interp_chain1), np.transpose(interp_chain2)
        )
        pos_chain1 = np.transpose(np.dot(R, pos_chain1.T) + t)

    dist_mat = distance.cdist(pos_chain1, pos_chain2, "euclidean")

    path, cost_mat, final_cost = __dp(
        dist_mat,
        start_d,
        back_d,
        w=w,
        fast=fast,
        centered_band=centered_band,
    )
    cost = final_cost / len(path)

    if cost_mat_p:
        return cost, path, cost_mat, pos_chain1, pos_chain2
    else:
        return cost, path
