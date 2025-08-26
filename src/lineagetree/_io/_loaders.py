from __future__ import annotations

import csv
import os
import pickle as pkl
import struct
import xml.etree.ElementTree as ET
from pathlib import Path
from warnings import warn

import numpy as np
from micromastodonreader import MicroMastodonReader

from ..lineage_tree import LineageTree

IMPLICIT_L_T = {
    "AB": "P0",
    "P1": "P0",
    "EMS": "P1",
    "P2": "P1",
    "MS": "EMS",
    "E": "EMS",
    "C": "P2",
    "P3": "P2",
    "D": "P3",
    "P4": "P3",
    "Z2": "P4",
    "Z3": "P4",
}

ASTEC_KEYDICTIONARY = {
    "cell_lineage": [
        "lineage_tree",
        "lin_tree",
        "Lineage tree",
        "cell_lineage",
    ],
    "cell_h_min": ["cell_h_min", "h_mins_information"],
    "cell_volume": [
        "cell_volume",
        "volumes_information",
        "volumes information",
        "vol",
    ],
    "cell_surface": ["cell_surface", "cell surface"],
    "cell_compactness": [
        "cell_compactness",
        "Cell Compactness",
        "compacity",
        "cell_sphericity",
    ],
    "cell_sigma": ["cell_sigma", "sigmas_information", "sigmas"],
    "cell_labels_in_time": [
        "cell_labels_in_time",
        "Cells labels in time",
        "time_labels",
    ],
    "cell_barycenter": [
        "cell_barycenter",
        "Barycenters",
        "barycenters",
    ],
    "cell_fate": ["cell_fate", "Fate"],
    "cell_fate_2": ["cell_fate_2", "Fate2"],
    "cell_fate_3": ["cell_fate_3", "Fate3"],
    "cell_fate_4": ["cell_fate_4", "Fate4"],
    "all_cells": [
        "all_cells",
        "All Cells",
        "All_Cells",
        "all cells",
        "tot_cells",
    ],
    "cell_principal_values": [
        "cell_principal_values",
        "Principal values",
    ],
    "cell_name": ["cell_name", "Names", "names", "cell_names"],
    "cell_contact_surface": [
        "cell_contact_surface",
        "cell_cell_contact_information",
    ],
    "cell_history": [
        "cell_history",
        "Cells history",
        "cell_life",
        "life",
    ],
    "cell_principal_vectors": [
        "cell_principal_vectors",
        "Principal vectors",
    ],
    "cell_naming_score": ["cell_naming_score", "Scores", "scores"],
    "problematic_cells": ["problematic_cells"],
    "unknown_key": ["unknown_key"],
}


def read_from_csv(
    file_path: str,
    z_mult: float,
    link: int = 1,
    delim: str = ",",
    name: None | str = None,
) -> LineageTree:
    """Read a lineage tree from a csv file with the following format:
    id, time, z, y, x, id, pred_id, lin_id

    Parameters
    ----------
        file_path : str
            path to the csv file
        z_mult : float
            aspect ratio
        link : int
            1 if the csv file is ordered by id, 2 if ordered by pred_id
        delim : str, default=","
            delimiter used in the csv file
        name : None or str, optional
           The name attribute of the LineageTree file. If given a non-empty string, the value of the attribute
           will be the name attribute, otherwise the name will be the stem of the file path.

    Returns
    -------
        LineageTree
            lineage tree
    """
    with open(file_path) as f:
        lines = f.readlines()
        f.close()
    successor = {}
    pos = {}
    time = {}
    lines_to_int = []
    corres = {}
    for line in lines:
        lines_to_int += [[eval(v.strip()) for v in line.split(delim)]]
    lines_to_int = np.array(lines_to_int)
    if link == 2:
        lines_to_int = lines_to_int[np.argsort(lines_to_int[:, 0])]
    else:
        lines_to_int = lines_to_int[np.argsort(lines_to_int[:, 1])]
    for unique_id, line in enumerate(lines_to_int):
        if link == 1:
            id_, t, z, y, x, pred, lin_id = line
        elif link == 2:
            t, z, y, x, id_, pred, lin_id = line
        else:
            id_, t, z, y, x, *_ = line
            pred = None
        t = int(t)
        positions = np.array([x, y, z])
        C = unique_id
        corres[id_] = C
        positions[-1] = positions[-1] * z_mult
        if pred in corres:
            M = corres[pred]
            successor.setdefault(M, []).append(C)
        pos[C] = positions
        time[C] = t
    if not name:
        tmp_name = Path(file_path).stem
        if name == "":
            warn(f"Name set to default {tmp_name}", stacklevel=2)
        name = tmp_name
    return LineageTree(successor=successor, time=time, pos=pos, name=name)


def _read_from_ASTEC_xml(file_path: str):
    def _set_dictionary_value(root):
        if len(root) == 0:
            if root.text is None:
                return None
            else:
                return eval(root.text)
        else:
            dictionary = {}
            for child in root:
                key = child.tag
                if child.tag == "cell":
                    key = int(child.attrib["cell-id"])
                dictionary[key] = _set_dictionary_value(child)
        return dictionary

    tree = ET.parse(file_path)
    root = tree.getroot()
    dictionary = {}

    for k in ASTEC_KEYDICTIONARY:
        if root.tag == k:
            dictionary[str(root.tag)] = _set_dictionary_value(root)
            break
    else:
        for child in root:
            value = _set_dictionary_value(child)
            if value is not None:
                dictionary[str(child.tag)] = value
    return dictionary


def _read_from_ASTEC_pkl(file_path: str, eigen: bool = False):
    with open(file_path, "rb") as f:
        tmp_data = pkl.load(f, encoding="latin1")
        f.close()
    new_ref = {}
    for k, v in ASTEC_KEYDICTIONARY.items():
        for key in v:
            new_ref[key] = k
    new_dict = {}

    for k, v in tmp_data.items():
        if k in new_ref:
            new_dict[new_ref[k]] = v
        else:
            new_dict[k] = v
    return new_dict


def read_from_ASTEC(
    file_path: str, eigen: bool = False, name: None | str = None
) -> LineageTree:
    """
    Read an `xml` or `pkl` file produced by the ASTEC algorithm.

    Parameters
    ----------
    file_path : str
        path to an output generated by ASTEC
    eigen : bool, default=False
        whether or not to read the eigen values, default False
    name : None or str, optional
        The name attribute of the LineageTree file. If given a non-empty string, the value of the attribute
        will be the name attribute, otherwise the name will be the stem of the file path.

    Returns
    -------
    LineageTree
        lineage tree
    """

    if os.path.splitext(file_path)[-1] == ".xml":
        tmp_data = _read_from_ASTEC_xml(file_path)
    else:
        tmp_data = _read_from_ASTEC_pkl(file_path, eigen)

    # make sure these are all named liked they are in tmp_data (or change dictionary above)
    properties = {}
    if "cell_volume" in tmp_data:
        properties["volume"] = {}
    if "cell_fate" in tmp_data:
        properties["fate"] = {}
    if "cell_barycenter" in tmp_data:
        pos = {}
    if "cell_name" in tmp_data:
        properties["label"] = {}
    lT2pkl = {}
    pkl2lT = {}
    image_label = {}

    lt = tmp_data["cell_lineage"]

    if "cell_contact_surface" in tmp_data:
        properties["contact"] = {}
        do_surf = True
        surfaces = tmp_data["cell_contact_surface"]
    else:
        do_surf = False

    inv = {vi: [c] for c, v in lt.items() for vi in v}
    nodes = set(lt).union(inv)

    unique_id = 0
    time = {}

    for unique_id, n in enumerate(nodes):
        t = n // 10**4
        image_label[unique_id] = n % 10**4
        lT2pkl[unique_id] = n
        pkl2lT[n] = unique_id
        time[unique_id] = t
        if "cell_volume" in tmp_data:
            properties["volume"][unique_id] = tmp_data["cell_volume"].get(
                n, 0.0
            )
        if "cell_fate" in tmp_data:
            properties["fate"][unique_id] = tmp_data["cell_fate"].get(n, "")
        if "cell_barycenter" in tmp_data:
            pos[unique_id] = tmp_data["cell_barycenter"].get(n, np.zeros(3))
        if "cell_name" in tmp_data:
            properties["label"][unique_id] = tmp_data["cell_name"].get(n, "")

    if do_surf:
        for c in nodes:
            if c in surfaces and c in pkl2lT:
                properties["contact"][pkl2lT[c]] = {
                    pkl2lT.get(n, -1): s
                    for n, s in surfaces[c].items()
                    if n % 10**4 == 1 or n in pkl2lT
                }

    successor = {}
    for n, new_id in pkl2lT.items():
        if n in lt:
            successor[new_id] = [pkl2lT[ni] for ni in lt[n] if ni in pkl2lT]

    # do this in the end of the process, skip lineage tree and whatever is stored already
    discard = {
        "cell_volume",  # already stored
        "cell_fate",  # already stored
        "cell_barycenter",  # already stored
        "cell_contact_surface",  # already stored
        "cell_lineage",  # already stored
        "cell_name",  # already stored
        "all_cells",  # not a property
        "cell_history",  # redundant
        "problematic_cells",  # not useful here
        "cell_labels_in_time",  # redundant
    }
    for prop_name, prop_values in tmp_data.items():
        if prop_name not in discard and isinstance(prop_values, dict):
            dictionary = {pkl2lT.get(k, -1): v for k, v in prop_values.items()}
            # is it a regular dictionary or a dictionary with dictionaries inside?
            for key, value in dictionary.items():
                if isinstance(value, dict):
                    # rename all ids from old to new
                    dictionary[key] = {
                        pkl2lT.get(k, -1): v for k, v in value.items()
                    }
            properties[prop_name] = dictionary
    if not name:
        tmp_name = Path(file_path).stem
        if name == "":
            warn(f"Name set to default {tmp_name}", stacklevel=2)
        name = tmp_name
    return LineageTree(
        successor=successor, time=time, pos=pos, name=name, **properties
    )


def read_from_binary(fname: str, name: None | str = None) -> LineageTree:
    """
    Reads a binary LineageTree file name.
    Format description: see LineageTree.to_binary

    Parameters
    ----------
    fname : string
        path to the binary file
    name : None or str, optional
        The name attribute of the LineageTree file. If given a non-empty string, the value of the attribute
        will be the name attribute, otherwise the name will be the stem of the file path.

    Returns
    -------
    LineageTree
        lineage tree
    """
    q_size = struct.calcsize("q")
    H_size = struct.calcsize("H")
    d_size = struct.calcsize("d")

    with open(fname, "rb") as f:
        len_tree = struct.unpack("q", f.read(q_size))[0]
        len_time = struct.unpack("q", f.read(q_size))[0]
        len_pos = struct.unpack("q", f.read(q_size))[0]
        number_sequence = list(
            struct.unpack("q" * len_tree, f.read(q_size * len_tree))
        )
        time_sequence = list(
            struct.unpack("H" * len_time, f.read(H_size * len_time))
        )
        pos_sequence = np.array(
            struct.unpack("d" * len_pos, f.read(d_size * len_pos))
        )

        f.close()

    successor = {}
    time = {}
    pos = {}
    is_root = {}
    waiting_list = []
    i = 0
    done = False
    t = 0
    if max(number_sequence[::2]) == -1:
        tmp = number_sequence[1::2]
        if len(tmp) * 3 == len(pos_sequence) == len(time_sequence) * 3:
            time = dict(list(zip(tmp, time_sequence, strict=True)))
            pos = dict(
                list(
                    zip(
                        tmp,
                        np.reshape(pos_sequence, (len_time, 3)),
                        strict=True,
                    )
                )
            )
            is_root = dict.fromkeys(tmp, True)
            done = True
    while (
        i < len(number_sequence) and not done
    ):  # , c in enumerate(number_sequence[:-1]):
        c = number_sequence[i]
        if c == -1:
            if waiting_list != []:
                prev_mother = waiting_list.pop()
                successor[prev_mother].insert(0, number_sequence[i + 1])
                t = time[prev_mother] + 1
            else:
                t = time_sequence.pop(0)

        elif c == -2:
            successor[waiting_list[-1]] = [number_sequence[i + 1]]
            is_root[number_sequence[i + 1]] = False
            pos[waiting_list[-1]] = pos_sequence[:3]
            pos_sequence = pos_sequence[3:]
            time[waiting_list[-1]] = t
            t += 1

        elif number_sequence[i + 1] >= 0:
            successor[c] = [number_sequence[i + 1]]
            pos[c] = pos_sequence[:3]
            pos_sequence = pos_sequence[3:]
            time[c] = t
            t += 1

        elif number_sequence[i + 1] == -2:
            waiting_list += [c]

        elif number_sequence[i + 1] == -1:
            pos[c] = pos_sequence[:3]
            pos_sequence = pos_sequence[3:]
            time[c] = t
            t += 1
            i += 1
            if waiting_list != []:
                prev_mother = waiting_list.pop()
                successor[prev_mother].insert(0, number_sequence[i + 1])
                t = time[prev_mother] + 1
            else:
                if len(time_sequence) > 0:
                    t = time_sequence.pop(0)
        i += 1
    if not name:
        tmp_name = Path(fname).stem
        if name == "":
            warn(f"Name set to default {tmp_name}", stacklevel=2)
        name = tmp_name
    return LineageTree(successor=successor, time=time, pos=pos, name=name)


def read_from_txt_for_celegans(
    file: str, name: None | str = None
) -> LineageTree:
    """
    Read a C. elegans lineage tree

    Parameters
    ----------
    file : str
        Path to the file to read
    name : None or str, optional
        The name attribute of the LineageTree file. If given a non-empty string, the value of the attribute
        will be the name attribute, otherwise the name will be the stem of the file path.

    Returns
    -------
    LineageTree
        lineage tree
    """
    with open(file) as f:
        raw = f.readlines()[1:]
        f.close()
    _labels = {}
    time_nodes = {}
    pos = {}
    time = {}
    successor = {}

    for unique_id, line in enumerate(raw):
        t = int(line.split("\t")[0])
        _labels[unique_id] = line.split("\t")[1]
        position = np.array(line.split("\t")[2:5], dtype=float)
        time_nodes.setdefault(t, set()).add(unique_id)
        pos[unique_id] = position
        time[unique_id] = t

    t_b = min(time_nodes)

    for t, cells in time_nodes.items():
        if t != t_b:
            prev_cells = time_nodes[t - 1]
            name_to_id = {_labels[c]: c for c in prev_cells}
            for c in cells:
                if _labels[c] in name_to_id:
                    p = name_to_id[_labels[c]]
                elif _labels[c][:-1] in name_to_id:
                    p = name_to_id[_labels[c][:-1]]
                elif IMPLICIT_L_T.get(_labels[c]) in name_to_id:
                    p = name_to_id[IMPLICIT_L_T.get(_labels[c])]
                else:
                    p = None
                successor.setdefault(p, []).append(c)
    if not name:
        tmp_name = Path(file).stem
        if name == "":
            warn(f"Name set to default {tmp_name}", stacklevel=2)
        name = tmp_name
    properties = {"_labels": _labels}
    return LineageTree(
        successor=successor, time=time, pos=pos, name=name, **properties
    )


def read_from_txt_for_celegans_CAO(
    file: str,
    reorder: bool = False,
    raw_size: np.ndarray | None = None,
    shape: float | None = None,
    name: str | None = None,
) -> LineageTree:
    """
    Read a C. elegans lineage tree from Cao et al.

    Parameters
    ----------
    file : str
        Path to the file to read
    name : None or str, optional
        The name attribute of the LineageTree file. If given a non-empty string, the value of the attribute
        will be the name attribute, otherwise the name will be the stem of the file path.

    Returns
    -------
    LineageTree
        lineage tree
    """

    def split_line(line):
        return (
            line.split()[0],
            eval(line.split()[1]),
            eval(line.split()[2]),
            eval(line.split()[3]),
            eval(line.split()[4]),
        )

    with open(file) as f:
        raw = f.readlines()[1:]
        f.close()
    label = {}
    time_nodes = {}
    pos = {}
    successor = {}
    time = {}

    unique_id = 0
    for unique_id, (label, t, z, x, y) in enumerate(map(split_line, raw)):
        label[unique_id] = label
        position = np.array([x, y, z], dtype=np.float)
        time_nodes.setdefault(t, set()).add(unique_id)
        if reorder:

            def flip(x):
                return np.array([x[0], x[1], raw_size[2] - x[2]])

            def adjust(x):
                return (shape / raw_size * flip(x))[[1, 0, 2]]

            pos[unique_id] = adjust(position)
        else:
            pos[unique_id] = position
        time[unique_id] = t

    t_b = min(time_nodes)

    for t, cells in time_nodes.items():
        if t != t_b:
            prev_cells = time_nodes[t - 1]
            name_to_id = {label[c]: c for c in prev_cells}
            for c in cells:
                if label[c] in name_to_id:
                    p = name_to_id[label[c]]
                elif label[c][:-1] in name_to_id:
                    p = name_to_id[label[c][:-1]]
                elif IMPLICIT_L_T.get(label[c]) in name_to_id:
                    p = name_to_id[IMPLICIT_L_T.get(label[c])]
                else:
                    warn(
                        f"error, cell {label[c]} has no predecessors",
                        stacklevel=2,
                    )
                    p = None
                successor.setdefault(p, []).append(c)
    if not name:
        tmp_name = Path(file).stem
        if name == "":
            warn(f"Name set to default {tmp_name}", stacklevel=2)
        name = tmp_name
    return LineageTree(
        successor=successor, time=time, pos=pos, label=label, name=name
    )


def read_from_txt_for_celegans_BAO(
    path: str, name: None | str = None
) -> LineageTree:
    """Read a C. elegans Bao file from http://digital-development.org

    Parameters
    ----------
    file : str
        Path to the file to read
    name : str, optional
        The name attribute of the LineageTree file. If given a non-empty string, the value of the attribute
        will be the name attribute, otherwise the name will be the stem of the file path.

    Returns
    -------
    LineageTree
        lineage tree
    """
    cell_times = {}
    properties = {}
    properties["expression"] = {}
    properties["_labels"] = {}
    with open(path) as f:
        for line in f:
            if "cell_name" not in line:
                cell_times[line.split("\t")[0]] = [
                    eval(val) for val in line.split("\t")[-1].split(",")
                ]
    unique_id = 0
    to_link = {}
    successor = {}
    for c, lc in cell_times.items():
        ids = list(range(unique_id, unique_id + len(lc)))
        successor.update({ids[i]: [ids[i + 1]] for i in range(len(ids) - 1)})
        properties["expression"].update(dict(zip(ids, lc, strict=True)))
        properties["_labels"].update(dict.fromkeys(ids, c))
        to_link[c] = (unique_id, unique_id + len(lc) - 1)
        unique_id += len(lc)

    for c_name, c_id in to_link.items():
        if c_name[:-1] in to_link:
            successor.setdefault(to_link[c_name[:-1]][1], []).append(c_id[0])
        elif c_name in IMPLICIT_L_T and IMPLICIT_L_T[c_name] in to_link:
            successor.setdefault(to_link[IMPLICIT_L_T[c_name]][1], []).append(
                c_id[0]
            )
    if not name:
        tmp_name = Path(path).stem
        if name == "":
            warn(f"Name set to default {tmp_name}", stacklevel=2)
        name = tmp_name
    return LineageTree(
        successor=successor, starting_time=0, name=name, **properties
    )


def read_from_tgmm_xml(
    file_format: str,
    tb: int,
    te: int,
    z_mult: float = 1.0,
    name: None | str = None,
) -> LineageTree:
    """Reads a lineage tree from TGMM xml output.

    Parameters
    ----------
    file_format : str
        path to the xmls location.
        it should be written as follow:
        path/to/xml/standard_name_t{t:06d}.xml where (as an example)
        {t:06d} means a series of 6 digits representing the time and
        if the time values is smaller that 6 digits, the missing
        digits are filed with 0s
    tb : int
        first time point to read
    te : int
        last time point to read
    z_mult : float, default=1.0
        aspect ratio
    name : str, optional
        The name attribute of the LineageTree file. If given a non-empty string, the value of the attribute
        will be the name attribute, otherwise the name will be the stem of the file path.

    Returns
    -------
    LineageTree
        lineage tree
    """
    unique_id = 0
    successor = {}
    pos = {}
    time_id = {}
    time = {}
    properties = {}
    properties["svIdx"] = {}
    properties["lin"] = {}
    properties["C_lin"] = {}
    properties["coeffs"] = {}
    properties["intensity"] = {}
    W = {}
    for t in range(tb, te + 1):
        tree = ET.parse(file_format.format(t=t))
        root = tree.getroot()
        for unique_id, it in enumerate(root):
            if "-1.#IND" not in it.attrib["m"] and "nan" not in it.attrib["m"]:
                M_id, positions, cell_id, svIdx, lin_id = (
                    int(it.attrib["parent"]),
                    [float(v) for v in it.attrib["m"].split(" ") if v != ""],
                    int(it.attrib["id"]),
                    [int(v) for v in it.attrib["svIdx"].split(" ") if v != ""],
                    int(it.attrib["lineage"]),
                )
                if (
                    "alpha" in it.attrib
                    and "W" in it.attrib
                    and "nu" in it.attrib
                    and "alphaPrior" in it.attrib
                ):
                    alpha, W, nu, alphaPrior = (
                        float(it.attrib["alpha"]),
                        [
                            float(v)
                            for v in it.attrib["W"].split(" ")
                            if v != ""
                        ],
                        float(it.attrib["nu"]),
                        float(it.attrib["alphaPrior"]),
                    )
                    positions = np.array(positions)
                    C = unique_id
                    positions[-1] = positions[-1] * z_mult
                    if (t - 1, M_id) in time_id:
                        M = time_id[(t - 1, M_id)]
                        successor.setdefault(M, []).append(C)
                    pos[C] = positions
                    time_id[(t, cell_id)] = C
                    time[C] = t
                    properties["svIdx"][C] = svIdx
                    properties["lin"].setdefault(lin_id, []).append(C)
                    properties["C_lin"][C] = lin_id
                    properties["intensity"][C] = max(alpha - alphaPrior, 0)
                    tmp = list(np.array(W) * nu)
                    W[C] = np.array(W).reshape(3, 3)
                    properties["coeffs"][C] = (
                        tmp[:3] + tmp[4:6] + tmp[8:9] + list(positions)
                    )
    if not name:
        tmp_name = Path(file_format).stem
        if name == "":
            warn(f"Name set to default {tmp_name}", stacklevel=2)
        name = tmp_name
    return LineageTree(
        successor=successor, time=time, pos=pos, name=name, **properties
    )


def read_from_mastodon(
    path: str, tag_set: str | None = None, name: str | None = None
) -> LineageTree:
    """Read a maston lineage tree.

    Parameters
    ----------
    path : str
        path to the mastodon file
    tag_set : str, optional
        If specified, `tag_set` will be used for labeling the cells
        Otherwise a random tag set will be used
    name : str, optional
        The name attribute of the LineageTree file. If given a non-empty string, the value of the attribute
        will be the name attribute, otherwise the name will be the stem of the file path.

    Returns
    -------
    LineageTree
        lineage tree
    """

    mr = MicroMastodonReader(path)
    spots, links = mr.read_tables()

    nodes = list(range(len(spots)))
    pos = dict(zip(nodes, spots[:, :3], strict=True))
    time = dict(zip(nodes, spots[:, 3], strict=True))
    predecessor = {}
    for succ, pred in zip(links[:, 1], links[:, 0]):
        predecessor[int(succ)] = int(pred)

    _, properties, _ = mr.read_tags()

    if isinstance(tag_set, str) and tag_set in properties:
        labels = properties[tag_set]
    elif 0 < len(properties):
        labels_name, labels = next(iter(properties.items()))

    if not name:
        if isinstance(path, Path):
            tmp_name = path.stem
        else:
            tmp_name = Path(path).stem
        if name == "":
            warn(f"Name set to default {tmp_name}", stacklevel=2)
        name = tmp_name

    return LineageTree(
        predecessor=predecessor,
        time=time,
        pos=pos,
        labels=labels,
        labels_name=labels_name,
        name=name,
        **properties,
    )


def read_from_mastodon_csv(
    paths: list[str], name: None | str = None
) -> LineageTree:
    """Read a lineage tree from a mastodon csv.

    Parameters
    ----------
    paths : list of strings
        list of paths to the csv files
    name : None or str, optional
        The name attribute of the LineageTree file. If given a non-empty string, the value of the attribute
        will be the name attribute, otherwise the name will be the stem of the file path.

    Returns
    -------
    LineageTree
        lineage tree
    """
    spots = []
    links = []
    label = {}
    time = {}
    pos = {}
    successor = {}

    with open(paths[0], encoding="utf-8", errors="ignore") as file:
        csvreader = csv.reader(file)
        for row in csvreader:
            spots.append(row)
    spots = spots[3:]

    with open(paths[1], encoding="utf-8", errors="ignore") as file:
        csvreader = csv.reader(file)
        for row in csvreader:
            links.append(row)
    links = links[3:]

    for spot in spots:
        unique_id = int(spot[1])
        x, y, z = spot[5:8]
        t = int(spot[4])
        time[unique_id] = t
        label[unique_id] = spot[1]
        pos[unique_id] = np.array([x, y, z], dtype=float)

    for link in links:
        source = int(float(link[4]))
        target = int(float(link[5]))
        successor.setdefault(source, []).append(target)
    if not name:
        tmp_name = Path(paths[0]).stem
        if name == "":
            warn(f"Name set to default {tmp_name}", stacklevel=2)
        name = tmp_name

    return LineageTree(
        successor=successor, time=time, pos=pos, label=label, name=name
    )


def read_from_mamut_xml(
    path: str, xml_attributes: list[str] | None = None, name: None | str = None
) -> LineageTree:
    """Read a lineage tree from a MaMuT xml.

    Parameters
    ----------
    path : str
        path to the MaMut xml
    name : None or str, optional
        The name attribute of the LineageTree file. If given a non-empty string, the value of the attribute
        will be the name attribute, otherwise the name will be the stem of the file path.

    Returns
    -------
    LineageTree
        lineage tree
    """
    tree = ET.parse(path)
    for elem in tree.getroot():
        if elem.tag == "Model":
            Model = elem
    FeatureDeclarations, AllSpots, AllTracks, FilteredTracks = list(Model)
    xml_attributes = xml_attributes or []

    properties = {}
    for attr in xml_attributes:
        properties[attr] = {}
    nodes = set()
    pos = {}
    time = {}
    properties["label"] = {}

    for frame in AllSpots:
        t = int(frame.attrib["frame"])
        for cell in frame:
            cell_id, n, x, y, z = (
                int(cell.attrib["ID"]),
                cell.attrib["name"],
                float(cell.attrib["POSITION_X"]),
                float(cell.attrib["POSITION_Y"]),
                float(cell.attrib["POSITION_Z"]),
            )
            nodes.add(cell_id)
            pos[cell_id] = np.array([x, y, z])
            time[cell_id] = t
            properties["label"][cell_id] = n
            if "TISSUE_NAME" in cell.attrib:
                if "fate" not in properties:
                    properties["fate"] = {}
                properties["fate"][cell_id] = cell.attrib["TISSUE_NAME"]
            if "TISSUE_TYPE" in cell.attrib:
                if "fate_nb" not in properties:
                    properties["fate_nb"] = {}
                properties["fate_nb"][cell_id] = eval(
                    cell.attrib["TISSUE_TYPE"]
                )
            for attr in cell.attrib:
                if attr in xml_attributes:
                    properties[attr][cell_id] = eval(cell.attrib[attr])

    properties["tracks"] = {}
    successor = {}
    properties["track_name"] = {}
    for track in AllTracks:
        if "TRACK_DURATION" in track.attrib:
            t_id, _ = (
                int(track.attrib["TRACK_ID"]),
                float(track.attrib["TRACK_DURATION"]),
            )
        else:
            t_id = int(track.attrib["TRACK_ID"])
        t_name = track.attrib["name"]
        properties["tracks"][t_id] = []
        for edge in track:
            s, t = (
                int(edge.attrib["SPOT_SOURCE_ID"]),
                int(edge.attrib["SPOT_TARGET_ID"]),
            )
            if s in nodes and t in nodes:
                if time[s] > time[t]:
                    s, t = t, s
                successor.setdefault(s, []).append(t)
                properties["track_name"][s] = t_name
                properties["track_name"][t] = t_name
                properties["tracks"][t_id].append((s, t))
    if not name:
        tmp_name = Path(path).stem
        if name == "":
            warn(f"Name set to default {tmp_name}", stacklevel=2)
        name = tmp_name

    return LineageTree(
        successor=successor,
        time=time,
        pos=pos,
        name=name,
        **properties,
    )
