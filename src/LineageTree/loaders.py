import csv
import os
import pickle as pkl
import struct
import xml.etree.ElementTree as ET
from warnings import warn

import numpy as np


class lineageTreeLoaders:
    implicit_l_t = {
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

    def read_from_csv(
        self, file_path: str, z_mult: float, link: int = 1, delim: str = ","
    ):
        """
        TODO: write doc
        """
        with open(file_path) as f:
            lines = f.readlines()
            f.close()
        self.time_nodes = {}
        self.time_edges = {}
        unique_id = 0
        self.nodes = set()
        self.edges = set()
        self.successor = {}
        self.predecessor = {}
        self.pos = {}
        self.time_id = {}
        self.time = {}
        self.lin = {}
        self.C_lin = {}
        if not link:
            self.displacement = {}
        lines_to_int = []
        corres = {}
        for line in lines:
            lines_to_int += [[eval(v.strip()) for v in line.split(delim)]]
        lines_to_int = np.array(lines_to_int)
        if link == 2:
            lines_to_int = lines_to_int[np.argsort(lines_to_int[:, 0])]
        else:
            lines_to_int = lines_to_int[np.argsort(lines_to_int[:, 1])]
        for line in lines_to_int:
            if link == 1:
                id_, t, z, y, x, pred, lin_id = line
            elif link == 2:
                t, z, y, x, id_, pred, lin_id = line
            else:
                id_, t, z, y, x, dz, dy, dx = line
                pred = None
                lin_id = None
            t = int(t)
            pos = np.array([x, y, z])
            C = unique_id
            corres[id_] = C
            pos[-1] = pos[-1] * z_mult
            if pred in corres:
                M = corres[pred]
                self.predecessor[C] = [M]
                self.successor.setdefault(M, []).append(C)
                self.edges.add((M, C))
                self.time_edges.setdefault(t, set()).add((M, C))
                self.lin.setdefault(lin_id, []).append(C)
                self.C_lin[C] = lin_id
            self.pos[C] = pos
            self.nodes.add(C)
            self.time_nodes.setdefault(t, set()).add(C)
            self.time[C] = t
            if not link:
                self.displacement[C] = np.array([dx, dy, dz * z_mult])
            unique_id += 1
        self.max_id = unique_id - 1
        self.t_b = min(self.time_nodes)
        self.t_e = max(self.time_nodes)

    def read_from_ASTEC(self, file_path: str, eigen: bool = False):
        """
        Read an `xml` or `pkl` file produced by the ASTEC algorithm.

        Args:
            file_path (str): path to an output generated by ASTEC
            eigen (bool): whether or not to read the eigen values, default False
        """
        self._astec_keydictionary = {
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

        if os.path.splitext(file_path)[-1] == ".xml":
            tmp_data = self._read_from_ASTEC_xml(file_path)
        else:
            tmp_data = self._read_from_ASTEC_pkl(file_path, eigen)

        # make sure these are all named liked they are in tmp_data (or change dictionary above)
        self.name = {}
        if "cell_volume" in tmp_data:
            self.volume = {}
        if "cell_fate" in tmp_data:
            self.fates = {}
        if "cell_barycenter" in tmp_data:
            self.pos = {}
        self.lT2pkl = {}
        self.pkl2lT = {}
        self.contact = {}
        self.prob_cells = set()
        self.image_label = {}

        lt = tmp_data["cell_lineage"]

        if "cell_contact_surface" in tmp_data:
            do_surf = True
            surfaces = tmp_data["cell_contact_surface"]
        else:
            do_surf = False

        inv = {vi: [c] for c, v in lt.items() for vi in v}
        nodes = set(lt).union(inv)

        unique_id = 0

        for n in nodes:
            t = n // 10**4
            self.image_label[unique_id] = n % 10**4
            self.lT2pkl[unique_id] = n
            self.pkl2lT[n] = unique_id
            self.time_nodes.setdefault(t, set()).add(unique_id)
            self.nodes.add(unique_id)
            self.time[unique_id] = t
            if "cell_volume" in tmp_data:
                self.volume[unique_id] = tmp_data["cell_volume"].get(n, 0.0)
            if "cell_fate" in tmp_data:
                self.fates[unique_id] = tmp_data["cell_fate"].get(n, "")
            if "cell_barycenter" in tmp_data:
                self.pos[unique_id] = tmp_data["cell_barycenter"].get(
                    n, np.zeros(3)
                )

            unique_id += 1
        if do_surf:
            for c in nodes:
                if c in surfaces and c in self.pkl2lT:
                    self.contact[self.pkl2lT[c]] = {
                        self.pkl2lT.get(n, -1): s
                        for n, s in surfaces[c].items()
                        if n % 10**4 == 1 or n in self.pkl2lT
                    }

        for n, new_id in self.pkl2lT.items():
            if n in inv:
                self.predecessor[new_id] = [self.pkl2lT[ni] for ni in inv[n]]
            if n in lt:
                self.successor[new_id] = [
                    self.pkl2lT[ni] for ni in lt[n] if ni in self.pkl2lT
                ]

                for ni in self.successor[new_id]:
                    self.time_edges.setdefault(t - 1, set()).add((new_id, ni))

        self.t_b = min(self.time_nodes)
        self.t_e = max(self.time_nodes)
        self.max_id = unique_id

        # do this in the end of the process, skip lineage tree and whatever is stored already
        discard = {
            "cell_volume",
            "cell_fate",
            "cell_barycenter",
            "cell_contact_surface",
            "cell_lineage",
            "all_cells",
            "cell_history",
            "problematic_cells",
            "cell_labels_in_time",
        }
        self.specific_properties = []
        for prop_name, prop_values in tmp_data.items():
            if not (prop_name in discard or hasattr(self, prop_name)):
                if isinstance(prop_values, dict):
                    dictionary = {
                        self.pkl2lT.get(k, -1): v
                        for k, v in prop_values.items()
                    }
                    # is it a regular dictionary or a dictionary with dictionaries inside?
                    for key, value in dictionary.items():
                        if isinstance(value, dict):
                            # rename all ids from old to new
                            dictionary[key] = {
                                self.pkl2lT.get(k, -1): v
                                for k, v in value.items()
                            }
                    self.__dict__[prop_name] = dictionary
                    self.specific_properties.append(prop_name)
                # is any of this necessary? Or does it mean it anyways does not contain
                # information about the id and a simple else: is enough?
                elif (
                    isinstance(prop_values, (list, set, np.ndarray))
                    and prop_name not in []
                ):
                    self.__dict__[prop_name] = prop_values
                    self.specific_properties.append(prop_name)

            # what else could it be?

        # add a list of all available properties

    def _read_from_ASTEC_xml(self, file_path: str):
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

        for k, _v in self._astec_keydictionary.items():
            if root.tag == k:
                dictionary[str(root.tag)] = _set_dictionary_value(root)
                break
        else:
            for child in root:
                value = _set_dictionary_value(child)
                if value is not None:
                    dictionary[str(child.tag)] = value
        return dictionary

    def _read_from_ASTEC_pkl(self, file_path: str, eigen: bool = False):
        with open(file_path, "rb") as f:
            tmp_data = pkl.load(f, encoding="latin1")
            f.close()
        new_ref = {}
        for k, v in self._astec_keydictionary.items():
            for key in v:
                new_ref[key] = k
        new_dict = {}

        for k, v in tmp_data.items():
            if k in new_ref:
                new_dict[new_ref[k]] = v
            else:
                new_dict[k] = v
        return new_dict

    def read_from_binary(self, fname: str):
        """
        Reads a binary lineageTree file name.
        Format description: see self.to_binary

        Args:
            fname: string, path to the binary file
            reverse_time: bool, not used
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
        predecessor = {}
        time = {}
        time_nodes = {}
        time_edges = {}
        pos = {}
        is_root = {}
        nodes = []
        edges = []
        waiting_list = []
        i = 0
        done = False
        if max(number_sequence[::2]) == -1:
            tmp = number_sequence[1::2]
            if len(tmp) * 3 == len(pos_sequence) == len(time_sequence) * 3:
                time = dict(list(zip(tmp, time_sequence)))
                for c, t in time.items():
                    time_nodes.setdefault(t, set()).add(c)
                pos = dict(
                    list(zip(tmp, np.reshape(pos_sequence, (len_time, 3))))
                )
                is_root = {c: True for c in tmp}
                nodes = tmp
                done = True
        while (
            i < len(number_sequence) and not done
        ):  # , c in enumerate(number_sequence[:-1]):
            c = number_sequence[i]
            if c == -1:
                if waiting_list != []:
                    prev_mother = waiting_list.pop()
                    successor[prev_mother].insert(0, number_sequence[i + 1])
                    edges.append((prev_mother, number_sequence[i + 1]))
                    time_edges.setdefault(t, set()).add(
                        (prev_mother, number_sequence[i + 1])
                    )
                    is_root[number_sequence[i + 1]] = False
                    t = time[prev_mother] + 1
                else:
                    t = time_sequence.pop(0)
                    is_root[number_sequence[i + 1]] = True

            elif c == -2:
                successor[waiting_list[-1]] = [number_sequence[i + 1]]
                edges.append((waiting_list[-1], number_sequence[i + 1]))
                time_edges.setdefault(t, set()).add(
                    (waiting_list[-1], number_sequence[i + 1])
                )
                is_root[number_sequence[i + 1]] = False
                pos[waiting_list[-1]] = pos_sequence[:3]
                pos_sequence = pos_sequence[3:]
                nodes.append(waiting_list[-1])
                time[waiting_list[-1]] = t
                time_nodes.setdefault(t, set()).add(waiting_list[-1])
                t += 1

            elif number_sequence[i + 1] >= 0:
                successor[c] = [number_sequence[i + 1]]
                edges.append((c, number_sequence[i + 1]))
                time_edges.setdefault(t, set()).add(
                    (c, number_sequence[i + 1])
                )
                is_root[number_sequence[i + 1]] = False
                pos[c] = pos_sequence[:3]
                pos_sequence = pos_sequence[3:]
                nodes.append(c)
                time[c] = t
                time_nodes.setdefault(t, set()).add(c)
                t += 1

            elif number_sequence[i + 1] == -2:
                waiting_list += [c]

            elif number_sequence[i + 1] == -1:
                pos[c] = pos_sequence[:3]
                pos_sequence = pos_sequence[3:]
                nodes.append(c)
                time[c] = t
                time_nodes.setdefault(t, set()).add(c)
                t += 1
                i += 1
                if waiting_list != []:
                    prev_mother = waiting_list.pop()
                    successor[prev_mother].insert(0, number_sequence[i + 1])
                    edges.append((prev_mother, number_sequence[i + 1]))
                    time_edges.setdefault(t, set()).add(
                        (prev_mother, number_sequence[i + 1])
                    )
                    if i + 1 < len(number_sequence):
                        is_root[number_sequence[i + 1]] = False
                    t = time[prev_mother] + 1
                else:
                    if len(time_sequence) > 0:
                        t = time_sequence.pop(0)
                    if i + 1 < len(number_sequence):
                        is_root[number_sequence[i + 1]] = True
            i += 1

        predecessor = {vi: [k] for k, v in successor.items() for vi in v}

        self.successor = successor
        self.predecessor = predecessor
        self.time = time
        self.time_nodes = time_nodes
        self.time_edges = time_edges
        self.pos = pos
        self.nodes = set(nodes)
        self.t_b = min(time_nodes)
        self.t_e = max(time_nodes)
        self.is_root = is_root
        self.max_id = max(self.nodes)

    def read_from_txt_for_celegans(self, file: str):
        """
        Read a C. elegans lineage tree

        Args:
            file (str): Path to the file to read
        """
        with open(file) as f:
            raw = f.readlines()[1:]
            f.close()
        self.name = {}

        unique_id = 0
        for line in raw:
            t = int(line.split("\t")[0])
            self.name[unique_id] = line.split("\t")[1]
            position = np.array(line.split("\t")[2:5], dtype=float)
            self.time_nodes.setdefault(t, set()).add(unique_id)
            self.nodes.add(unique_id)
            self.pos[unique_id] = position
            self.time[unique_id] = t
            unique_id += 1

        self.t_b = min(self.time_nodes)
        self.t_e = max(self.time_nodes)

        for t, cells in self.time_nodes.items():
            if t != self.t_b:
                prev_cells = self.time_nodes[t - 1]
                name_to_id = {self.name[c]: c for c in prev_cells}
                for c in cells:
                    if self.name[c] in name_to_id:
                        p = name_to_id[self.name[c]]
                    elif self.name[c][:-1] in name_to_id:
                        p = name_to_id[self.name[c][:-1]]
                    elif self.implicit_l_t.get(self.name[c]) in name_to_id:
                        p = name_to_id[self.implicit_l_t.get(self.name[c])]
                    else:
                        p = None
                    self.predecessor.setdefault(c, []).append(p)
                    self.successor.setdefault(p, []).append(c)
                    self.time_edges.setdefault(t - 1, set()).add((p, c))
            self.max_id = unique_id

    def read_from_txt_for_celegans_CAO(
        self,
        file: str,
        reorder: bool = False,
        raw_size: float = None,
        shape: float = None,
    ):
        """
        Read a C. elegans lineage tree from Cao et al.

        Args:
            file (str): Path to the file to read
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
        self.name = {}

        unique_id = 0
        for name, t, z, x, y in map(split_line, raw):
            self.name[unique_id] = name
            position = np.array([x, y, z], dtype=np.float)
            self.time_nodes.setdefault(t, set()).add(unique_id)
            self.nodes.add(unique_id)
            if reorder:

                def flip(x):
                    return np.array([x[0], x[1], raw_size[2] - x[2]])

                def adjust(x):
                    return (shape / raw_size * flip(x))[[1, 0, 2]]

                self.pos[unique_id] = adjust(position)
            else:
                self.pos[unique_id] = position
            self.time[unique_id] = t
            unique_id += 1

        self.t_b = min(self.time_nodes)
        self.t_e = max(self.time_nodes)

        for t, cells in self.time_nodes.items():
            if t != self.t_b:
                prev_cells = self.time_nodes[t - 1]
                name_to_id = {self.name[c]: c for c in prev_cells}
                for c in cells:
                    if self.name[c] in name_to_id:
                        p = name_to_id[self.name[c]]
                    elif self.name[c][:-1] in name_to_id:
                        p = name_to_id[self.name[c][:-1]]
                    elif self.implicit_l_t.get(self.name[c]) in name_to_id:
                        p = name_to_id[self.implicit_l_t.get(self.name[c])]
                    else:
                        warn(
                            f"error, cell {self.name[c]} has no predecessors",
                            stacklevel=2,
                        )
                        p = None
                    self.predecessor.setdefault(c, []).append(p)
                    self.successor.setdefault(p, []).append(c)
                    self.time_edges.setdefault(t - 1, set()).add((p, c))
            self.max_id = unique_id

    def read_tgmm_xml(
        self, file_format: str, tb: int, te: int, z_mult: float = 1.0
    ):
        """Reads a lineage tree from TGMM xml output.

        Args:
            file_format (str): path to the xmls location.
                    it should be written as follow:
                    path/to/xml/standard_name_t{t:06d}.xml where (as an example)
                    {t:06d} means a series of 6 digits representing the time and
                    if the time values is smaller that 6 digits, the missing
                    digits are filed with 0s
            tb (int): first time point to read
            te (int): last time point to read
            z_mult (float): aspect ratio
        """
        self.time_nodes = {}
        self.time_edges = {}
        unique_id = 0
        self.nodes = set()
        self.successor = {}
        self.predecessor = {}
        self.pos = {}
        self.time_id = {}
        self.time = {}
        self.mother_not_found = []
        self.ind_cells = {}
        self.svIdx = {}
        self.lin = {}
        self.C_lin = {}
        self.coeffs = {}
        self.intensity = {}
        self.W = {}
        for t in range(tb, te + 1):
            tree = ET.parse(file_format.format(t=t))
            root = tree.getroot()
            self.time_nodes[t] = set()
            self.time_edges[t] = set()
            for it in root:
                if (
                    "-1.#IND" not in it.attrib["m"]
                    and "nan" not in it.attrib["m"]
                ):
                    M_id, pos, cell_id, svIdx, lin_id = (
                        int(it.attrib["parent"]),
                        [
                            float(v)
                            for v in it.attrib["m"].split(" ")
                            if v != ""
                        ],
                        int(it.attrib["id"]),
                        [
                            int(v)
                            for v in it.attrib["svIdx"].split(" ")
                            if v != ""
                        ],
                        int(it.attrib["lineage"]),
                    )
                    try:
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
                        pos = np.array(pos)
                        C = unique_id
                        pos[-1] = pos[-1] * z_mult
                        if (t - 1, M_id) in self.time_id:
                            M = self.time_id[(t - 1, M_id)]
                            self.successor.setdefault(M, []).append(C)
                            self.predecessor.setdefault(C, []).append(M)
                            self.time_edges[t].add((M, C))
                        else:
                            if M_id != -1:
                                self.mother_not_found.append(C)
                        self.pos[C] = pos
                        self.nodes.add(C)
                        self.time_nodes[t].add(C)
                        self.time_id[(t, cell_id)] = C
                        self.time[C] = t
                        self.svIdx[C] = svIdx
                        self.lin.setdefault(lin_id, []).append(C)
                        self.C_lin[C] = lin_id
                        self.intensity[C] = max(alpha - alphaPrior, 0)
                        tmp = list(np.array(W) * nu)
                        self.W[C] = np.array(W).reshape(3, 3)
                        self.coeffs[C] = (
                            tmp[:3] + tmp[4:6] + tmp[8:9] + list(pos)
                        )
                        unique_id += 1
                    except Exception:
                        pass
                else:
                    if t in self.ind_cells:
                        self.ind_cells[t] += 1
                    else:
                        self.ind_cells[t] = 1
        self.max_id = unique_id - 1

    def read_from_mastodon(self, path: str, name: str):
        """
        TODO: write doc
        """
        from mastodon_reader import MastodonReader

        mr = MastodonReader(path)
        spots, links = mr.read_tables()

        self.node_name = {}

        for c in spots.iloc:
            unique_id = c.name
            x, y, z = c.x, c.y, c.z
            t = c.t
            n = c[name] if name is not None else ""
            self.time_nodes.setdefault(t, set()).add(unique_id)
            self.nodes.add(unique_id)
            self.time[unique_id] = t
            self.node_name[unique_id] = n
            self.pos[unique_id] = np.array([x, y, z])

        for e in links.iloc:
            source = e.source_idx
            target = e.target_idx
            self.predecessor.setdefault(target, []).append(source)
            self.successor.setdefault(source, []).append(target)
            self.time_edges.setdefault(self.time[source], set()).add(
                (source, target)
            )
        self.t_b = min(self.time_nodes.keys())
        self.t_e = max(self.time_nodes.keys())

    def read_from_mastodon_csv(self, path: str):
        """
        TODO: Write doc
        """
        spots = []
        links = []
        self.node_name = {}

        with open(path[0], encoding="utf-8", errors="ignore") as file:
            csvreader = csv.reader(file)
            for row in csvreader:
                spots.append(row)
        spots = spots[3:]

        with open(path[1], encoding="utf-8", errors="ignore") as file:
            csvreader = csv.reader(file)
            for row in csvreader:
                links.append(row)
        links = links[3:]

        for spot in spots:
            unique_id = int(spot[1])
            x, y, z = spot[5:8]
            t = int(spot[4])
            self.time_nodes.setdefault(t, set()).add(unique_id)
            self.nodes.add(unique_id)
            self.time[unique_id] = t
            self.node_name[unique_id] = spot[1]
            self.pos[unique_id] = np.array([x, y, z], dtype=float)

        for link in links:
            source = int(float(link[4]))
            target = int(float(link[5]))
            self.predecessor.setdefault(target, []).append(source)
            self.successor.setdefault(source, []).append(target)
            self.time_edges.setdefault(self.time[source], set()).add(
                (source, target)
            )
        self.t_b = min(self.time_nodes.keys())
        self.t_e = max(self.time_nodes.keys())

    def read_from_mamut_xml(self, path: str):
        """Read a lineage tree from a MaMuT xml.

        Args:
            path (str): path to the MaMut xml
        """
        tree = ET.parse(path)
        for elem in tree.getroot():
            if elem.tag == "Model":
                Model = elem
        FeatureDeclarations, AllSpots, AllTracks, FilteredTracks = list(Model)

        for attr in self.xml_attributes:
            self.__dict__[attr] = {}
        self.time_nodes = {}
        self.time_edges = {}
        self.nodes = set()
        self.pos = {}
        self.time = {}
        self.node_name = {}
        for frame in AllSpots:
            t = int(frame.attrib["frame"])
            self.time_nodes[t] = set()
            for cell in frame:
                cell_id, n, x, y, z = (
                    int(cell.attrib["ID"]),
                    cell.attrib["name"],
                    float(cell.attrib["POSITION_X"]),
                    float(cell.attrib["POSITION_Y"]),
                    float(cell.attrib["POSITION_Z"]),
                )
                self.time_nodes[t].add(cell_id)
                self.nodes.add(cell_id)
                self.pos[cell_id] = np.array([x, y, z])
                self.time[cell_id] = t
                self.node_name[cell_id] = n
                if "TISSUE_NAME" in cell.attrib:
                    if not hasattr(self, "fate"):
                        self.fate = {}
                    self.fate[cell_id] = cell.attrib["TISSUE_NAME"]
                if "TISSUE_TYPE" in cell.attrib:
                    if not hasattr(self, "fate_nb"):
                        self.fate_nb = {}
                    self.fate_nb[cell_id] = eval(cell.attrib["TISSUE_TYPE"])
                for attr in cell.attrib:
                    if attr in self.xml_attributes:
                        self.__dict__[attr][cell_id] = eval(cell.attrib[attr])

        tracks = {}
        self.successor = {}
        self.predecessor = {}
        self.track_name = {}
        for track in AllTracks:
            if "TRACK_DURATION" in track.attrib:
                t_id, _ = (
                    int(track.attrib["TRACK_ID"]),
                    float(track.attrib["TRACK_DURATION"]),
                )
            else:
                t_id = int(track.attrib["TRACK_ID"])
            t_name = track.attrib["name"]
            tracks[t_id] = []
            for edge in track:
                s, t = (
                    int(edge.attrib["SPOT_SOURCE_ID"]),
                    int(edge.attrib["SPOT_TARGET_ID"]),
                )
                if s in self.nodes and t in self.nodes:
                    if self.time[s] > self.time[t]:
                        s, t = t, s
                    self.successor.setdefault(s, []).append(t)
                    self.predecessor.setdefault(t, []).append(s)
                    self.track_name[s] = t_name
                    self.track_name[t] = t_name
                    tracks[t_id].append((s, t))
        self.t_b = min(self.time_nodes.keys())
        self.t_e = max(self.time_nodes.keys())

    def read_C_elegans_bao(self, path):
        cell_times = {}
        self.expression = {}
        with open(path) as f:
            for line in f:
                if "cell_name" not in line:
                    cell_times[line.split("\t")[0]] = list(
                        line.split("\t")[-1].split(",")
                    )
        new_dict = {}
        end_dict = {}
        self.t_e = 0
        self.t_b = 0
        for c, lc in cell_times.items():
            new_dict[c] = self.add_node(0)
            tmp = self.add_branch(
                new_dict[c],
                length=len(lc) - 1,
                reverse=True,
                move_timepoints=True,
            )
            for i, node in enumerate(self.get_cycle(tmp)):
                self.expression[node] = int(lc[i])
            self._labels[self.get_cycle(tmp)[0]] = c
            self._labels.pop(tmp)
            end_dict[c] = self.get_cycle(new_dict[c])[-1]
        cell_names = list(cell_times.keys())
        c_to_p = {}
        while cell_names:
            cur = cell_names.pop()
            if cur[:-1] in cell_names:
                c_to_p[cur] = cur[:-1]
        c_to_p.update(self.implicit_l_t)
        for c, p in c_to_p.items():
            if p in cell_times:
                cyc = end_dict[p]
                self.predecessor[new_dict[c]] = [cyc]
                if cyc not in self.successor:
                    self.successor[cyc] = []
                self.successor[cyc].append(new_dict[c])
        self.time_nodes.clear()
        for root in self.roots:
            to_do = [root]
            while to_do:
                cur = to_do.pop()
                self.time_nodes.setdefault(self.time[cur], set()).add(cur)
                _next = self.successor.get(cur, [])
                to_do += _next
                for n in _next:
                    self.time[n] = self.time[cur] + 1
        self.t_e = max(self.time.values())
