import numpy as np
import pickle as pkl
import xml.etree.ElementTree as ET
import csv


class lineageTreeLoaders:
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

    def read_from_txt_for_celegans(self, file: str):
        """
        Read a C. elegans lineage tree

        Args:
            file (str): Path to the file to read
        """
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
                    elif implicit_l_t.get(self.name[c]) in name_to_id:
                        p = name_to_id[implicit_l_t.get(self.name[c])]
                    else:
                        print(
                            "error, cell %s has no predecessors" % self.name[c]
                        )
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
                    elif implicit_l_t.get(self.name[c]) in name_to_id:
                        p = name_to_id[implicit_l_t.get(self.name[c])]
                    else:
                        print(
                            "error, cell %s has no predecessors" % self.name[c]
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
            print(t, end=" ")
            if t % 10 == 0:
                print()
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
