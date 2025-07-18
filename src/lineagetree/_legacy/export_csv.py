import csv

from lineagetree import LineageTree

"""Writes a lineage tree into a series of csv files.

 * spots.csv: id; label; timePoint; x; y; z; volume
 * links.csv: source spot id; target spot id
 * fates1.csv: tagValue
 * fate_values1.csv: spot id; tagValue
 * fates2.csv: tagValue
 * fate_values2.csv: spot id; tagValue
 * fates3.csv: tagValue
 * fate_values3.csv: spot id; tagValue
"""


def export_to_csv():
    tree = LineageTree("Astec-Pm10_properties.pkl", file_type="ASTEC")

    spots = [["id", "label", "timePoint", "x", "y", "z", "volume"]]
    for node_id in tree.nodes:
        cell_name = tree["cell_name"].get(node_id, "unknown")
        spots.append(
            [
                node_id,
                cell_name,
                tree.time.get(node_id),
                tree.pos[node_id][0],
                tree.pos[node_id][1],
                tree.pos[node_id][2],
                tree.volume[node_id],
            ]
        )
    write("spots.csv", spots)

    links = [["source", "target"]]
    for edge in tree.edges:
        links.append([edge[0], edge[1]])
    write("links.csv", links)

    write_fate(tree.fates, "fates1.csv", tree, "fate_values1.csv")
    write_fate(tree["cell_fate_2"], "fates2.csv", tree, "fate_values2.csv")
    write_fate(tree["cell_fate_3"], "fates3.csv", tree, "fate_values3.csv")


def write_fate(fates, fate_name, tree, fate_value_name):
    all_fates = []
    for index in fates:
        all_fates.append(fates[index])
    unique_fates = set(all_fates)
    fates_list = [["fate"]]
    for fate in unique_fates:
        fates_list.append([fate])
    write(fate_name, fates_list)

    fate_values = [["id", "value"]]
    for node_id in tree.nodes:
        fate_values.append([node_id, tree.fates.get(node_id)])
    write(fate_value_name, fate_values)


def write(file_path, data):
    with open(file_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(data)


if __name__ == "__main__":
    export_to_csv()
