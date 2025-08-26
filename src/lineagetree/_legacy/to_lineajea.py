import csv


def write_csv_from_lT_to_lineaja(
    lT, path_to, start: int = 0, finish: int = 300
):
    csv_dict = {}
    for time in range(start, finish):
        for node in lT.time_nodes[time]:
            csv_dict[node] = {"pos": lT.pos[node], "t": time}
    with open(path_to, "w", newline="\n") as file:
        fieldnames = [
            "time",
            "positions_x",
            "positions_y",
            "positions_z",
            "id",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for node in csv_dict:
            writer.writerow(
                {
                    "time": csv_dict[node]["t"],
                    "positions_z": csv_dict[node]["pos"][0],
                    "positions_y": csv_dict[node]["pos"][1],
                    "positions_x": csv_dict[node]["pos"][2],
                    "id": node,
                }
            )
