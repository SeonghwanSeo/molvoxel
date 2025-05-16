# https://github.com/mattragoza/LiGAN
def write_grid_to_dx_file(dx_path, values, center, resolution):
    """
    Write a grid with the provided values,
    center, and resolution to a .dx file.
    """
    assert len(values.shape) == 3
    # assert values.shape[0] == values.shape[1] == values.shape[2]
    assert len(center) == 3

    size_x, size_y, size_z = values.shape
    center_x, center_y, center_z = center
    # origin = center - resolution * (size - 1) / 2.
    origin = (
        (center_x - resolution * (size_x - 1) / 2.0),
        (center_y - resolution * (size_y - 1) / 2.0),
        (center_z - resolution * (size_z - 1) / 2.0),
    )

    lines = [
        f"object 1 class gridpositions counts {size_x:d} {size_y:d} {size_z:d}\n",
        "origin {:.5f} {:.5f} {:.5f}\n".format(*origin),
        f"delta {resolution:.5f} 0 0\n",
        f"delta 0 {resolution:.5f} 0\n",
        f"delta 0 0 {resolution:.5f}\n",
        f"object 2 class gridconnections counts {size_x:d} {size_y:d} {size_z:d}\n",
        "object 3 class array type double rank 0 items " + f"[ {size_x * size_y * size_z:d} ] data follows\n",
    ]
    line = ""
    values = values.reshape(-1).tolist()
    for i, value in enumerate(values):
        if i % 3 == 2:
            line += f"{value:.5f}\n"
        else:
            line += f"{value:.5f} "
    lines.append(line)

    with open(dx_path, "w") as f:
        f.write("".join(lines))
