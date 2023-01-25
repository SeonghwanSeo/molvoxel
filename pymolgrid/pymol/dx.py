# https://github.com/mattragoza/LiGAN
def write_grid_to_dx_file(dx_path, values, center, resolution):
    '''
    Write a grid with the provided values,
    center, and resolution to a .dx file.
    '''
    assert len(values.shape) == 3
    assert values.shape[0] == values.shape[1] == values.shape[2]
    assert len(center) == 3

    size = values.shape[0]
    origin = center - resolution*(size - 1)/2.

    lines = [
        'object 1 class gridpositions counts {:d} {:d} {:d}\n'.format(
            size, size, size
        ),
        'origin {:.5f} {:.5f} {:.5f}\n'.format(*origin),
        'delta {:.5f} 0 0\n'.format(resolution),
        'delta 0 {:.5f} 0\n'.format(resolution),
        'delta 0 0 {:.5f}\n'.format(resolution),
        'object 2 class gridconnections counts {:d} {:d} {:d}\n'.format(
            size, size, size
        ),
        'object 3 class array type double rank 0 items ' \
            + '[ {:d} ] data follows\n'.format(size**3),
    ]
    n_points = 0
    line = ''
    values = values.contiguous().view(-1).tolist()
    for i, value in enumerate(values) :
        if i % 3 == 2 :
            line += f'{value:.5f}\n'
        else :
            line += f'{value:.5f} '
    lines.append(line)

    with open(dx_path, 'w') as f:
        f.write(''.join(lines))

