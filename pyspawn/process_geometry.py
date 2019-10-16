import numpy as np


def process_geometry(geom_file='geometry.xyz'):
    """
    This function takes a standard xyz file as input where
    first line is number of atoms
    second line contains comment
    the rest of the lines contain 3D coordinates
    Returns a tuple of:
    1. Number of atoms
    2. Atom names
    3. Coordinates
    4. Comment
    """

    with open(geom_file, 'r') as open_file:
        coords = []
        atom_names = []
        for i, line in enumerate(open_file):
            if i == 0:
                assert len(line.split()) == 1
                n_atoms = int(line)
            elif i == 1:
                comment = line
            else:
                line_list = line.split()
                if len(line_list) > 0:
                    assert len(line_list) == 4, 'wrong xyz file format'
                    coord = [float(num) for num in line_list[1:4]]
                    coords.append(coord)
                    atom_names.append(line_list[0])

    # converting coordinates into 1D numpy array
    array = np.asarray(coords)

    return n_atoms, atom_names, array, comment

#
# number_of_atoms, names, coordinates, comm = process_geometry('geometry.xyz')
# print 'number of atoms:', number_of_atoms
# print 'atom_names', names
# print "coordinates:\n", coordinates
# print "comment =", comm
