import json

import numpy as np


def define_grid_cell(lat_1, long_1, lat_2, long_2, n=256):
    """
    Returns width and height of cell grid after partition the whole
    road network into n x n disjoint but equal-sized grid cells.

    :param lat_1: latitude for west south point
    :param long_1: longitude for west south point
    :param lat_2: latitude for east north point
    :param long_2: longitude for east north point
    :param n: size of the grid

    :return:
    """

    width = (long_2 - long_1) / n
    height = (lat_2 - lat_1) / n

    return width, height


def map_gps_to_grid(longs, lats, cell_params):
    T_path_X = []
    T_path_Y = []

    G_path_X = []
    G_path_Y = []

    points = zip(longs, lats)
    prev_point = ()

    for ind, point_coords in enumerate(points):
        # define indices of the grid cell to which this gps point belongs (from 0 to 255)
        x_ind = point_coords[0] // cell_params[0]
        y_ind = point_coords[1] // cell_params[1]

        # avoid adding same grid cell more than once (if some consecutive points belongs to one grid cell)
        if T_path_X and x_ind == T_path_X[-1] and y_ind == T_path_Y[-1]:
            continue

        T_path_X.append(x_ind)
        T_path_Y.append(y_ind)

        # find intermediate cells  if exist before adding current cell to G_path
        # (only if current cell is not very first one)
        if G_path_X:

            cells = find_intermediate_cells(
                prev_point,
                G_path_X[-1], G_path_Y[-1],
                point_coords,
                T_path_X[-1], T_path_Y[-1],
                cell_params
            )

            for cell in cells:
                G_path_X.append(cell[0])
                G_path_Y.append(cell[1])

        G_path_X.append(x_ind)
        G_path_Y.append(y_ind)

        prev_point = point_coords

    return T_path_X, T_path_Y, G_path_X, G_path_Y


def find_intermediate_cells(coords_s, s_x_ind, s_y_ind, coords_f, f_x_ind, f_y_ind, cell_params):

    """
    Example of mini grid used in this fuction:

    start cell in global grid [154, 105] (initial value 1)
    finish cell in global grid [157, 104] (initial value 0)

    start cell in mini grid [1, 1]
    finish cell in mini grid [2, 4]


    zero point of mini grid - left upper corner

    [0,0]
    -------------------------
      1 | 1 | 1 | 1 | 1 | 1
    -------------------------
      1 | S | 0 | 0 | 0 | 1
    -------------------------
      1 | 0 | 0 | 0 | F | 1
    -------------------------
      1 | 1 | 1 | 1 | 1 | 1
    -------------------------

    :param coords_s: (lng, lat) of start point
    :param s_x_ind: x (abs) index of start in global grid
    :param s_y_ind: y (ord) index of start in global grid
    :param coords_f: (lng, lat) of finish point
    :param f_x_ind: x (abs) index of finish in global grid
    :param f_y_ind: y (ord) index of finish in global grid
    :param cell_params: (width, height) of grid cell
    :return:
    """

    # initialize mini grid for looking for intermediate path cells
    # contain extra lines and columns from each side to avoid indexing issues
    grid = np.zeros([int(abs(f_y_ind - s_y_ind)) + 3, int(abs(f_x_ind - s_x_ind)) + 3])
    grid[0][:] = 1
    grid[-1][:] = 1

    # label extra lines and columns with 1's (as there is no need to check them)
    for i in range(1, grid.shape[0] - 1):
        grid[i][0] = 1
        grid[i][-1] = 1

    # zero coords for mini grid (left upper corner of mini grid)
    zero_coords = [
        (min(s_x_ind, f_x_ind) - 1)  * cell_params[0],
        (max(s_y_ind, f_y_ind) + 2) * cell_params[1]
    ]

    # coords of line segments between two  historical gps points (start and finish)
    path_line = [[coords_s[0], coords_s[1]], [coords_f[0], coords_f[1]]]

    # define start cell in the mini grid
    i = 1 if s_y_ind >= f_y_ind else int(f_y_ind - s_y_ind) + 1
    j = 1 if s_x_ind <= f_x_ind else int(s_x_ind - f_x_ind) + 1

    # define finish cell in the mini grid
    i_f = 1 if f_y_ind >= s_y_ind else int(s_y_ind - f_y_ind) + 1
    j_f = 1 if f_x_ind <= s_x_ind else int(f_x_ind - s_x_ind) + 1

    # label start cell with 1
    grid[i][j] = 1

    # create list of step directions
    i_steps = [1, -1, 0, 0, 1, -1, 1, -1]
    j_steps = [0, 0, 1, -1, 1, -1, -1, 1]

    intersection_found = False
    intermediate_path = [[s_x_ind, s_y_ind], ]

    # until the finish cell is reached
    while i != i_f or j != j_f:

        for i_step, j_step in zip(i_steps, j_steps):
            i_check = i + i_step
            j_check = j + j_step

            if grid[i_check][j_check]:
                continue

            grid[i_check][j_check] = 1

            # get 4 border line segments of check cell
            lines = get_borders_coords(i_check, j_check, zero_coords, cell_params)
            for line in lines:
                if check_intersection(path_line, line):
                    intersection_found = True
                    i = i_check
                    j = j_check
                    intermediate_path.append([intermediate_path[-1][0] + j_step, intermediate_path[-1][1] - i_step])
                    break

            if intersection_found:
                break

        if intersection_found:
            intersection_found = False
            continue

        break

    # for now intermediate_path array includes start and finish cells as first and last items
    # cut those values to return only intermediate cells (empty if there are no cells found)
    return intermediate_path[1:-1]




def check_intersection(a_coeffs, b_coeffs):
    a_coeffs = line(a_coeffs[0], a_coeffs[1])
    b_coeffs = line(b_coeffs[0], b_coeffs[1])

    d = a_coeffs[0] * b_coeffs[1] - a_coeffs[1] * b_coeffs[0]
    dx = a_coeffs[2] * b_coeffs[1] - a_coeffs[1] * b_coeffs[2]
    dy = a_coeffs[0] * b_coeffs[2] - a_coeffs[2] * b_coeffs[0]
    if d != 0:
        x = dx / d
        y = dy / d
        return x, y
    else:
        return False


def line(a, b):
    coeff_a = (a[1] - b[1])
    coeff_b = (b[0] - a[0])
    coeff_c = (a[0] * b[1] - b[0] * a[1])
    return coeff_a, coeff_b, -coeff_c


def get_borders_coords(i, j, zero_coords, cell_params):
    """
    Returns array of 4 line segments that are borders of the cell.
    Each segment includes coordinates of the beginning and the end of the segment.
    :param i: vertical index for mini grid
    :param j: horizontal index for mini grid
    :param zero_coords:
    :param cell_params:
    :return:
    """
    lines = []

    # left bottom point coords
    x_coords = zero_coords[0] + j * cell_params[0]
    y_coords = zero_coords[1] - (i + 1) * cell_params[1]

    lines.append([
        [x_coords, y_coords],
        [x_coords, y_coords + cell_params[1]]
    ])

    lines.append([
        [x_coords, y_coords + cell_params[1]],
        [x_coords + cell_params[0], y_coords + cell_params[1]]
    ])

    lines.append([
        [x_coords + cell_params[0], y_coords + cell_params[1]],
        [x_coords + cell_params[0], y_coords]
    ])

    lines.append([
        [x_coords + cell_params[0], y_coords],
        [x_coords, y_coords]
    ])

    return lines


def read_data(data_file):
    with open('./data/' + data_file) as dfile:
        data = dfile.readlines()
        data = [json.loads(s) for s in data]

    return data


def read_config(config_file='./config.json'):
    with open(config_file) as cfile:
        config = json.load(cfile)

    return config
