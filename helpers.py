from __future__ import division

import datetime
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


def map_gps_to_grid(longs, lats, timeID, weekID, time_gap, dist_gap, cell_params, short_ttf, long_ttf, dist):
    T_path_X = []
    T_path_Y = []

    G_path_X = []
    G_path_Y = []

    hour_bins = []
    time_bins = []
    dr_state = [np.zeros(4), ]

    points = zip(longs, lats)
    prev_point = ()

    for ind, point_coords in enumerate(points):
        # define indices of the grid cell to which this gps point belongs (from 0 to 255)
        x_ind = int(point_coords[0] // cell_params[0])
        y_ind = int(point_coords[1] // cell_params[1])
        hour_bin = int((timeID + time_gap[ind - 1] // 60) % 1439 // 60)

        # avoid adding same grid cell more than once (if some consecutive points belongs to one grid cell)
        if T_path_X and x_ind == T_path_X[-1] and y_ind == T_path_Y[-1]:
            pass
        else:
            T_path_X.append(x_ind)
            T_path_Y.append(y_ind)

        # find intermediate cells  if exist before adding current cell to G_path
        # (only if current cell is not very first one)
        if G_path_X:

            cells, intersection_points = find_intermediate_cells(
                prev_point,
                G_path_X[-1], G_path_Y[-1],
                point_coords,
                T_path_X[-1], T_path_Y[-1],
                cell_params
            )

            # extract historical speed and time data for short_ttf and long_ttf dicts
            extract_traffic_features(
                cells,
                prev_point, point_coords,
                intersection_points,
                timeID, weekID,
                time_gap[ind - 1], time_gap[ind],
                dist_gap[ind - 1], dist_gap[ind],
                short_ttf, long_ttf,
                dr_state, len(G_path_X) - 1,
                dist
            )

            time_bin = int((timeID + time_gap[ind - 1] // 60) % 1439 // 5)

            for cell in cells[1: -1]:
                G_path_X.append(cell[0])
                G_path_Y.append(cell[1])

                hour_bins.append(hour_bin)
                time_bins.append(time_bin)

        if G_path_X and x_ind == G_path_X[-1] and y_ind == G_path_Y[-1]:
            pass
        else:
            G_path_X.append(x_ind)
            G_path_Y.append(y_ind)
            hour_bins.append(hour_bin)

            time_bin = int((timeID + time_gap[ind] // 60) % 1439 // 5)

            time_bins.append(time_bin)

        prev_point = point_coords

    dr_state = [nd.tolist() for nd in dr_state]

    return T_path_X, T_path_Y, G_path_X, G_path_Y, hour_bins, time_bins, dr_state


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
    # if i == i_f and j == j_f:
    #     intermediate_path.append([f_x_ind, f_y_ind])

    intersection_points = set()

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
                    coords = find_intersection_point(path_line, line)
                    intersection_found = True
                    intersection_points.add(coords)

            if intersection_found:
                i = i_check
                j = j_check
                intermediate_path.append([intermediate_path[-1][0] + j_step, intermediate_path[-1][1] - i_step])
                break

        if intersection_found:
            intersection_found = False
            continue

        break

    # for now intermediate_path array includes start and finish cells as first and last items
    intersection_points = list(intersection_points)
    intersection_points.sort(key=lambda coords: find_line_segment_length(coords_s[0], coords_s[1], coords[0], coords[1]))

    return intermediate_path, intersection_points


def aggregate_historical_data(short_ttf, long_ttf):
    """
    For each grid cell in short_ttf and long_ttf calculate average historical speed and travel time
    Remove collected historical speeds and times
    :param short_ttf:
    :param long_ttf:
    :return:
    """

    for i in range(256):
        for j in range(256):
            for time_bin, data in short_ttf[i][j].items():
                short_ttf[i][j][time_bin]['speed'] = np.mean(short_ttf[i][j][time_bin]['speeds'])
                short_ttf[i][j][time_bin]['time'] = np.mean(short_ttf[i][j][time_bin]['times'])
                short_ttf[i][j][time_bin]['n'] = len(short_ttf[i][j][time_bin]['times'])
                del short_ttf[i][j][time_bin]['speeds']
                del short_ttf[i][j][time_bin]['times']

    for i in range(256):
        for j in range(256):
            for day, data in long_ttf[i][j].items():
                long_ttf[i][j][day]['speed'] = np.mean(long_ttf[i][j][day]['speeds'])
                long_ttf[i][j][day]['time'] = np.mean(long_ttf[i][j][day]['times'])
                long_ttf[i][j][day]['n'] = len(long_ttf[i][j][day]['times'])
                del long_ttf[i][j][day]['speeds']
                del long_ttf[i][j][day]['times']


def extract_traffic_features(cells, s_point, f_point, int_points, timeID, weekID, s_time, f_time, s_dist, f_dist,
                             short_ttf, long_ttf, dr_state, g_path_len, dist):

    def get_dist_in_metres(dist_gap, dist_gap_in_deg, dist_in_deg):
        """
        Returns calculated part of the path in metres.

        :param dist_gap: part of the path in kilometres
        :param dist_gap_in_deg: part of the path in degrees
        :param dist_in_deg: full path in degrees
        :return: part of the path in metres
        """
        return dist_gap_in_deg / dist_in_deg * dist_gap

    speed_array = 'speeds'
    time_array = 'times'

    dr_state.extend([np.zeros(4) for _ in cells[1:]])

    start_time = (timeID + s_time) % 1439
    time_bin = int((start_time - start_time % 5) // 5)

    # calculate length of the whole path between cons gps points in degrees and metres
    dist_in_deg = find_line_segment_length(*s_point, *f_point)
    dist_gap = (f_dist - s_dist) * 1000

    speed = calculate_speed_for_cell(f_time - s_time, dist_gap)

    if not speed:
        return

    # initialize starting segment with start gps point
    s_segment = s_point

    segs = []

    if not int_points:
        cell = cells[0]
        segs.append(dist_gap)
        short_ttf[cell[1]][cell[0]][time_bin][speed_array].append(speed)
        short_ttf[cell[1]][cell[0]][time_bin][time_array].append(dist_gap / speed)
        long_ttf[cell[1]][cell[0]][weekID][speed_array].append(speed)
        long_ttf[cell[1]][cell[0]][weekID][time_array].append(dist_gap / speed)
        update_driving_states(dr_state, g_path_len, s_dist, dist_gap, dist)

        return

    # extract short-term and long-term features for each segment between two consequential points.
    # points: start_point, [intermediate points ...], finish_point
    # intersection point found for each segment, and travel time divided in ratio of the segment part's lengths.
    for ind, cell in enumerate(cells[1:]):
        prev_cell = cells[ind]
        int_point = int_points[ind]
        if len(int_points) > ind + 1:
            f_segment = int_points[ind + 1]
        else:
            f_segment = f_point

        # calculate lenght of the each part of the segment
        # save extracted speed and calculated time as short-term and ling-term traffic feature for particular cells
        seg_dist = get_dist_in_metres(dist_gap, find_line_segment_length(*s_segment, *int_point), dist_in_deg)
        if seg_dist:
            segs.append(seg_dist)
            short_ttf[prev_cell[1]][prev_cell[0]][time_bin][speed_array].append(speed)
            short_ttf[prev_cell[1]][prev_cell[0]][time_bin][time_array].append(seg_dist / speed)
            long_ttf[prev_cell[1]][prev_cell[0]][weekID][speed_array].append(speed)
            long_ttf[prev_cell[1]][prev_cell[0]][weekID][time_array].append(seg_dist / speed)
            update_driving_states(dr_state, g_path_len + ind, s_dist, seg_dist, dist)

        seg_dist = get_dist_in_metres(dist_gap, find_line_segment_length(*int_point, *f_segment), dist_in_deg)
        if seg_dist:
            segs.append(seg_dist)
            short_ttf[cell[1]][cell[0]][time_bin][speed_array].append(speed)
            short_ttf[cell[1]][cell[0]][time_bin][time_array].append(seg_dist / speed)
            long_ttf[cell[1]][cell[0]][weekID][speed_array].append(speed)
            long_ttf[cell[1]][cell[0]][weekID][time_array].append(seg_dist / speed)
            update_driving_states(dr_state, g_path_len + ind + 1, s_dist, seg_dist, dist)

        # change start segment for the next iteration
        s_segment = f_segment


def update_driving_states(dr_state, ind, s_dist, seg_dist, dist):
    if ind >= len(dr_state):
        dr_state.append(np.zeros(4))
    dr_state[ind][3] += seg_dist / (dist * 1000)
    stage = 0
    if 0.2 < s_dist / dist < 0.8:
        stage = 1
    elif s_dist / dist > 0.8:
        stage = 2

    dr_state[ind][stage] = 1


def calculate_speed_for_cell(time, dist):
    """
    :param time: travel time between two consequential points (sec)
    :param dist: travel distance between two consequential points (m)
    :return: estimated historical speed (m/sec)
    """
    return dist / time


def ccw(a, b, c):
    return (b[1] - a[1]) * (c[0] - b[0]) > (b[0] - a[0]) * (c[1] - b[1])


def check_intersection(a, b):
    return ccw(a[0], a[1], b[0]) != ccw(a[0], a[1], b[1]) and ccw(b[0], b[1], a[0]) != ccw(b[0], b[1], a[1])


def find_intersection_point(a_coords, b_coords):
    """
    Return coordinates of two segments intersection point.
    :param a_coords:
    :param b_coords:
    :return:
    """

    def line(a, b):
        coeff_a = (a[1] - b[1])
        coeff_b = (b[0] - a[0])
        coeff_c = (a[0] * b[1] - b[0] * a[1])
        return coeff_a, coeff_b, -coeff_c

    a_coeffs = line(a_coords[0], a_coords[1])
    b_coeffs = line(b_coords[0], b_coords[1])

    d = a_coeffs[0] * b_coeffs[1] - a_coeffs[1] * b_coeffs[0]
    dx = a_coeffs[2] * b_coeffs[1] - a_coeffs[1] * b_coeffs[2]
    dy = a_coeffs[0] * b_coeffs[2] - a_coeffs[2] * b_coeffs[0]
    if d != 0:
        x = dx / d
        y = dy / d
        return x, y
    else:
        return False


def find_line_segment_length(p1_x, p1_y, p2_x, p2_y):
    p1 = complex(p1_x, p1_y)
    p2 = complex(p2_x, p2_y)

    return abs(p1 - p2)


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


def save_extracted_traffic_features(short_ttf, long_ttf, folder):
    with open(folder + 'short_ttf', 'w') as file:
        for f in short_ttf:
            json.dump(f, file)
            file.write('\n')

    with open(folder + 'long_ttf', 'w') as file:
        for f in long_ttf:
            json.dump(f, file)
            file.write('\n')


def save_processed_data(data, folder, data_file):
    with open(folder + data_file, 'w') as file:
        for d in data:
            json.dump(d, file)
            file.write('\n')


def read_config(config_file='./config.json'):
    with open(config_file) as cfile:
        config = json.load(cfile)

    return config
