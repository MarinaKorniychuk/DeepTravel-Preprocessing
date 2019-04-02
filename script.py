import collections
import utils
import numpy as np


def define_travel_grid_path(data, coords, n):

    # compute size of grid cell
    cell_params = utils.define_grid_cell(*coords, n)

    # initialize arrays for short-term and long-term traffic features
    speed_array = 'speeds'
    time_array = 'times'
    short_ttf = [[collections.defaultdict(lambda: {speed_array: [], time_array: []}) for _ in range(256)] for _ in range(256)]
    long_ttf = [[collections.defaultdict(lambda: {speed_array: [], time_array: []}) for _ in range(256)] for _ in range(256)]

    for ddict in data:

        # relative coordinates (as first cell has coordinates coords[1][0]
        x = np.array(ddict['lngs']) - coords[1]
        y = np.array(ddict['lats']) - coords[0]

        # T_path - sequence of grid indices that correspond historical gps points
        # G_path - sequence of grid indices  of full path (with intermediate cells without gps points)
        ddict['T_path_X'], ddict['T_path_Y'], ddict['G_path_X'], ddict['G_path_Y'] = utils.map_gps_to_grid(
            x, y,
            ddict['timeID'],
            ddict['weekID'],
            ddict['time_gap'],
            ddict['dist_gap'],
            cell_params,
            short_ttf,
            long_ttf
        )

    utils.aggregate_historical_data(short_ttf, long_ttf)

    return short_ttf, long_ttf


def main():
    config = utils.read_config()

    for data_file in config['data']:
        data = utils.read_data(data_file)

        short_ttf, long_ttf = define_travel_grid_path(data, config['coords'], config['n'])

        utils.save_extracted_traffic_features(short_ttf, long_ttf, data_file)


if __name__ == '__main__':
    main()
