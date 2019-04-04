import collections
import utils
import argparse
import logger
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('--grid_size', type=int, default=256)
parser.add_argument('--ttf_destination_folder', type=str, default='./traffic_features/')
parser.add_argument('--data_destination_folder', type=str, default='./processed_data/')

args = parser.parse_args()


def define_travel_grid_path(data, coords, short_ttf, long_ttf, n):

    # compute size of grid cell
    cell_params = utils.define_grid_cell(*coords, n)

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


def main():
    config = utils.read_config()
    elogger = logger.get_logger()

    # initialize arrays for short-term and long-term traffic features
    speed_array = 'speeds'
    time_array = 'times'
    short_ttf = [
        [collections.defaultdict(lambda: {speed_array: [], time_array: []}) for _ in range(256)] for _ in range(256)
    ]
    long_ttf = [
        [collections.defaultdict(lambda: {speed_array: [], time_array: []}) for _ in range(256)] for _ in range(256)
    ]

    for data_file in config['data']:
        elogger.info('Generating G and T paths and extracting traffic features on {} ...'.format(data_file))

        data = utils.read_data(data_file)

        define_travel_grid_path(data, config['coords'], short_ttf, long_ttf, args.grid_size)

        elogger.info('Saving extended with G and T paths data in {}{}.\n'.format(args.data_destination_folder, data_file))
        utils.save_processed_data(data, args.data_destination_folder, data_file)

    elogger.info('Aggregate historical traffic features ...')
    utils.aggregate_historical_data(short_ttf, long_ttf)
    elogger.info('Saving extracted traffic features in {}'.format(args.ttf_destination_folder))
    utils.save_extracted_traffic_features(short_ttf, long_ttf, args.ttf_destination_folder)


if __name__ == '__main__':
    main()
