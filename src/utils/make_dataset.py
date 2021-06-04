import os
import numpy as np
import pandas as pd

def make_dataset(file_name):
    """
    remove unnecessary positional data and entries with missing values

    Args:
        file_name (str) : the file name of the csv file that stores positional data
    """

    # read the dataset and set the index
    columns = ['timestamp', 'tag_id', 'x_pos', 'y_pos', 'heading', 'direction', 'energy', 'speed', 'total_distance']
    path = os.path.join('../data/input/raw/', file_name)
    data = pd.read_csv(path, names=columns)

    # convert timestamp to second
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data['timestamp'] = data['timestamp'] - data['timestamp'][0]
    data['timestamp'] = [time.total_seconds() for time in data['timestamp']]

    # make the list of player id
    player_ids = data['tag_id'].unique()

    # remove the entries of the goalkeeper and substitute players
    x_means = [] # a list of x-coordinate mean for each player
    y_means = [] # a list of y-coordinate mean for each player
    for player_id in player_ids:
        x_means.append(data['x_pos'][data['tag_id']==player_id].mean())
        y_means.append(data['y_pos'][data['tag_id']==player_id].mean())
    player_ids = np.delete(player_ids, np.argsort(x_means)[0]) # a goalkeeper has the lowest x-position mean
    player_ids = np.delete(player_ids, np.argsort(y_means)[-3:]) # substitute players have the highest y-position mean
    data = data[data['tag_id'].isin(player_ids)]

    # remove the entries with missing values
    timestamp_quarter = [time for time in data['timestamp'].unique() if time % 0.25 == 0] # convert 20 fps to 4 fps
    timestamp_new = []
    x_new = []
    y_new = []
    for time in timestamp_quarter:
        x_quarter = []
        y_quarter = []
        for tag_id in data['tag_id'].unique():
            x_quarter.append(data['x_pos'][(data['tag_id']==tag_id) & (time<=data['timestamp']) & (data['timestamp']<time+0.25)].mean())
            y_quarter.append(data['y_pos'][(data['tag_id']==tag_id) & (time<=data['timestamp']) & (data['timestamp']<time+0.25)].mean())
        if (np.nan not in x_quarter) and (np.nan not in y_quarter): # add data only when the entry does not have any missing values
            timestamp_new.append(time)
            x_new.append(x_quarter)
            y_new.append(y_quarter)

    # make new dataframe
    new_data = pd.DataFrame({'timestamp' : sorted(timestamp_new*10), 
                             'tag_id' : [i for i in data['tag_id'].unique()] * len(timestamp_new),
                             'x_pos' : [x_pos for sublist in x_new for x_pos in sublist],
                             'y_pos' : [y_pos for sublist in y_new for y_pos in sublist]})

    # save new dataframe to csv
    new_path = os.path.join('../data/input/processed/', 'prepped_', file_name)
    new_data.to_csv(new_path, index=False)