"""
NAME - data_loading.py contains all functions necessary to load the google landmarks dataset

FILE - Users/ahldenbrough/documents/HCL/data_loading.py

FUNCTIONS:
    -load_data: loads the google landmarks data

"""
import pandas as pd

def load_data():
    """Loads google landmarks data from the path it is stored in

    Args:

    Returns:
        df_train: the training dataset as a pandas dataframe
        df_test: the test dataset as a pandas dataframe
        df_boxes_1: the boxes_1 dataset as a pandas dataframe
        df_boxes_2: the boxes_2 dataset as a pandas dataframe

    Raises:
    """
    base_path = '/Users/ahldenbrough/Documents/HCL/data/'

    # load train data
    df_train = pd.read_csv(base_path + 'train.csv')
    df_test = pd.read_csv(base_path + 'test.csv')
    df_boxes_1 = pd.read_csv(base_path + 'boxes_split1.csv')
    df_boxes_2 = pd.read_csv(base_path + 'boxes_split2.csv')
    return df_train, df_test, df_boxes_1, df_boxes_2