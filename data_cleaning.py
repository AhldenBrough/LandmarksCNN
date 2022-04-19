"""
NAME - data_cleaning.py contains all functions necessary to clean the google landmarks dataset

FILE - Users/ahldenbrough/documents/HCL/data_cleaning.py

FUNCTIONS:
    -drop_by_keyword: creates a dataframe of all rows that contain a url with a keyword
    -clean: cleans the google landmarks dataset

"""

import pandas as pd

def drop_by_keyword(keyword, df_in):
    """Takes in a string keyword and dataframe
    and creates a dataframe that contains all rows with that keyword

    Args:
        keyword: a string containing the keyword to use to identify the rows
        df: the pandas dataframe that will be searched

    Returns:
        None: if PIL.UnidentifiedImageError is raised
        "ok": if PIL.UnidentifiedImageError is not raised

    Raises:
        PIL.UnidentifiedImageError
    """
    drop = df_in[df_in.url.str.contains(keyword)]
    return drop

def clean(df_train, df_test, df_boxes_1, df_boxes_2):
    """Cleans the google landmarks dataset

    Args:
        df_train: the training dataset as a pandas dataframe
        df_test: the test dataset as a pandas dataframe
        df_boxes_1: the boxes_1 dataset as a pandas dataframe
        df_boxes_2: the boxes_2 dataset as a pandas dataframe

    Returns:
        df_train: the cleaned training dataset with boxes merged on as a pandas dataframe
        df_test: the cleaned test dataset as a pandas dataframe
        df_boxes: the cleaned boxes dataset (boxes_1 and boxes_2 concatenated) as a pandas dataframe

    Raises:
    """

    #drop all rows from website panoraio as it is no longer active
    dropping_train = drop_by_keyword('panoramio', df_train)
    dropping_test = drop_by_keyword('panoramio', df_test)

    #drop all rows with no url as those are useless
    dropping_train_2 = df_train.loc[df_train.url == 'None']
    dropping_test_2 = df_test.loc[df_test.url == 'None']

    #create dataframes with all the rows needed to be dropped
    dropping_tr = pd.concat([dropping_train, dropping_train_2], sort=True)
    dropping_te = pd.concat([dropping_test, dropping_test_2], sort=True)

    #drop rows
    df_train = df_train[~df_train.id.isin(dropping_tr.id)]
    df_test = df_test[~df_test.id.isin(dropping_te.id)]
    df_boxes_1 = df_boxes_1[~df_boxes_1.id.isin(dropping_tr.id)]
    df_boxes_2 = df_boxes_2[~df_boxes_2.id.isin(dropping_tr.id)]

    #merge boxes on train set
    df_boxes = pd.concat([df_boxes_1, df_boxes_2])
    df_train = pd.merge(df_train, df_boxes, on='id', how='left')
    return df_train, df_test, df_boxes
