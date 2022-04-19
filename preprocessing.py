"""
NAME - preprocessing.py contains all functions necessary to preprocess the google landmarks dataset

FILE - Users/ahldenbrough/documents/HCL/preprocessing.py

FUNCTIONS:
    -analytics: a basic function for plotting frequency of landmarks
    -url_to_image: converts url to PIL image format
    -url_to_np_array: converts url to numpy array format
    -draw_bbox: draws a bounding box around the landmark
    -save_image: saves an image from a url onto your local machine
    -create_csv: creates a csv of a certain subset of landmarks with an images column
    -drop_faulty: drop all rows that cause an UnidentifiedImageError
    -get_subset_of_landmark: creates a subset of rows

"""
import warnings
from os import path
import numpy as np
import pandas as pd
from skimage import io
import cv2
import PIL
from PIL import Image as im
from PIL import ImageDraw
import seaborn as sns
import matplotlib.pyplot as plt
from data_loading import load_data
from data_cleaning import clean

warnings.filterwarnings("ignore")

#load in and clean data
DF_TRAIN, DF_TEST, DF_BOXES_1, DF_BOXES_2 = load_data()
DF_BOXES = pd.DataFrame()
DF_TRAIN, DF_TEST, DF_BOXES = \
clean(DF_TRAIN, DF_TEST, DF_BOXES_1, DF_BOXES_2)
TOTAL = pd.concat([DF_TRAIN, DF_TEST], sort=True)
base_path = '/Users/ahldenbrough/Documents/HCL/images2/'


def analytics():
    """Basic analysis of landmark frequency for personal use

    Args:

    Returns: displays a graph of the most common landmark ids
    Prints information about the least common landmark ids

    Raises:
    """
    temp = pd.DataFrame(DF_TRAIN.landmark_id.value_counts().head(30))
    temp.reset_index(inplace=True)
    temp.columns = ['landmark_id', 'count']
    # Plot the most frequent landmark_ids
    plt.figure(figsize=(9, 8))
    plt.title('Most frequent landmarks')
    sns.set_color_codes("pastel")
    sns.barplot(x="landmark_id", y="count", data=temp, label="Count")
    plt.show()

    #total unique landmarks
    len(pd.unique(TOTAL['landmark_id']))
    print("Number of classes under 20 occurences", \
        (TOTAL['landmark_id'].value_counts() <= 20).sum(), \
        'out of TOTAL number of categories', len(TOTAL['landmark_id'].unique()))
    print("Number of classes under 10 occurences", \
        (TOTAL['landmark_id'].value_counts() <= 10).sum(), \
        'out of TOTAL number of categories', len(TOTAL['landmark_id'].unique()))
    print("Number of classes with 1 occurence", \
        (TOTAL['landmark_id'].value_counts() == 1).sum(), \
        'out of TOTAL number of categories', len(TOTAL['landmark_id'].unique()))

    DF_TRAIN.head()
    print("shape of DF_TRAIN is: " + str(DF_TRAIN.shape))

def url_to_image(url):
    """Converts a url to an image

    Args:
        url: A string representing the url where the image can be found.

    Returns:
        A PIL image or None if the url cannot be accessed

    Raises:
        Any error caused by trying to access the image and convert
        it will result in a return value of None
    """
    try:
        image = io.imread(url)
        image = np.asarray(image, dtype="uint8")
        image = cv2.resize(image, (224, 224)) #turn black and white
        pil_image = im.fromarray(image)
        print("converts image " + str(url))
        return pil_image
    except:
        print("couldnt convert")
        return None
    #pil_image.show()

def url_to_np_array(row):
    """Converts a url to an numpy array

    Args:
        row: A row of a dataframe.

    Returns:
        An np array representing the image or none if it cannot be accessed or converted

    Raises:
        Any error caused by trying to access the image and convert
        it will result in a return value of None
    """
    try:
        image = io.imread(row.url)
        image = np.asarray(image, dtype="uint8")
        print("converts image " + str(row.url))
        return image
    except:
        print("couldn't convert")
        return None

def draw_bbox(img, boxes):
    """Takes an image bounding box and plots it on image

    Args:
        img: a PIL image containing a landmark
        boxes: a string containing the 4 corners of boxes

    Returns:
        Nothing. Shows an image with the landmark drawn in a bounding box.

    Raises:
    """
    boxes = boxes.split()
    boxes = [224 * float(i) for i in boxes]
    y_coord = boxes[0]
    x_coord = boxes[1]
    height = (boxes[2] - boxes[0])
    width = (boxes[3] - boxes[1])
    shape = [(x_coord, y_coord), (width, height)]
    print(boxes)
    img1 = ImageDraw.Draw(img)
    img1.rectangle(shape, fill=None, outline="green")
    img.show()

def save_image(row):
    """attempts to pull image from internet and download to local machine,
    as well as return path to image download for future reference.

    Args:
        row: A row from a dataframe containing the url of where the image is located.

    Returns:
        A string containing the image path.

    Raises:
    """
    image_name = base_path + row.id +'.jpg'
    if not path.exists(image_name):
        image = url_to_image(row.url)
        if image is not None:
            image.save(base_path + row.id +'.jpg', 'JPEG')
            print("creates image for " + row.landmark_id)
            return image_name
        return None
    print("path exists for " + image_name)
    return image_name

def create_csv(idx, df_in):
    """Creates a csv for a particular landmark id

    Args:
        idx: a string containing a landmark id
        df: a pandas dataframe that you would like to search for the landmark id

    Returns:
        Nothing
        Will save a csv containing the dataframe with all of the corresponding landmarks with idx

    Raises:
    """
    landmarks = df_in.loc[df_in.landmark_id == idx]
    print(landmarks)
    landmarks["images"] = landmarks.apply(save_image, axis='columns')
    landmarks.to_csv(idx + "test.csv")
    print("creates " + idx)

def drop_faulty(row):
    """Marks an images that cause PIL.UnidentifiedImageError

    Args:
        row: a row in a dataframe containing a url

    Returns:
        None: if PIL.UnidentifiedImageError is raised
        "ok": if PIL.UnidentifiedImageError is not raised

    Raises:
        PIL.UnidentifiedImageError
    """
    try:
        PIL.Image.open(row.images)
        #print(row.images + " is good")
    except PIL.UnidentifiedImageError:
        print(row.images + " is faulty")
        return None
    return "ok"

def get_subset_of_landmark(num_images, df_in, image_id):
    """creates a subset of rows corresponding to the image id num images long

    Args:
        num_images: integer representing the number of rows you want to get for your csv
        df: the pandas dataframe to search
        image_id: string containing the landmark id

    Returns:
        nothing, creates a csv using create_csv containing the subset of rows

    Raises:
    """
    name = image_id + "_test"
    subset = df_in[df_in.landmark_id.isin([image_id])]
    num_images_subset = subset.head(num_images)
    create_csv(name, num_images_subset)
