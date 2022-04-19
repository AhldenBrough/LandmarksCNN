"""
NAME - visualizer.py contains all code for the visualization
of the google landmarks classification project

FILE - Users/ahldenbrough/documents/HCL/visualizer.py

FUNCTIONS:
    -make_string: Makes a string of all the confidence levels for a model to display
    -classify: Classifies an image using all 4 models
    -show_classify_button: shows a button that when clicked calls the classify function
    -upload_image: function for uploading an image

"""
import tkinter as tk
from tkinter import filedialog
from tkinter import Label, Button, BOTTOM
import time
import numpy
from PIL import ImageTk, Image
from keras.models import load_model
import tensorflow as tf

VGG16 = tf.keras.models.load_model('VGG16_FINAL2.h5')
INCEPTIONV3 = tf.keras.models.load_model('INCEPTIONV3_FINAL.h5')
RESNET50 = tf.keras.models.load_model('RESNET50_FINAL.h5')
EFFICIENTNETB7 = tf.keras.models.load_model('EFFICIENTNETB7_FINAL.h5')

classes = {
    0:'Castel Nuovo',
    1:'Luxor Hotel & Casino',
    2:'Conciergerie',
    3:'Lions Gate Bridge',
    4:'Morskie Oko',
    5:'Oslo City Hall',
    6:'Peace Palace',
    7:'Town Hall Tower',
    8:'Montserrat',
    9:'Frankfurt Cathedral',
}

top=tk.Tk()
top.geometry('800x600')
top.title('Landmark Classification')
top.configure(background='#619ead')
label=Label(top,background='#619ead', font=('arial',8,'bold'))

sign_image = Label(top)


def make_string(predictions, model, time_taken):
    """Makes a string of all the confidence levels for a model to display

    Args:
        a: array of confidence levels
        model: string containing the model name

    Returns:
        displays a label with the predictions of all 4 models
    Raises:
    """
    result = model + " predictions"+ "(" + str(round(time_taken, 2))+ "s): \n"
    for i in range(10):
        if predictions[i]*100 > .5:
            result = result + classes[i] + ' ' + str(round(predictions[i]*100, 2)) + "% \n"
    return result + "\n"

def classify(file_path):
    """Classifies an image using all 4 models

    Args:
        file_path: the location of the image to be classified

    Returns:
        the string of confidence levels

    Raises:
    """
    #global label_packed
    image = Image.open(file_path)
    image = image.resize((112,112))
    image = numpy.expand_dims(image, axis=0)
    image = numpy.array(image)
    start_vgg16 = time.perf_counter()
    vgg16_prediction = VGG16.predict([image])[0]
    end_vgg16 = time.perf_counter()
    inceptionv3_prediction = INCEPTIONV3.predict([image])[0]
    end_inceptionv3 = time.perf_counter()
    resnet50_prediction = RESNET50.predict([image])[0]
    end_resnet50 =time.perf_counter()
    efficientnetb7_prediction = EFFICIENTNETB7.predict([image])[0]
    end_efficientnetb7 = time.perf_counter()
    percentages = make_string(vgg16_prediction, "VGG16", end_vgg16-start_vgg16) \
    + make_string(inceptionv3_prediction, "InceptionV3", end_inceptionv3 - end_vgg16) \
    + make_string(resnet50_prediction, "Resnet50", end_resnet50 - end_inceptionv3) \
    + make_string(efficientnetb7_prediction, "EfficientNetB7", end_efficientnetb7 - end_resnet50)

    label.configure(foreground='#011638', text=percentages)


def show_classify_button(file_path):
    """shows a button that when clicked calls the classify function

    Args:
        file_path: the image location being passed into classify()

    Returns:

    Raises:
    """
    classify_b=Button(top,text="Classify Image", command=lambda: classify(file_path),padx=10,pady=5)
    classify_b.configure(background='#364156', foreground='red', font=('arial',10,'bold'))
    classify_b.place(relx=0.79,rely=0.46)

def upload_image():
    """function for uploading an image

    Args:

    Returns:
        calls show_classify_button if the image can be read
    Raises:
    """
    try:
        file_path=filedialog.askopenfilename()
        uploaded=Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width()/2.25),
                            (top.winfo_height()/2.25)))
        img=ImageTk.PhotoImage(uploaded)
        sign_image.configure(image=img)
        sign_image.image=img
        label.configure(text='')
        show_classify_button(file_path)
    except:
        pass

upload=Button(top,text="Upload an image",command=upload_image,padx=10,pady=5)
upload.configure(background='#364156', foreground='red',
font=('arial',10,'bold'))

upload.pack(side=BOTTOM,pady=50)
sign_image.pack(side=BOTTOM,expand=True)
label.pack(side=BOTTOM,expand=True)
heading = Label(top, text='Landmark Classification',pady=20, font=('arial',20,'bold'))

heading.configure(background='#619ead',foreground='#364156')
heading.pack()
top.mainloop()
