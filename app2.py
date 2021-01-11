import tensorflow as tf
from tensorflow import keras
import numpy as np
import streamlit as st
import cv2
from streamlit_drawable_canvas import st_canvas
import pandas as pd
import random
from PIL import Image

SIZE = 252
datasets = keras.datasets 
image = Image.open('mnist.jpg')

st.image(image, caption='Handwritting Processing',use_column_width=True)
# HEADER
st.write('''
    # Image Processing Neural Network
    
    -----
     
     This is a Neural Network that analyzes and identifies handwritten data in real time
     
  
     
''')

st.subheader('Mode Selection')
mode = st.selectbox('Select a mode ', ('Handwritten Digits Using MNIST Dataset', 'Fashion Items Using MNIST Dataset', 'Handwritten Alphabets', 'Handwritten Three Lettered Words'))

if mode ==('Handwritten Digits Using MNIST Dataset') :


    st.subheader('Neural Layers Structure')
    st.write("""
      - Input Layer - 784 neurons gotten from 28 * 28 individual pixel values
      - Hidden Layer - 128 neurons
      - Output Layer - 10 nodes

      """)

    st.subheader('Training')
    st.write("""

    The MNIST database is available at http://yann.lecun.com/exdb/mnist/

    The MNIST database is a dataset of handwritten digits. It has 60,000 training samples, and 10,000 test samples. Each image is represented by 28x28 pixels, each containing a value 0 - 255 with its grayscale value.

    This Model has been trained to ~98% accuracy in analyzing and recognizing handwritten digits using a 3 layered Neural Network
    
    *Real Time Training is coming Soon*
    """)
    # NUMBERS DATA SET IDENTIFIER

    trained = False;

    # User Inputs
    st.subheader('User Input')
    st.write('''Write any single digit from 0 - 9''')
    # getting dataset
    data = datasets.mnist

    # loading train and test data
    (train_images, train_labels), (test_images, test_labels) = data.load_data()

    # different labels in dataset for later encoding
    class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    # normalizing values
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    # Instantiating the network
    model = keras.models.load_model('model-digits.h5', compile = False)

    # Instantiating Drawable canvas
    # mode = st.checkbox("Draw (or Delete)?", True)
    canvas_result = st_canvas(
        fill_color='#000000',
        stroke_width=20,
        stroke_color='#FFFFFF',
        background_color='#000000',
        width=SIZE,
        height=SIZE,
        drawing_mode="freedraw",
        key='canvas')

    if canvas_result.image_data is not None:
        img = cv2.resize(canvas_result.image_data.astype('uint8'), (28, 28))
        rescaled = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
        st.write("Model's Input")
        st.image(rescaled)


    st.write('## Prediction')


    st.write('#### Model Metrics ')
    st.write(pd.DataFrame({'Accuracy' : '98.57', "Loss" : '0.0464'}, index = [0]))
    if st.button('Predict'):


        test_x = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        val = model.predict(test_x.reshape(1, 28, 28))
        st.write(f'# Result : {np.argmax(val[0])}')
        st.bar_chart(val[0])

        st.write("""
                The following table dipicts the prediction probability 
                """)
        st.write(val)






elif mode == ('Fashion Items Using MNIST Dataset') :

    st.subheader('Neural Layers Structure')
    st.write("""
      - Input Layer - 784 neurons gotten from 28 * 28 individual pixel values
      - Hidden Layer - 128 neurons
      - Output Layer - 10 nodes

      """)

    st.subheader('Training')
    st.write("""

        This Fashion MNIST database is available at https://github.com/zalandoresearch/fashion-mnist

        It is a dataset of Zalando's article images — consisting of a training set of 60,000 examples and a test set of 10,000 examples. Each example is a 28x28 grayscale image, associated with a label from 10 classes. The dataset serves as a direct drop-in replacement for the original MNIST dataset for benchmarking machine learning algorithms. It shares the same image size and structure of training and testing splits.

        This Model has been trained to ~90% accuracy in analyzing and recognizing fashion items using a 3 layered Neural Network

        *Real Time Training is coming Soon*
        """)

    # FASHION DATA SET IDENTIFIER

    image = Image.open('head.png')

    st.image(image, use_column_width=True)

    #getting dataset
    data = datasets.fashion_mnist

    model = keras.models.load_model('model-fashion.h5', compile = False)


    # loading train and test data
    (train_images, train_labels), (test_images, test_labels) = data.load_data()

    # different labels in dataset for later encoding
    class_names = ['Top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot']

    # normalizing values
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    # Instantiating Drawable canvas
    st.subheader('Data Classes')
    st.write(pd.Series(class_names))
    st.write('## Prediction')

    st.write('Select Random Item from Test Set')

    input_image = ''
    input_label = ''

    if st.button('Predict') :
        index = random.randint(0, 10000)
        input_image = test_images[index]
        input_label = class_names[test_labels[index]]
        rescaled = cv2.resize(input_image, (190, 190), interpolation=cv2.INTER_NEAREST)
        st.write("Model's Input")
        st.image(rescaled)
        st.write(f'Actual : {input_label}')

        st.write('#### Model Metrics ')
        st.write(pd.DataFrame({'Accuracy': '89.09', "Loss": '0.02954'}, index=[0]))

        val = model.predict(np.array(input_image).reshape(1,28,28))
        st.write(f'# Prediction : {class_names[np.argmax(val[0])]}')
        st.write("""
                                    Prediction Probability 
                                    """)
        bar = pd.Series(val[0], index=class_names)
        st.bar_chart(bar)

        st.write("""
                   The following table dipicts the prediction probability 
                   """)
        st.write(bar)


elif mode == ('Handwritten Alphabets') :

    st.subheader('Convolutional Neural Layers Structure')
    st.write("""
    
      - Input Conv2D Layer - 28 * 28 individual pixel values
      - MaxPooling Layer
      - Droupout Layer
      - Flattened Layer
      - Dense Layer - 128 neurons
      - Output Layer - 26 nodes

      """)

    st.subheader('Training')
    st.write("""

            This A_Z Handwritten Dataset is available at https://www.kaggle.com/sachinpatel21/az-handwritten-alphabets-in-csv-format

            The dataset contains 26 folders (A-Z) containing handwritten images in size 2828 pixels, each alphabet in the image is centre fitted to 2020 pixel box.
            
            The images are taken from NIST(https://www.nist.gov/srd/nist-special-database-19) and NMIST large dataset and few other sources which were then formatted as mentioned above.
            
            This Model has been trained to ~98% accuracy in analyzing and recognizing fashion items using a Convolutional Neural Network

            *Real Time Training is coming Soon*
            """)

    st.subheader('Handwritten Alphabets Identifier')
    st.write('This Neural network is trained using a Kaggle Handrwritten A_Z Dataset to indentify handwritten alphabets normalized to a 28 * 28 grid')

    # User Inputs
    st.subheader('User Input')
    st.write('''Write any latter from A - Z''')

    # different labels in dataset for later encoding
    alphabets_class = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']


    # Instantiating the network
    model = keras.models.load_model('model-alphabets.h5',compile = False)

    # Instantiating Drawable canvas
    # mode = st.checkbox("Draw (or Delete)?", True)
    canvas_result = st_canvas(
        fill_color='#000000',
        stroke_width=20,
        stroke_color='#FFFFFF',
        background_color='#000000',
        width=SIZE ,
        height=SIZE,
        drawing_mode="freedraw",
        key='canvas')

    if canvas_result.image_data is not None:
        img = cv2.resize(canvas_result.image_data.astype('uint8'), (28, 28))
        rescaled = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
        st.write("Model's Input")
        st.image(rescaled)


    st.write('## Prediction')


    st.write('#### Model Metrics ')
    st.write(pd.DataFrame({'Accuracy' : '97.57', "Loss" : '0.057'}, index = [0]))
    if st.button('Predict'):


        test_x = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        test_x = np.array(test_x).astype('float32')
        val = model.predict(test_x.reshape(1,28,28,1))
        st.write(f'# Result : {alphabets_class[np.argmax(val[0])]}')
        bar = pd.DataFrame(val[0], index=alphabets_class)
        st.bar_chart(bar)

        st.write("""
                The following table dipicts the prediction probability 
                """)
        st.write(bar)


elif mode == ('Handwritten Three Lettered Words') :

    # User Inputs
    st.subheader('User Input')
    st.write('''Write any latter from A - Z''')

    # different labels in dataset for later encoding
    alphabets_class = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S',
                       'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

    # Instantiating the network
    model = keras.models.load_model('model-alphabets.h5')

    # Instantiating Drawable canvas
    # mode = st.checkbox("Draw (or Delete)?", True)
    canvas_result = st_canvas(
        fill_color='#000000',
        stroke_width=20,
        stroke_color='#FFFFFF',
        background_color='#000000',
        width=SIZE *3,
        height=SIZE,
        drawing_mode="freedraw",
        # key='canvas'
    )

    if canvas_result.image_data is not None:
        img = cv2.resize(canvas_result.image_data.astype('uint8'), (84, 28))
        rescaled = cv2.resize(img, (SIZE * 3, SIZE), interpolation=cv2.INTER_NEAREST)
        st.write("Model's Input")
        st.image(rescaled)

    st.write('## Prediction')

    st.write('# Coming Soon')
