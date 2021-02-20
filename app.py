import tensorflow
from tensorflow import keras
import numpy as np
import streamlit as st
import cv2
from streamlit_drawable_canvas import st_canvas
import pandas as pd
from PIL import Image

SIZE = 252
st.set_option('deprecation.showfileUploaderEncoding', False)

image = Image.open('mnist.jpg')

st.image(image, caption='Handwritting Processing',use_column_width=True)

upload = False

# HEADER
st.write('''
    # Image Processing Neural Network
    
    -----
     
     This is a Neural Network that analyzes and identifies handwritten data in real time
     
    *Click [here](https://github.com/Timmyy3000/handwritten-image-classifier), to find my github repository for this project*
     
  
     
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

    st.subheader('Image Input')
    st.write('''Upload image of a number''')
    uploaded_file = st.file_uploader("Choose an image ...", type="jpg")
    if uploaded_file is not None:
        image = Image.open(uploaded_file)

        image_cv= cv2.rotate((cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)), cv2.ROTATE_90_CLOCKWISE)
        display_img = cv2.resize(image_cv, (300, 300), interpolation=cv2.INTER_NEAREST)
        st.image(display_img, caption='Uploaded Image.', use_column_width=True)
        upload = True

    # User Inputs
    st.subheader('User Input')
    st.write('''Write any single digit from 0 - 9''')


    # different labels in dataset for later encoding
    class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']



    # Instantiating the network
    model = keras.models.load_model('model-digits.h5')

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
        key='canvas'
    )

    if canvas_result.image_data is not None and not upload:
        img = cv2.resize(canvas_result.image_data.astype('uint8'), (28, 28))
        rescaled = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        st.write("Model's Input")
        st.image(rescaled)

    elif upload:

        (thresh, img) = cv2.threshold(image_cv, 90, 255, cv2.THRESH_BINARY_INV)

        img = cv2.resize(img, (120, 120), interpolation=cv2.INTER_CUBIC)


        img = cv2.GaussianBlur(img, (5, 5), 0)
        # steps 2 and 3: Extract the Region of Interest in the image and center in square


        img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_NEAREST)

        rescaled = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
        st.write("Model's Input")
        st.image(rescaled)








    st.write('## Prediction')


    st.write('#### Model Metrics ')
    st.write(pd.DataFrame({'Accuracy' : '98.57', "Loss" : '0.0464'}, index = [0]))
    if st.button('Predict'):
        test_x = np.array(img).astype('float32')
        val = model.predict(test_x.reshape(1, 28, 28, 1))
        st.write(f'# Result : {np.argmax(val[0])}')
        st.bar_chart(val[0])

        st.write("""
                The following table dipicts the prediction probability 
                """)
        st.write(val)







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
            
            This Model has been trained to ~98% accuracy in analyzing and recognizing alphabets characters using a Convolutional Neural Network

            *Real Time Training is coming Soon*
            """)

    st.subheader('Handwritten Alphabets Identifier')
    st.write('This Neural network is trained using a Kaggle Handrwritten A_Z Dataset to indentify handwritten alphabets normalized to a 28 * 28 grid')

    #Image input
    st.subheader('Image Input')
    st.write('''Upload image of a number''')
    uploaded_file = st.file_uploader("Choose an image ...", type="jpg")
    if uploaded_file is not None:
        image = Image.open(uploaded_file)

        image_cv= cv2.rotate((cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)), cv2.ROTATE_90_CLOCKWISE)
        display_img = cv2.resize(image_cv, (300, 300), interpolation=cv2.INTER_NEAREST)
        st.image(display_img, caption='Uploaded Image.', use_column_width=True)
        upload = True


    # User Inputs
    st.subheader('User Input')
    st.write('''Write any latter from A - Z''')

    # different labels in dataset for later encoding
    alphabets_class = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']


    # Instantiating the network
    model = keras.models.load_model('model-alphabets.h5')

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

    elif upload:

        (thresh, img) = cv2.threshold(image_cv, 90, 255, cv2.THRESH_BINARY_INV)

        img = cv2.resize(img, (120, 120), interpolation=cv2.INTER_CUBIC)

        img = cv2.GaussianBlur(img, (5, 5), 0)
        # steps 2 and 3: Extract the Region of Interest in the image and center in square

        img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_NEAREST)

        rescaled = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
        st.write("Model's Input")
        st.image(rescaled)

    st.write('## Prediction')


    st.write('#### Model Metrics ')
    st.write(pd.DataFrame({'Accuracy' : '97.57', "Loss" : '0.057'}, index = [0]))
    if st.button('Predict'):



        test_x = np.array(img).astype('float32')
        val = model.predict(test_x.reshape(1,28,28,1))
        st.write(f'# Result : {alphabets_class[np.argmax(val[0])]}')
        bar = pd.DataFrame(val[0], index=alphabets_class)
        st.bar_chart(bar)

        st.write("""
                The following table dipicts the prediction probability 
                """)
        st.write(bar)


elif mode == ('Handwritten Three Lettered Words') :

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
    st.write(
        'This Neural network is trained using a Kaggle Handrwritten A_Z Dataset to indentify handwritten alphabets normalized to a 28 * 28 grid')

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
         key='canvas'
    )

    if canvas_result.image_data is not None:
        img = cv2.resize(canvas_result.image_data.astype('uint8'), (84, 28))
        rescaled = cv2.resize(img, (SIZE * 3, SIZE), interpolation=cv2.INTER_NEAREST)
        st.write("Model's Input")
        st.image(rescaled)

    st.write('## Prediction')

    st.write('# Coming Soon')
