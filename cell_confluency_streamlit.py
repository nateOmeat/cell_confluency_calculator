import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, filters, measure, morphology
from skimage.color import label2rgb
from matplotlib.patches import Ellipse
import os
from io import BytesIO
from skimage.filters import threshold_local


st.set_page_config(layout="wide")

#logo
st.image("https://uploads-ssl.webflow.com/643e8b8f7e656b61bd29c098/644240807d96bd6b2322d3a0_OmeatFooterLogo.png",
         width = 500)

st.title('Omeat Cell Confluency Calculator')


#loading live FITC - 488nm Image
live_image = st.file_uploader(f" **Upload Your 3d Live Image Below:**")

#loading dead 594nm image
dead_image = st.file_uploader(f" **Upload Your 3d Dead Image Below:**")



if live_image and dead_image is not None:

    # Display uploaded images side by side
    col1, col2 = st.columns(2)
    col1.image(live_image, caption='Uploaded Live Image.', use_column_width=True)
    col2.image(dead_image, caption='Uploaded Dead Image.', use_column_width=True)




def percent_confluence(image_594, image_fitc, alpha=1, beta=.12, pictures=True):
        # Reading the images directly from the UploadedFile object

    dead_load = io.imread(BytesIO(image_594.read()), as_gray=True)
    #dead_load = np.clip(dead_load* alpha + beta, 0, 1)
    image_594.seek(0)  # Reset file pointer to the beginning
    live_load = io.imread(BytesIO(image_fitc.read()), as_gray=True)
    image_fitc.seek(0)  # Reset file pointer to the beginning


    # Segmentation: Apply thresholding to microcarrier particles

    threshold = filters.threshold_minimum(dead_load)
    mc_binary = dead_load > threshold



    #Invert MC picture to isolate the background
    mc_inverted = np.invert(mc_binary)

    #Image Subtraction to remove cell floaters
    no_floaters =  live_load - mc_inverted


    #Isolating cell signal
    cell_threshold = filters.threshold_otsu(no_floaters)
    cell_signal_binary = no_floaters > cell_threshold
    #cell_signal_binary = no_floaters > threshold_local(no_floaters, block_size=1001, method='mean')

    #color mc images
    mc_labeled = measure.label(mc_binary)
    color_mc = label2rgb(mc_labeled)


    #get area of mc_binary
    mc_load = np.clip(dead_load * alpha + beta, 0, 1)

    mc_area = np.sum(mc_load)

    #get area of mc_cell_signal
    cell_area = np.sum(cell_signal_binary)



    #divide area of cell signal/ mc binary for percent confluence
    percent_confluence = np.round(cell_area *100 / mc_area,2)


    if pictures == True:

        st.write(f"Analyzing images: **{image_fitc.name}** ----------- **{image_594.name}**")

        # Create a figure and set of subplots
        fig, axs = plt.subplots(1, 3, figsize=(20, 5))

        # Original image
        axs[0].imshow(live_load)
        axs[0].set_title('16 Bit Live Image')

        # Binary image
        axs[1].imshow(mc_load)
        axs[1].set_title('16 Bit Dead Image')

        # Eroded binary image
        axs[2].imshow(cell_signal_binary)
        axs[2].set_title('Cell Signal Binary Image')



        st.pyplot(fig)
        st.write(f'Percent Confluence: {percent_confluence}%')

percent_confluence(dead_image,live_image)
