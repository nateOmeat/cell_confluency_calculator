import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, filters, measure, morphology
from skimage.color import label2rgb
from matplotlib.patches import Ellipse
import os
from io import BytesIO
from skimage.filters import threshold_local


st.set_page_config(layout="wide")


st.image('confluency_pic.png', width = 300)

st.title('Cell Confluency Calculator')


st.warning("Note: Make sure the live/dead pairs have the same name other than differing by _fitc/ _af594")
st.info('Example: **test_file_0000_fitc.jpg** / **test_file_0000_af594.jpg**')

col1, col2, = st.columns(2)


with col1: 
    st.header('Live Images (FITC)')

    #loading live FITC - 488nm Image
    live_images = st.file_uploader(f" **Upload Your 3d Live(s) Image Below:**", 
    accept_multiple_files = True)


with col2: 
    st.header('Dead Images (594 nm)')

    #loading dead 594nm image
    dead_images = st.file_uploader(f" **Upload Your 3d Dead Image(s) Below:**",
    accept_multiple_files = True)


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

    mc_area = np.sum(mc_load) - 8478

    #get area of mc_cell_signal
    cell_area = np.sum(cell_signal_binary) - 8478



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
        return percent_confluence

uploaded_files = live_images + dead_images

data = {
    "Live File": [],
    "Dead File": [],
    "Percent Confluence": []
}

if uploaded_files:
    files = [file.name for file in uploaded_files]
    processed_files = []

    for file in files:
        if '_fitc' in file:
            dead_file = file.replace('_fitc', '_af594')

            if dead_file in files and dead_file not in processed_files:
                live_path = next(item for item in live_images if item.name == file)
                dead_path = next(item for item in dead_images if item.name == dead_file)

                result = percent_confluence(dead_path, live_path)

                # Store in data dictionary
                data["Live File"].append(file)
                data["Dead File"].append(dead_file)
                data["Percent Confluence"].append(result)

                # Add to the processed list to avoid double-processing
                processed_files.append(file)
                processed_files.append(dead_file)

    # Generate dataframe from percent confluence results
    df = pd.DataFrame(data)
    mean = np.mean(df['Percent Confluence'])
    std = np.std(df["Percent Confluence"])
    st.dataframe(df)

    csv = df.to_csv(index=False)
    st.download_button(
        label="Download data as CSV",
        data=csv,
        file_name="confluence_data.csv",
        mime="text/csv"
        )



    # Plot 
    fig = px.bar(df, x="Live File", y="Percent Confluence", title="Percent Confluence Summary", template = 'ggplot2', width = 1000)
    fig.update_layout(title_x = 0.5)
    st.plotly_chart(fig)
    st.write('Mean Percent Confluence:',np.round(mean,2), 'Standard Deviation:',np.round(std,2))

