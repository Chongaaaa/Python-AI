import streamlit as st
import pandas as pd
import base64

# Load the dataset
df = pd.read_csv(r'C:/Users/Asus/Desktop/Python/dataset/imdb_top_1000.csv')

# Set page configuration
st.set_page_config(page_title="Mobie", page_icon=":movie_camera:", layout="wide")

# Hide full-screen button and style header
css_styles = ''' 
<style>
    button[title="View fullscreen"] {
        display: none;
    }

    .header {
        background-color: #D2DCE6;
        border-radius: 5px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1%;
    }

    .library-selection {
        display: flex;
        align-items: center;
    }

    .library-selection h3 {
        margin: 0 0 0 20px;
        font-size: 1.8em;
        max-width: 100px;
        color: #9DABB4;
    }

    .library-selection button {
        padding: 5px;
        margin-left: 5px;
        border-radius: 10px;
        background-color: #9DABB4;
        border: none;
        color: white;
        cursor: pointer;
        width: 120px;
    }

    .library-selection button:hover {
        background-color: #819192;
    }
</style>
'''

st.markdown(css_styles, unsafe_allow_html=True)

# Create header section with a title and image
header_content = '''
<div class="header">
    <div style="flex: 9;">
        <h1 style="color: #FFFFFF; padding-left: 3%; font-size: 3.5em;">Mobie</h1>
    </div>
    <div style="flex: 1;">
        <img src="data:image/png;base64,{}" width="80">
    </div>
</div>
'''

# Read and encode image to base64
with open('C:/Users/Asus/Desktop/Python/image/profile_pop.png', 'rb') as img_file:
    img_data = base64.b64encode(img_file.read()).decode('utf-8')

# Display header
st.markdown(header_content.format(img_data), unsafe_allow_html=True)

# Year selection section
year_content = ''' 
<div class="library-selection">
    <h3>Year</h3>
    <button>2024</button>
    <button>2023</button>
    <button>2022</button>
    <button>2021</button>
    <button>2020</button>
    <button>2019</button>
    <button>2018</button>
    <button>2017</button>
</div>
'''

# Genres selection section
genres_content = ''' 
<div class="library-selection" style="margin-bottom: 1%;">
    <h3>Genres</h3>
    <button>Action</button>
    <button>Horror</button>
    <button>Sci-fi</button>
    <button>Comedy</button>
    <button>Romance</button>
</div>
'''

# Display year and genres sections
st.markdown(year_content, unsafe_allow_html=True)
st.markdown(genres_content, unsafe_allow_html=True)

st.subheader("Popular Movies")

st.subheader("High Rating Movies")
# Optionally, print the DataFrame
st.dataframe(df)
