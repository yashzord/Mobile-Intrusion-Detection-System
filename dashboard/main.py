# dashboard/main.py
import streamlit as st
import pandas as pd
import sqlite3
import os
from PIL import Image
import plotly.express as px

# Define the path to the SQLite database.
DB_PATH = os.path.join(os.path.dirname(__file__), "..", "traffic_data.db")

# Function to load data from the SQLite database based on a query.
@st.cache_data(ttl=60)
def load_data(query: str) -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

# Sidebar navigation for dashboard pages.
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select Page", ("Summary", "Raw Data", "Visualizations", "Image Gallery"))

if page == "Summary":
    st.title("Dashboard Summary")
    try:
        # Load all raw flows from the "flows" table.
        df = load_data("SELECT * FROM flows")
    except Exception as e:
        st.error(f"Error loading data from database: {e}")
    else:
        st.write("### Total Raw Flows Captured:")
        st.write(df.shape[0])
        st.write("### Available Columns:")
        st.write(list(df.columns))
        if "method" in df.columns:
            st.write("### HTTP Method Distribution:")
            method_counts = df["method"].value_counts()
            st.bar_chart(method_counts)
        if "status_code" in df.columns:
            st.write("### Status Code Distribution:")
            fig = px.histogram(df, x="status_code", title="Status Code Distribution")
            st.plotly_chart(fig, use_container_width=True)

elif page == "Raw Data":
    st.title("Raw Flows Data")
    try:
        df = load_data("SELECT * FROM flows")
    except Exception as e:
        st.error(f"Error loading data: {e}")
    else:
        st.write("Displaying the first 100 records:")
        st.dataframe(df.head(100))

elif page == "Visualizations":
    st.title("Data Visualizations")
    try:
        df = load_data("SELECT * FROM flows")
    except Exception as e:
        st.error(f"Error loading data: {e}")
    else:
        if "status_code" in df.columns:
            st.subheader("Status Code Distribution")
            fig = px.histogram(df, x="status_code", title="Status Code Distribution")
            st.plotly_chart(fig, use_container_width=True)
        if "request_content_length" in df.columns:
            st.subheader("Request Content Length Distribution")
            fig2 = px.histogram(df, x="request_content_length", title="Request Content Length")
            st.plotly_chart(fig2, use_container_width=True)
        if "response_content_length" in df.columns:
            st.subheader("Response Content Length Distribution")
            fig3 = px.histogram(df, x="response_content_length", title="Response Content Length")
            st.plotly_chart(fig3, use_container_width=True)

elif page == "Image Gallery":
    st.title("Image Gallery")
    # Images are for your reference; they reside in the 'images' directory.
    images_dir = os.path.join(os.path.dirname(__file__), "..", "images")
    if os.path.isdir(images_dir):
        image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]
        if image_files:
            for image_file in image_files:
                st.subheader(image_file)
                img_path = os.path.join(images_dir, image_file)
                image = Image.open(img_path)
                st.image(image, use_column_width=True)
        else:
            st.info("No images found in the images directory.")
    else:
        st.error("Images directory not found!")
