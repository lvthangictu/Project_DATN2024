import streamlit as st 
from app import display_main_page
import time
from PIL import Image, ImageDraw


st.set_page_config(
    page_title="YOLO app",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)
def display_footer():
    st.markdown(
        """
        <style>
            .footer {
                position: fixed;
                left: 0;
                bottom: 0;
                width: 100%;
                background-color: #FFFFFF;
                text-align: center;
                padding: 5px;
            }
        </style>
        <div class="footer">
            <p> 🥝 &copy; 2024 La Văn Thắng | Email: <a href="mailto:thanglavan201@gmail.com">thanglavan201@gmail.com</a> | 
            Facebook: <a href="https://www.facebook.com/thawg203/" target="_blank">Thắng La</a></p>
        </div>
        """, 
        unsafe_allow_html=True
    )

# def main():
#     display_main_page()

if __name__ == "__main__":
   # main()
   display_main_page()
   display_footer()
    