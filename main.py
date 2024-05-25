import streamlit as st
from app import  display_main_page
#from avatar_manager import  get_avatar_url, store_avatar
import time
from PIL import Image, ImageDraw
import requests
from io import BytesIO
from dashboard import display_dashboard


#-------------Page Configuration-------------------
st.set_page_config(
    page_title="YOLO app",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)




def display_sidebar():
    if st.session_state.page != 'main':
        if st.sidebar.button("Back to Main Page"):
            st.session_state.page = 'main'
            st.rerun()
    if st.session_state.page != 'dashboard':
        if st.sidebar.button("Dashboard"):
            st.session_state.page = 'dashboard'
            st.rerun()


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
            <p> 🥝 &copy; 2024 La Van Thang | Email: <a href="mailto:thanglavan201@gmail.com">thanglavan201@gmail.com</a> 
            
        </div>
        """, 
        unsafe_allow_html=True
    )

def main():
    if 'page' not in st.session_state:
        st.session_state.page = 'main'
    display_sidebar()
    if st.session_state.page == 'main':
       display_main_page()
    elif st.session_state.page == 'dashboard':
       display_dashboard()
    display_footer()

      
   

      

if __name__ == "__main__":
    main()
 