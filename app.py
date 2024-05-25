from pathlib import Path
from PIL import Image
import streamlit as st
import config
from utils_app import load_model, infer_uploaded_image, infer_uploaded_video
from streamlit_lottie import st_lottie
import requests
import json




def load_lottie_url(url:str):
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()    

def load_lottie_file(file_path:str):
    with open(file_path, "r") as file:
        return json.load(file)




def display_main_page():
    
    # st.image('images/banner.jpg', use_column_width=True)
        # Sidebar
    st.sidebar.header(" 🚀 YOLOv8 Models")

    

    model_type = st.sidebar.selectbox(
        "Select Model",
        config.DETECTION_MODEL_LIST_V8,
        key='models_selectbox'
    )

    confidence = float(st.sidebar.slider(
        "Select Model Confidence", 30, 100, 50)) / 100

    model_path = ""
    if model_type:
        model_path = Path(config.DETECTION_MODEL_DIR_V8, str(model_type))
    else:
        st.error("Please Select Model in Sidebar")

    #--------------- Load pretrained DL model------------
    
    try:
        model = load_model(model_path)
    except Exception as e:
        st.error(f"Unable to load model. Please check the specified path: {model_path}")

   
    #--------------Image/video options-------------------

    st.sidebar.header("🖼️ Image/Video Upload")
    source_selectbox = st.sidebar.selectbox(
            "Select Source",
            config.SOURCES_LIST
        )

    uploaded_file = st.sidebar.file_uploader(label="Choose a file...")





    # Check if a file has been uploaded or instructions have been displayed before:
    if uploaded_file is None :
     
       
            st.session_state['instructions_displayed'] = True  
            lottie = """"""
            # lottie = """
            # <script src="https://unpkg.com/@lottiefiles/lottie-player@latest/dist/lottie-player.js"></script>
            # <lottie-player src="https://raw.githubusercontent.com/irinagetman1973/YOLO-Streamlit/main/animation_sphere.json" background="transparent" speed="1" style="width: 400px; height: 400px;" loop autoplay></lottie-player>
            
            # """
            st.markdown("""
                <style>
                    iframe {
                        position: fixed;
                        top: 16rem;
                        bottom: 0;
                        left: 1205;
                        right: 0;
                        margin: auto;
                        z-index=-1;
                    }
                </style>
                """, unsafe_allow_html=True
            )


            st.components.v1.html(lottie, width=410, height=410)
            image = Image.open("logo.jpg")
            col1, col2 = st.columns([1, 4])

            # Display the image in the left column
            with col1:
                st.image(image, use_column_width=False, width=80)
            st.markdown('<div style="text-align: center;font-size: 28px; font-weight: bold;">TRƯỜNG ĐẠI HỌC CÔNG NGHỆ THÔNG TIN VÀ TRUYỀN THÔNG</div>', unsafe_allow_html=True)
            st.markdown('<div style="text-align: center;font-size: 26px; font-weight: bold;">KHOA CÔNG NGHỆ THÔNG TIN</div>', unsafe_allow_html=True)
            
            st.divider()

            st.markdown('<div style="text-align: center;font-size: 26px; font-weight: bold;">ĐỒ ÁN TỐT NGHIỆP</div>', unsafe_allow_html=True)
            st.markdown('<div style="text-align: center;font-size: 24px;">ĐỀ TÀI</div>', unsafe_allow_html=True)
            st.markdown('<div style="text-align: center;font-size: 26px; font-weight: bold;">ỨNG DỤNG THUẬT TOÁN DEEP LEARNING VÀO BÀI TOÁN PHÁT HIỆN ĐIỂM BẤT THƯỜNG CHO HỆ THỐNG GIAO THÔNG ĐƯỜNG BỘ</div>', unsafe_allow_html=True)
            
            st.divider()

            st.markdown('<div style="text-align: center;font-size: 22px;">Sinh viên thực hiện: La Văn Thắng</div>', unsafe_allow_html=True)
            st.markdown('<div style="text-align: center;font-size: 22px;">Giáo viên hướng dẫn: TS. Nguyễn Đình Dũng</div>', unsafe_allow_html=True)
            

                
    if uploaded_file:
        if source_selectbox == config.SOURCES_LIST[0]:  # Image
            infer_uploaded_image(confidence, model, uploaded_file)
        elif source_selectbox == config.SOURCES_LIST[1]:  # Video
            infer_uploaded_video(confidence, model, uploaded_file)

    
