from PIL import Image
import streamlit as st
from comparison import compare_models_function
from vizualization import visualize_inferences



def display_dashboard():   
  
    

    dashboard_sections = ["Compare models", "Statistics"]

        # By default, the section is set to None to show the instruction page.
    section = st.session_state.get('dashboard_section', "")

        # Render the selectbox and store the choice in 'section'
    section = st.sidebar.selectbox("Choose a section to continue:", [""] + dashboard_sections, key="dashboard_section_selectbox", format_func=lambda x: "Select a section..." if x == "" else x)

    st.session_state.dashboard_section = section

    if not section:
        lottie = """
            <script src="https://unpkg.com/@lottiefiles/lottie-player@latest/dist/lottie-player.js"></script>
            <lottie-player src="https://raw.githubusercontent.com/irinagetman1973/YOLO-Streamlit/main/animation_sphere.json" background="transparent" speed="1" style="width: 400px; height: 400px;" loop autoplay></lottie-player>
            """
        st.markdown("""
            <style>
                iframe {
                    position: fixed;
                    top: 16rem;
                    bottom: 0;
                    left: 105;
                    right: 0;
                    margin: auto;
                    z-index=-1;
                }
            </style>
            """, unsafe_allow_html=True
        )


        st.components.v1.html(lottie, width=410, height=410) # When the selection is empty
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
        

            
            

    

    elif section == "Compare models":
      
        compare_models_function() 
    
    elif section == "Statistics":
      
      visualize_inferences()

     
