import streamlit as st
from streamlit_option_menu import option_menu
import json
from streamlit_lottie import st_lottie


#Layout
st.set_page_config(
    page_title="PixelMagic",
    layout="wide",
    initial_sidebar_state="expanded")

#Data Pull and Functions
st.markdown("""
<style>
.big-font {
    font-size:80px !important;
}
</style>
""", unsafe_allow_html=True)


def load_lottiefile(filepath: str):
    with open(filepath,"r") as f:
        return json.load(f)

st.sidebar.write('Welcome to PixelMagic! 👋 ')
languages = ['English', 'Chinese']
selected_language = st.sidebar.selectbox('Please choose your language.', languages)


def english():
    with st.sidebar:
        selected = option_menu('PixelMagic', ["Introduce", 'Image Restoration', 'Upload', 'Perf Comp'],
                               icons=['play-btn', 'image', 'upload', 'info-circle'], menu_icon='intersect',
                               default_index=0)
        lottie = load_lottiefile("Cartoon/cat.json")
        st_lottie(lottie, key='loc')

    from web import intorduce, two_page, update_session_state, display_selected_model
    if selected == "Introduce":
        intorduce()

    if selected == "Image Restoration":
        two_page()

    if selected == 'Upload':
        update_session_state()

    if selected == 'Perf Comp':
        display_selected_model()


def chinese():
    with st.sidebar:
        selected = option_menu('像素魔法', ["功能简介", '图像恢复', '模型上传', '性能评估'],
                               icons=['play-btn', 'image', 'upload', 'info-circle'], menu_icon='intersect',
                               default_index=0)
        lottie = load_lottiefile("Cartoon/cat.json")
        st_lottie(lottie, key='loc')
    from demo import introduce, two_page, update_session_state, display_selected_model

    if selected == "功能简介":
        introduce()

    if selected == "图像恢复":
        two_page()

    if selected == '模型上传':
        update_session_state()

    if selected == '性能评估':
        display_selected_model()


if selected_language == 'English':
    english()

elif selected_language == 'Chinese':
    chinese()
