#中文版

import streamlit as st
from streamlit_option_menu import option_menu
import json
from streamlit_lottie import st_lottie
from Restormer import restore
from io import StringIO, BytesIO
import numpy as np
from PIL import Image
import time
import os
import pandas as pd
import shutil
from LAM import Lam
from LPIPS import lp
#from LPIPS import lpips
import yaml
import matplotlib.pyplot as plt
import cv2
from upload import ModelData
from LPIPS.lp import lp

def initialize_session_state():
    class SessionState:
        def __init__(self):
            self.language = ''
            self.last_uploaded_model = None
            self.selected = None  # 在这里添加 selected 属性
            self.weight_file = None
            self.yaml_file = None
            self.arch_file = None

            self.model_names = [
                'RCAN',
                'CARN',
                'RRDBNet',
                'SAN',
                'EDSR',
                'HAT',
                'SWINIR'
            ]
            self.MODEL_LIST = {
                'RCAN': {'Base': 'RCAN.pt'},
                'CARN': {'Base': 'CARN_7400.pth'},
                'RRDBNet': {'Base': 'RRDBNet_PSNR_SRx4_DF2K_official-150ff491.pth'},
                'SAN': {'Base': 'SAN_BI4X.pt'},
                'EDSR': {'Base': 'EDSR-64-16_15000.pth'},
                'HAT': {'Base': 'HAT_SRx4_ImageNet-pretrain.pth'},
                'SWINIR': {'Base': "SwinIR.pth"}

            }
            self.metrics = {
                'RCAN': {'psnr': 32.63, 'ssim': 0.9002, 'lpips': 0.1692},
                'CARN': {'psnr': 32.13, 'ssim': 0.8937, 'lpips': 0.1792},
                'RRDBNet': {'psnr': 32.60, 'ssim': 0.9002, 'lpips': 0.1698},
                'SAN': {'psnr': 32.64, 'ssim': 0.9003, 'lpips': 0.1689},
                'EDSR': {'psnr': 32.46, 'ssim': 0.8968, 'lpips': 0.1725},
                'HAT': {'psnr': 33.18, 'ssim': 0.9037, 'lpips': 0.1618},
                'SWINIR': {'psnr': 32.72, 'ssim': 0.9021, 'lpips': 0.1681}
            }

    return SessionState()



def load_lottiefile(filepath: str):
    with open(filepath,"r") as f:
        return json.load(f)

def click_restore(image, choice):
    img = image
    option = choice
    return restore.main(img, option)


def click_lam(path, choice, Model_list, Model_pth_list):
    img_path = path
    option = choice
    model_list = Model_list
    model_pth_list = Model_pth_list
    return Lam.main0(img_path, option, model_list, model_pth_list)


def click_lam1(path, choice, Model_list, Model_pth_list, Weight_file, Yaml_file, Arch_file):
    img_path = path
    option = choice
    model_list = Model_list
    model_pth_list = Model_pth_list
    return Lam.main1(img_path, option, model_list, model_pth_list, Weight_file, Yaml_file, Arch_file)


def convert_path(path):
    # 将斜杠转换为反斜杠
    converted_path = path.replace("\\", "/")
    return converted_path

def save_image_to_absolute_path(relative_image_path, absolute_output_path):
    current_directory = os.getcwd()
    input_image_path = os.path.join(current_directory, relative_image_path)
    shutil.copyfile(input_image_path, absolute_output_path)


def save_image(image_path, image):
    img = Image.open(image)
    img.save(image_path)


def process_image(upload_file):

    if upload_file is not None:
        #st.markdown('')
        menu_options = ['运动去模糊', '图像去雨', '单图像散焦去模糊', '高斯灰度去噪', '高斯彩色去噪']
        selected_option = st.selectbox('请选择要实现的功能', menu_options)  # 这个是task
        if st.button('开始恢复'):

            # 处理
            st.write('正在恢复中. . .')
            restore_res = click_restore(upload_file, selected_option)
            # 显示检测完成消息
            st.write('{}任务已完成!\n 结果如下:'.format(selected_option))
            st.image(restore_res)
            st.write('图像恢复前后的对比图如下:')
            col1, col2 = st.columns(2)
            with col1:
                st.image(upload_file, width=300)

            with col2:
                st.image(restore_res, width=300)
            return restore_res


def save(res_img, upload_file, save_folder_path):

    # 生成保存图像的文件名
    file_name = upload_file.name.split('.')[0] + '_restored.jpg'
    save_path = os.path.join(save_folder_path, file_name)

    # 将图像数据保存到文件
    try:
        res_img.save(save_path)
        st.success(f"恢复后的图像已成功保存在: {save_path} 🌟")
    except Exception as e:
        st.error(f"保存图像时出错: {e}")


def validate_folder_path(folder_path):
    # 检查路径是否为空
    if not folder_path:
        return False, "文件夹路径不能为空⚠️"

    # 检查路径是否存在
    if not os.path.exists(folder_path):
        return False, f"文件夹路径 '{folder_path}' 不存在."

    # 检查路径是否是文件夹
    if not os.path.isdir(folder_path):
        return False, f"'{folder_path}' 不是有效的文件夹路径."

    return True, ""


def two_page():
    col1, col2 = st.columns([8, 2])
    # 在左侧列添加内容
    with col1:
        st.title("图像恢复")
        st.markdown(" ")
        st.write("在这个页面，您可以体验到图像恢复的神奇之处。🤗")
        # st.write("Please upload the pictures that need to be restored")
    # 在右侧列添加内容
    with col2:
        lottie6 = load_lottiefile("Cartoon/sheep.json")
        st_lottie(lottie6, key='come', height=130, width=150)

    upload_file = st.file_uploader(label='请上传需要恢复的图片', type=['jpg', 'png', 'jpeg'], key="uploader5")
    if upload_file is not None:
        st.image(upload_file)

    save_folder_path = st.text_input("请输入恢复后图像所要保存在的文件夹路径", "")

    is_valid_folder_path, folder_path_error = validate_folder_path(save_folder_path)
    if not is_valid_folder_path:
        st.warning(folder_path_error)
        return

    res_img = process_image(upload_file)
    if res_img is not None:
        save(res_img, upload_file, save_folder_path)


def update_session_state():
    session_state = initialize_session_state()
    col1, col2 = st.columns([7.5, 2.5])
    # 在左侧列添加内容
    with col1:
        st.title('上传模型')
        st.markdown(" ")
        st.write("在这个页面，您可以上传您自己的模型，进而实现模型评估。\n\n 请按照下面的步骤操作！👇")
        st.markdown(" ")
    # 在右侧列添加内容
    with col2:
        lottie8 = load_lottiefile("Cartoon/animal.json")
        st_lottie(lottie8, key='up', height=160, width=160)

    weight_file = st.file_uploader("请上传模型的权重文件", type=['pt', 'pth'])
    yaml_file = st.file_uploader("请上传模型的YAML文件", type=['yml'])
    arch_file = st.file_uploader("请上传模型的结构文件", type=['py'])

    if st.button("开始上传"):
        st.write("正在上传中...")
        if weight_file and yaml_file and arch_file:
            try:
                x = yaml.safe_load(yaml_file.getvalue())
                s = x.get('network_g', {}).get('type')  # s为模型的名字

                if s:
                    if s not in session_state.model_names:
                        st.session_state = initialize_session_state()

                        data = ModelData()
                        yaml_file_content = yaml_file.getvalue().decode('utf-8')
                        st.session_state.weight_file = weight_file
                        st.session_state.yaml_file = yaml_file_content
                        st.session_state.arch_file = arch_file
                        upload_result = data.update(weight_file, yaml_file_content, arch_file)

                        st.session_state.model_names = upload_result[0]
                        st.session_state.MODEL_LIST = upload_result[1]
                        st.session_state.metrics = upload_result[2]

                        #st.write(st.session_state.metrics)
                    with st.container():
                        col1, col2 = st.columns([5, 5])
                        with col1:
                            st.markdown(" ")
                            st.markdown(" ")
                            st.markdown(" ")
                            st.markdown(" ")
                            st.markdown(" ")
                            st.markdown(" ")
                            st.write(f"<span style='font-size: 22px;'>已成功上传{s}模型✅</span>",
                                     unsafe_allow_html=True)

                        with col2:
                            lottie4 = load_lottiefile("Cartoon/star.json")
                            st_lottie(lottie4, key='great', height=250, width=250)

                    return True

            except Exception as e:
                st.error(f"处理YAML文件时出错: {e}")
        else:
            st.error("请上传所有必要的文件并提供架构文件路径.")
    return False


def display_selected_model():
    #session_state = initialize_session_state()

    weight_file = None
    yaml_file = None
    arch_file = None
    if hasattr(st.session_state, 'model_names'):
        model_options = st.session_state.model_names
        model_pth = st.session_state.MODEL_LIST
        metrics = st.session_state.metrics
        weight_file = st.session_state.weight_file
        yaml_file = st.session_state.yaml_file
        arch_file = st.session_state.arch_file

    else:
        session_state = initialize_session_state()
        model_options = session_state.model_names
        model_pth = session_state.MODEL_LIST
        metrics = session_state.metrics


    col1, col2 = st.columns([7.85, 2.15])
    # 在左侧列添加内容
    with col1:
        st.title('性能评估')
        st.markdown(" ")
        st.write("👉 在此页面上，您可以选择不同的图像超分辨率模型进行评估，并查看不同指标的可视化结果，以帮助您评估模型性能的优劣。✨")
        st.markdown(" ")
    with col2:
        lottie9 = load_lottiefile("Cartoon/panda.json")
        st_lottie(lottie9, key='up', height=170, width=170)

    uploaded_file = st.file_uploader("请上传测试所需的图片", type=['jpg', 'png', 'jpeg'], key="uploader2")
    if uploaded_file is not None:
        st.image(uploaded_file)

        # 用户自由选择模型名称
        selected_models = st.multiselect("请选择您想要比较的模型（可多选）", model_options)
        number = len(selected_models)
        if number != 0:
            if st.button("开始评估"):
                st.write('正在评估中. . .')
                model_names = []
                lam = []
                psnr = []
                ssim = []
                lpip = []
                DI = []
                model_s_DI = 0
                model_na = ['RCAN','CARN','RRDBNet','SAN','EDSR','HAT','SWINIR']
                for i in range(number):
                    model_name = selected_models[i]
                    model_names.append(model_name)
                    if model_name in model_na:
                        lam_result = click_lam(uploaded_file, model_name, model_options, model_pth)
                    else:
                        lam_result = click_lam1(uploaded_file, model_name, model_options, model_pth,
                                                weight_file, yaml_file, arch_file)
                    img = lam_result[0]
                    di = lam_result[1]
                    DI.append(di)
                    lam.append(img)
                    psnr.append(metrics[model_name]['psnr'])
                    ssim.append(metrics[model_name]['ssim'])
                    lpip.append(metrics[model_name]['lpips'])

                    if model_name not in model_na:
                        model_s_DI = di
                        new_model_name = model_name

                st.write("所选模型的平均PSNR（峰值信噪比）、SSIM（结构相似性）和LPIPS（学习感知图像块相似度，即人类感知相似度）的比较如下")
                plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 使用微软雅黑作为中文字体
                plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
                # 用户选择的模型和对应的指标值
                data = {"模型名字": model_names, "PSNR": ["{:.2f}".format(p) for p in psnr],
                        "SSIM": ssim,
                        "LPIPS": lpip}
                df = pd.DataFrame(data)
                df['PSNR'] = df['PSNR'].astype(float)
                df['SSIM'] = df['SSIM'].astype(float)
                df['LPIPS'] = df['LPIPS'].astype(float)

                fig, ax = plt.subplots(1, 2, figsize=(12, 5))

                # 绘制PSNR的柱状图
                ax[0].bar(df['模型名字'], df['PSNR'], color='#326fa8')
                ax[0].set_title('平均PSNR比较')
                #ax[0].set_xlabel('模型')
                ax[0].set_ylabel('PSNR')

                # 标记最大值
                max_psnr = df['PSNR'].max()
                min_psnr = df['PSNR'].min()
                max_psnr_index = df['PSNR'].idxmax()
                ax[0].bar(df['模型名字'][max_psnr_index], max_psnr, color='#223d7d')

                # 设置Y轴范围
                ax[0].set_ylim([min_psnr - 1, max_psnr + 1])

                # 在条形上添加数值
                for i, v in enumerate(df['PSNR']):
                    ax[0].text(i, v + 0.01 * (max_psnr - min_psnr), "{:.2f}".format(v), ha='center', va='bottom')

                # 绘制SSIM的柱状图
                ax[1].bar(df['模型名字'], df['SSIM'], color='#dec50b')
                ax[1].set_title('平均SSIM比较')
                #ax[1].set_xlabel('模型')
                ax[1].set_ylabel('SSIM')

                # 标记最大值
                max_ssim = df['SSIM'].max()
                min_ssim = df['SSIM'].min()
                max_ssim_index = df['SSIM'].idxmax()
                ax[1].bar(df['模型名字'][max_ssim_index], max_ssim, color='#e69809')

                # 设置Y轴范围
                ax[1].set_ylim([min_ssim - 0.005, max_ssim + 0.005])

                # 在条形上添加数值
                for i, v in enumerate(df['SSIM']):
                    ax[1].text(i, v + 0.005 * (max_ssim - min_ssim), "{:.4f}".format(v), ha='center',
                               va='bottom')

                st.pyplot(fig)
                fig.tight_layout(pad=3.0)

                # 如需要绘制LPIPS图，可以创建新的子图
                fig2, ax2 = plt.subplots(figsize=(6, 5))

                # 绘制LPIPS的折线图
                ax2.plot(df['模型名字'], df['LPIPS'], marker='o', color='#32a88c')
                ax2.set_title('平均LPIPS比较')
                #ax2.set_xlabel('模型')
                ax2.set_ylabel('LPIPS')

                min_lpip = df['LPIPS'].min()
                min_lpip_index = df['LPIPS'].idxmin()
                max_lpip = df['LPIPS'].max()
                ax2.plot(df['模型名字'][min_lpip_index], min_lpip, marker='o', color='#134a16')

                # 在折线图上的每个数据点上添加数值
                for i, v in enumerate(df['LPIPS']):
                    ax2.text(i, v + 0.05 * (max_lpip - min_lpip), "{:.4f}".format(v), ha='center', va='bottom')


                #使用streamlit显示图形
                with st.container():
                    col1, col2 = st.columns(2)
                    with col1:
                        st.pyplot(fig2)
                    with col2:
                        st.write("")
                        lottie3 = load_lottiefile("Cartoon/evaluation.json")
                        st_lottie(lottie3, key='eval', height=360, width=360)

                #st.table(df)
                st.write('LAM图（局部归因图）的比较结果如下')
                # 图片恢复后储存到这里
                for i in range(len(lam)):
                    st.write(f"模型: {model_names[i]}")
                    st.image(lam[i], caption='', use_column_width=True)
                    # st.write(f"The DI of this case is {DI[i]:.2f}")
                if model_s_DI != 0:
                    st.markdown(" ")
                    st.write("&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;" + f"由上述对比结果中可以看出，{new_model_name}模型的感受野比较小，即图像恢复所能够利用的像素点较少，所以可以通过增大模型的感受野，进而提升图像超分辨率的效果。", unsafe_allow_html=True)

                else:
                    st.write("")


def introduce():
    # Header
    # 使用HTML和CSS设置字体样式

    # 定义中文字体样式
    chinese_font_style = """
            <style>
                @font-face {
                    font-family: 'CustomChineseFont';
                    src: url('path/to/your/chinese/font.ttf');
                }
                body {
                    font-family: 'CustomChineseFont', sans-serif;
                }
            </style>
        """

    # 将样式添加到Streamlit页面
    st.markdown(chinese_font_style, unsafe_allow_html=True)
    st.markdown("""
            <style>
            .title-font {
                font-size:42px !important;
                font-weight:bold !important;
                font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
            }
            .subtitle-font {
                font-size:20px !important;
                font-style:italic !important;
                color:#4a4a4a;
            }
            </style>
            """, unsafe_allow_html=True)

    # 应用自定义样式到标题和子标题
    st.markdown('<div class="title-font">👋欢迎使用像素魔法！</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle-font" style="text-align: right;">一种全新的图像恢复和图像超分辨率模型性能评估工具</div>',
        unsafe_allow_html=True)

    #st.divider()
    st.markdown('---')
    # Use Cases
    with st.container():
        col1, col2 = st.columns([5.5, 4.5])
        with col1:
            st.header('使用案例')

            st.markdown(" ")
            st.markdown(" ")
            st.markdown(
                """

                - 您是否对低分辨率图像感到厌倦，并且希望使用超分辨率图像恢复工具来提升图像质量？
                - 您是否对探索超分辨率图像恢复的技术和能力感兴趣，以改善图像的视觉质量？
                - 您是否正在评估各种模型在超分辨率图像恢复方面的性能，以优化图像处理流程？
                - 您是否想深入了解图像恢复的世界，更多地了解超分辨率图像恢复工具这个迷人的领域？

                """
            )
        with col2:
            lottie2 = load_lottiefile("Cartoon/robot.json")
            st_lottie(lottie2, key='place', height=350, width=340)