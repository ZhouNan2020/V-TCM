# %%
import numpy as np
import pandas as pd
from io import BytesIO
from xlsxwriter import Workbook
from pyxlsb import open_workbook as open_xlsb
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import font_manager
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import PCA, TruncatedSVD
from collections import Counter, OrderedDict
import copy as cp
from sklearn.decomposition import LatentDirichletAllocation as LDiA
import gensim
from PIL import Image
import streamlit as st
import altair as alt
# %%
# Path: V-TCM\streamlit_app.py
# 全局设置
# streamlit
tab1, tab2, tab3, tab4, tab5, tab6,tab7,tab8 = st.tabs(
    ["Descriptive statistics", "Prescription similarity", "herbal generality",
     "LSA topic distribution", "LDiA topic distribution","word embedding", "Download",
     "About the program"])

# matplotlib
font = font_manager.FontProperties(fname='simhei.ttf')

parameters = {'xtick.labelsize': 20,
              'ytick.labelsize': 20,
              'font.family':'SimHei',
              'axes.unicode_minus':False}
plt.rcParams.update(parameters)
fontsize = 17
plt.style.use('ggplot')

# %%
from library import conver, read_file,tf_idf, base_frame,dot_cos_cal

# %%
# 读取示例数据
out1 = pd.read_excel('English example.xlsx')
out2 = pd.read_excel('中文示例.xlsx')
eng_exmp=conver(out1)
chn_exmp=conver(out2)


# %%
# 侧栏上传文件区域
with st.sidebar:
    file = st.file_uploader("Click “Browse files” to upload files", type=["xlsx"])
    st.write('Please upload a file no larger than 200MB')
    st.write('The file must be a .xlsx file')
    st.download_button('Download sample data in English', data=eng_exmp.file, file_name='sample data in English.xlsx',
                       )
    st.download_button('下载中文示例数据', data=chn_exmp.file, file_name='中文示例数据.xlsx')
    st.write('Note: You can understand the workflow of this program by uploading sample data.')
    st.write(
        'Note: When the program is running, there will be a little man doing sports in the upper right corner of the web page,don\`t refresh this page or do anything else until he stops.')
#%%
# 测试文件
# file=pd.read_csv("English example.csv")
#%%
def file_pre(f):
    if file!=None:
        txt = read_file.read(file)
    else:
        txt = pd.DataFrame(out1)
        col = txt.columns
        txt = txt.set_index(col[0])
    return txt
txt=file_pre(file)
# %%
st.write('You can use the cursor keys "←" and "→" to see more tags')

f=base_frame(txt)
herb_list = f.herb_list
file_dict = f.file_dict
list_vect = f.list_vect
total_herb_list = f.total_herb_list()
total_herb_word_list = f.total_herb_word_list()
avg_len = f.avg_len()
count_herb = f.count_herb()

Counter_every_herb = f.count_herb()
most_common_herb2 = Counter_every_herb.most_common()
most_common_herb2 = pd.DataFrame(most_common_herb2, columns=['herb', 'count'])
full_common_data = most_common_herb2.copy()
herb_dense_dataframe = f.herb_dense_dataframe()
lexicon = f.lexicon()
tf_idf_dict = f.tf_idf_dict(lexicon=lexicon)
idf_df=tf_idf_dataframe = tf_idf.tf_idf_dataframe(tf_idf_dict)

#herb_list = format.herb_list(txt)
#file_dict = format.file_dict(txt)

#total_herb_list = count_list.total_herb_list(herb_list)
#total_herb_word_list = count_list.total_herb_word_list(herb_list)
#avg_len = count_dict.avg_len(file_dict)
#count_herb = count_list.count_herb(file_dict)
#
#Counter_every_herb = count_list.count_herb(herb_list)
#most_common_herb2 = Counter_every_herb.most_common()
#most_common_herb2 = pd.DataFrame(most_common_herb2, columns=['herb', 'count'])
#full_common_data = most_common_herb2.copy()

with tab1:


    st.write('1.The total number of different herbs: ', total_herb_list)
    st.write('2.The total number of herbs is:', total_herb_word_list)
    st.write('3.The average length of prescription: ', round(avg_len, 0))
    st.write('4.The most common herb')
    num1 = st.select_slider(
        'How many herbs do you need to display by frequency?',
        options=range(1, 50, 1), key=1)
    most_common_herb1 = Counter_every_herb.most_common(num1)
    most_common_herb1 = pd.DataFrame(most_common_herb1, columns=['herb', 'count'])
    if st.button('Launch', key=2):
        st.write('The most common herb is: ')
        st.table(most_common_herb1)
        if most_common_herb1.empty == False:
            fig1, ax1 = plt.subplots()
            x = most_common_herb1['herb']
            y = most_common_herb1['count']
            y = list(y)
            y.reverse()  # 倒序
            ax1.barh(x, y, align='center', color='c', tick_label=list(x))
            plt.ylabel('herbs', fontsize=fontsize, fontproperties=font)
            plt.yticks(x,fontsize=fontsize,fontproperties=font)
            st.pyplot(fig1)
    # 排序下载
    if full_common_data.empty == False:
        full_common_data = conver(full_common_data)
        st.download_button(
            label="Download full herb frequency data",
            data=full_common_data.file,
            file_name='full_common_data.xlsx',
            mime='xlsx')
    # 密集矩阵下载
    if herb_dense_dataframe.empty == False:
        herb_dense_dataframe = conver(herb_dense_dataframe)
        st.download_button(
            label='Download dense matrix',
            data=herb_dense_dataframe.file,
            file_name='dense matrix.xlsx')
    # tf-idf矩阵下载
    if idf_df.empty == False:
        tf_idf_matrix = conver(idf_df)
        st.download_button(
            label='Download tf_idf_matrix',
            data=tf_idf_matrix.file,
            file_name='tf_idf_matrix.xlsx')
        close=st.button('Terminate descriptive statistics', key=3)
        st.write('To ensure that the WebApp retains enough memory, try to terminate unneeded modules when appropriate')
        if close:
            st.experimental_memo.clear()
#%%




with tab2:
    st.write('3.Focus on dot product and cosine similarity for a specific prescription')
    options = list(txt.index)
    select_result = st.multiselect(
        'Please select the name of the prescription you wish to follow', options=options, key=7)
    dense_dot_df = pd.DataFrame()
    cos_dot_df = pd.DataFrame()
    if st.button('Launch', key=8):
        dot=(dot_cos_cal.dot_cos(select_result=select_result,herb_dense_dataframe=herb_dense_dataframe))[0]
        cos=(dot_cos_cal.dot_cos(select_result=select_result,herb_dense_dataframe=herb_dense_dataframe))[1]

        fig2, ax2 = plt.subplots()
        sns.heatmap(dot, annot=True, fmt=".2g", linewidths=.5, cmap='YlOrRd')
        ax2.set_title('Dot product')
        plt.xticks(font=font)
        plt.yticks(font=font)
        st.pyplot(fig2)

        fig3, ax3 = plt.subplots()
        sns.heatmap(cos, annot=True, fmt=".2g", linewidths=.5, cmap='YlGnBu')
        ax3.set_title('Cosine similarity')
        plt.xticks(font=font)
        plt.yticks(font=font)
        st.pyplot(fig3)


    if st.button('Calculate the dot product between all prescriptions', key=9):
        dense_dot_df = dot_cos_cal.dot(herb_dense_dataframe=herb_dense_dataframe)
    if st.button('Calculate the cosine similarity between all prescriptions', key=10):
        cos_dot_df = dot_cos_cal.cos(herb_dense_dataframe=herb_dense_dataframe)
    st.write(
        'Reminder: Calculating the dot product and cosine between all prescriptions can take a lot of time and cause the program to crash, depending on your dataset size')
    st.write(
        'Reminder: We recommend that you start the process with the desktop app whenever possible, however, time consuming and system crashes are still possible roadblocks')



