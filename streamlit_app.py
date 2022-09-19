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
fontsize = 30

# %%
from library import conver, read_file, format, count_h,count_f

# %%
# 读取示例数据
out1 = pd.read_excel('English example.xlsx')
out2 = pd.read_excel('中文示例.xlsx')
eng_exmp = conver.to_xlsx(out1)
chn_exmp = conver.to_xlsx(out2)
# %%
# 侧栏上传文件区域
with st.sidebar:
    file = st.file_uploader("Click “Browse files” to upload files", type=["xlsx"])
    st.write('Please upload a file no larger than 200MB')
    st.write('The file must be a .xlsx file')
    st.download_button('Download sample data in English', data=eng_exmp, file_name='sample data in English.xlsx',
                       )
    st.download_button('下载中文示例数据', data=chn_exmp, file_name='中文示例数据.xlsx')
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
    return txt
txt=file_pre(file)
# %%
st.write('You can use the cursor keys "←" and "→" to see more tags')
herb_list = format.herb_list(txt)
file_dict = format.file_dict(txt)

total_herb_list = count_h.total_herb_list(herb_list)
total_herb_word_list = count_h.total_herb_word_list(herb_list)
avg_len = count_f.avg_len(file_dict)
count_herb = count_h.count_herb(file_dict)


with tab1:
    st.write('1.The total number of different herbs: ', total_herb_list)
    st.write('2.The total number of herbs is:', total_herb_word_list)
    st.write('3.The average length of prescription: ', round(avg_len, 0))
    st.write('4.The most common herb')
    num1 = st.select_slider(
        'How many herbs do you need to display by frequency?',
        options=range(1, 50, 1), key=1)

    most_common_herb1 = (count_herb).most_common(num1)
    most_common_herb1 = pd.DataFrame(most_common_herb1, columns=['herb', 'count'])





