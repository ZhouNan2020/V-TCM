#%%
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
#%%
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

#%%
# 定义文件转换函数
class conver:
    "Convert xlsx files"
    def __int__(self, file):
        self.file = file
    def to_xlsx(self):
        buffer = BytesIO()
        xls= self.file
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            self.file.to_excel(writer, sheet_name='Sheet1')
            writer.save()
            processed_data = buffer.getvalue()
        return processed_data
#%%
# 读取示例数据
out1 = conver()
out2 = conver()
out1.file = pd.read_excel('English example.xlsx')
out2.file = pd.read_excel('中文示例.xlsx')
eng_exmp = out1.to_xlsx()
chn_exmp = out2.to_xlsx()



#%%
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





