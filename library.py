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
# 文件格式转换
class conver:
    "Convert xlsx files"
    #def __int__(self, file):
    #    self.file = file
    def to_xlsx(self):
        buffer = BytesIO()
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            self.file.to_excel(writer, sheet_name='Sheet1')
            writer.save()
            processed_data = buffer.getvalue()
        return processed_data

#%%
## 读取上传数据
#class read_file:
#    def __init__(self, file):
#        self.file = file
#    def read(self):
#
