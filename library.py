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
    def to_xlsx(self):
        buffer = BytesIO()
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            self.to_excel(writer, sheet_name='Sheet1')
            writer.save()
            processed_data = buffer.getvalue()
        return processed_data

#%%
## 读取上传数据
class read_file:
   def read(self):
       txt = pd.read_excel(self)
       txt = pd.DataFrame(txt)
       col = txt.columns
       txt = txt.set_index(col[0])
       return txt

## 计数对象
class format:
    len_herb_list = 0
    def herb_list(self):
        sentence = ""
        for index, row in self.iterrows():
            for sen in row:
                sentence = sentence + sen + ','
        herb_list = sentence.split(sep=',')
        return herb_list
    def file_dict(self):
        file_dict = dict()
        for index, row in self.iterrows():
            for sen in row:
                per_vect = []
                ws = sen.split(sep=',')
                for herb in ws:
                    per_vect.append(herb)
                file_dict[index] = per_vect
        return file_dict


class count_f:
    def count_herb(self):
        total_len = len(self.keys())
        return total_len
    def avg_len(self):
        len_herb_list = 0
        for index in self():
            herb_list = self.get(index)
            herb_list = list(set(herb_list))
            len_list = len(herb_list)
            len_herb_list = len_herb_list + len_list
        avg_len = len_herb_list / (len(self.keys()))
        return avg_len


class count_h:
    def count_herb(self):
        Counter_every_herb = Counter(self)
        return Counter_every_herb
    def total_herb_list(self):
        total_herb_list = len(Counter(self))
        return total_herb_list
    def total_herb_word_list(self):
        total_herb_word_list = len(self)
        return total_herb_word_list
