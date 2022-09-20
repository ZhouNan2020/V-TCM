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
    def list_vect(self):
        list_vect = []
        for index, row in self.iterrows():
            for sen in row:
                sen_row = []
                sent = sen.split(sep=',')
                ','.join(sent)
                for herb in sent:
                    sen_row.append(herb)
                list_vect.append(sen_row)
        return list_vect

class base_frame:
    def __init__(self,txt):
        sentence = ""
        for index, row in txt.iterrows():
            for sen in row:
                sentence = sentence + sen + ','
        herb_list = sentence.split(sep=',')
        self.herb_list = herb_list
        file_dict = dict()
        for index, row in txt.iterrows():
            for sen in row:
                per_vect = []
                ws = sen.split(sep=',')
                for herb in ws:
                    per_vect.append(herb)
                file_dict[index] = per_vect
        self.file_dict = file_dict
        list_vect = []
        for index, row in txt.iterrows():
            for sen in row:
                sen_row = []
                sent = sen.split(sep=',')
                ','.join(sent)
                for herb in sent:
                    sen_row.append(herb)
                list_vect.append(sen_row)
        self.list_vect = list_vect




class count_dict:
    def count_herb(self):
        total_len = len(self.keys())
        return total_len
    def avg_len(self):
        len_herb_list = 0
        for index in self.keys():
            herb_list = self.get(index)
            herb_list = list(set(herb_list))
            len_list = len(herb_list)
            len_herb_list = len_herb_list + len_list
        avg_len = len_herb_list / (len(self.keys()))
        return avg_len
    def dense_frame(self):
        herb_dense_dataframe = pd.DataFrame(columns=['pres_name', 'herb_name'])
        for pres_name in self.keys():
            herb_list = self.get(pres_name)
            pres_name = [pres_name]
            pres_name = pd.DataFrame(pres_name, columns=['pres_name'])
            herb_dense_dataframe = pd.concat([herb_dense_dataframe, pres_name], axis=0, join='outer')
            for herb in herb_list:
                herb_df = pd.DataFrame(columns=['herb_name'])
                herb = [herb]
                herb = pd.DataFrame(herb, columns=['herb_name'])
                herb_df = pd.concat([herb_df, herb], axis=0, join='outer')
                herb_dense_dataframe = pd.concat([herb_dense_dataframe, herb_df], axis=0, join='outer')
        herb_dense_dataframe['count'] = 1
        herb_dense_dataframe['pres_name'] = herb_dense_dataframe['pres_name'].fillna(method='ffill')
        herb_dense_dataframe.dropna(subset=['herb_name'], axis=0, inplace=True, how="any")
        herb_dense_dataframe = herb_dense_dataframe.pivot_table(
            'count', index=herb_dense_dataframe['pres_name'], columns=['herb_name']).fillna(0)
        herb_dense_dataframe = herb_dense_dataframe.astype('int')
        return herb_dense_dataframe
    def tf_idf_dict(self,lexicon,list_vect):
        tf_idf_dict = dict()
        for tf_pres_name in self.keys():
            ini_tf_vect = dict()
            herbs = self.get(tf_pres_name)
            herbs_counts = Counter(herbs)
            for index, value in herbs_counts.items():
                docs_contain_key = 0
                for herb_row in list_vect:
                    if (index in herb_row) == True:
                        docs_contain_key = docs_contain_key + 1
                tf = value / len(lexicon)
                if docs_contain_key != 0:
                    idf = len(self.keys()) / docs_contain_key
                else:
                    idf = 0
                ini_tf_vect[index] = tf * idf
            tf_idf_dict[tf_pres_name] = ini_tf_vect
        return tf_idf_dict


class count_list:
    def count_herb(self):
        Counter_every_herb = Counter(self)
        return Counter_every_herb
    def total_herb_list(self):
        total_herb_list = len(Counter(self))
        return total_herb_list
    def total_herb_word_list(self):
        total_herb_word_list = len(self)
        return total_herb_word_list

class tf_idf:
    def tf_idf_dataframe(self):
        tf_idf_dataframe = pd.DataFrame(columns=['pres_name', 'herb_name'])
        for pres_name in self.keys():
            herb_tf_idf_dict = self.get(pres_name)
            pres_name = [pres_name]
            pres_name = pd.DataFrame(pres_name, columns=['pres_name'])
            tf_idf_dataframe = pd.concat([tf_idf_dataframe, pres_name], axis=0, join='outer')
            for herb_name in herb_tf_idf_dict:
                herb_df = pd.DataFrame(columns=['herb_name', 'herb_tf_idf_value'])
                herb_tf_value = herb_tf_idf_dict.get(herb_name)
                herb_name = [herb_name]
                herb_name = pd.DataFrame(herb_name, columns=['herb_name'])
                herb_df = pd.concat([herb_df, herb_name], axis=0, join='outer')
                herb_tf_value = round(herb_tf_value, 3)
                herb_tf_value = [herb_tf_value]
                herb_tf_value = pd.DataFrame(herb_tf_value, columns=['herb_tf_idf_value'])
                herb_df = pd.concat([herb_df, herb_tf_value], axis=0, join='outer')
                tf_idf_dataframe = pd.concat([tf_idf_dataframe, herb_df], axis=0, join='outer')
        idf_df = cp.copy(tf_idf_dataframe)
        idf_df['pres_name'] = idf_df['pres_name'].fillna(method='ffill')
        idf_df['herb_name'] = idf_df['herb_name'].fillna(method='ffill')
        idf_df.dropna(subset=['herb_tf_idf_value'], axis=0, inplace=True, how="any")
        idf_df = idf_df.pivot_table('herb_tf_idf_value', index=['pres_name'], columns=['herb_name']).fillna(round(0, 3))
        return idf_df
