#%%
import numpy as np
import pandas as pd
from io import BytesIO
from sklearn.decomposition import PCA, TruncatedSVD
from collections import Counter, OrderedDict
import copy as cp
from sklearn.decomposition import LatentDirichletAllocation as LDiA
import gensim

#%%
# 文件格式转换
class conver:
    "Convert xlsx files"
    def __init__(self,file):
        buffer = BytesIO()
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            file.to_excel(writer, sheet_name='Sheet1')
            writer.save()
            processed_data = buffer.getvalue()
        self.file = processed_data


#%%
## 读取上传数据
class read_file:
   def read(self):
       txt = pd.read_excel(self)
       txt = pd.DataFrame(txt)
       col = txt.columns
       txt = txt.set_index(col[0])
       return txt

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
    def count_herb(self):
        total_len = len(self.file_dict.keys())
        return total_len
    def avg_len(self):
        len_herb_list = 0
        for index in self.file_dict.keys():
            herb_list = self.file_dict.get(index)
            herb_list = list(set(herb_list))
            len_list = len(herb_list)
            len_herb_list = len_herb_list + len_list
        avg_len = len_herb_list / (len(self.file_dict.keys()))
        return avg_len
    def herb_dense_dataframe(self):
        herb_dense_dataframe = pd.DataFrame(columns=['pres_name', 'herb_name'])
        for pres_name in self.file_dict.keys():
            herb_list = self.file_dict.get(pres_name)
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
    def lexicon(self):
        lexicon=sorted(set(self.herb_list))
        return lexicon
    def tf_idf_dict(self,lexicon):
        tf_idf_dict = dict()
        for tf_pres_name in self.file_dict.keys():
            ini_tf_vect = dict()
            herbs = self.file_dict.get(tf_pres_name)
            herbs_counts = Counter(herbs)
            for index, value in herbs_counts.items():
                docs_contain_key = 0
                for herb_row in self.list_vect:
                    if (index in herb_row) == True:
                        docs_contain_key = docs_contain_key + 1
                tf = value / len(lexicon)
                if docs_contain_key != 0:
                    idf = len(self.file_dict.keys()) / docs_contain_key
                else:
                    idf = 0
                ini_tf_vect[index] = tf * idf
            tf_idf_dict[tf_pres_name] = ini_tf_vect
        return tf_idf_dict
    def count_herb(self):
        Counter_every_herb = Counter(self.herb_list)
        return Counter_every_herb
    def total_herb_list(self):
        total_herb_list = len(Counter(self.herb_list))
        return total_herb_list
    def total_herb_word_list(self):
        total_herb_word_list = len(self.herb_list)
        return total_herb_word_list
    def w2v(self,avg_len):
        model = gensim.models.Word2Vec(self.list_vect, sg=0, min_count=1, vector_size=100, window=avg_len)
        return model
    def tf_idf_sort(self,idf_df):
        sum_table=pd.DataFrame(idf_df['tf_idf_sum'])
        tf_idf_sort_dict=dict()
        for index, row in sum_table.iterrows():
            for i in row:
                herb_list = self.file_dict.get(index)
                len_pres = len(herb_list)
                mean_tf_idf = i / len_pres
                tf_idf_sort_dict[index] = mean_tf_idf
        tf_idf_mean_value=pd.DataFrame.from_dict(tf_idf_sort_dict, orient='index')
        tf_idf_mean_value.columns=['tf_idf_mean']
        tf_idf_herb_list=pd.DataFrame.from_dict(self.file_dict, orient='index')
        tf_idf_mean_value_herb_list=pd.concat([tf_idf_mean_value, tf_idf_herb_list], axis=1)
        tf_idf_sort = tf_idf_mean_value_herb_list.sort_values(by=['tf_idf_mean'], ascending=False)
        return tf_idf_sort




class dot_cos_cal:
    def dot_cos(select_result,herb_dense_dataframe):
        dot_df = pd.DataFrame()
        cos_df = pd.DataFrame()
        for res1 in select_result:
            dot_matrix = pd.DataFrame()
            cos_matrix = pd.DataFrame()
            for res2 in select_result:
                vec1 = herb_dense_dataframe.loc[res1]
                vec2 = herb_dense_dataframe.loc[res2]
                dot = np.dot(vec1, vec2)
                cos = dot / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
                dot_matrix = dot_matrix.join(pd.DataFrame(dot,columns=[res2],index=[res1]), how='right')
                cos_matrix = cos_matrix.join(pd.DataFrame(cos,columns=[res2],index=[res1]), how='right')
            dot_df = pd.concat([dot_df, dot_matrix], axis=0, join="outer")
            cos_df = pd.concat([cos_df, cos_matrix], axis=0, join="outer")
        return dot_df, cos_df
    def dot(herb_dense_dataframe):
        dot_df = pd.DataFrame()
        for res1 in herb_dense_dataframe.index:
            dot_matrix = pd.DataFrame()
            for res2 in herb_dense_dataframe.index:
                vec1 = herb_dense_dataframe.loc[res1]
                vec2 = herb_dense_dataframe.loc[res2]
                dot = np.dot(vec1, vec2)
                dot_matrix = dot_matrix.join(pd.DataFrame(dot,columns=[res2],index=[res1]), how='right')
            dot_df = pd.concat([dot_df, dot_matrix], axis=0, join="outer")
        return dot_df
    def cos(herb_dense_dataframe):
        cos_df = pd.DataFrame()
        for res1 in herb_dense_dataframe.index:
            cos_matrix = pd.DataFrame()
            for res2 in herb_dense_dataframe.index:
                vec1 = herb_dense_dataframe.loc[res1]
                vec2 = herb_dense_dataframe.loc[res2]
                dot = np.dot(vec1, vec2)
                cos = dot / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
                cos_matrix = cos_matrix.join(pd.DataFrame(cos,columns=[res2],index=[res1]), how='right')
            cos_df = pd.concat([cos_df, cos_matrix], axis=0, join="outer")
        return cos_df


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
        idf_df['tf_idf_sum'] = idf_df.apply(lambda x: x.sum(),axis=1)
        return idf_df



class svd:
    def svd_topic(df,num):
        df = df.T
        svd = TruncatedSVD(n_components=num, n_iter=10, random_state=123)
        svd_model = svd.fit(df)
        svd_topic = svd.transform(df)
        explvara_list = list(svd.explained_variance_ratio_)
        sing = svd_model.singular_values_
        expl_cum = np.cumsum(explvara_list)
        lsa_topic = pd.DataFrame(
            {'topic': range(1, num + 1), 'explained_variance': explvara_list, 'cumulative_explained_variance': expl_cum,
             'singular_values': sing})
        lsa_topic = lsa_topic.set_index('topic')
        return lsa_topic
    def svd_confirm(df,num):
        df = df.T
        svd = TruncatedSVD(n_components=num, n_iter=10, random_state=123)
        svd_model = svd.fit(df)
        svd_topic = svd.transform(df)
        columns = ['topic{}'.format(i) for i in range(svd.n_components)]
        pres_svd_topic = pd.DataFrame(svd_topic, columns=columns, index=df.index)
        herb_svd_weight = pd.DataFrame(svd.components_, columns=df.columns,
                                       index=columns)
        herb_svd_weight = herb_svd_weight.T
        return pres_svd_topic, herb_svd_weight

class ldia:
    def ldia_topic(df,num):
        x = []
        y = []
        for i in range(1, num + 1):
            ldia = LDiA(n_components=i, learning_method='batch', evaluate_every=1, verbose=1, max_iter=50,
                        random_state=123)
            ldia = ldia.fit(df)
            plex = ldia.perplexity(df)
            x.append(i)
            y.append(plex)
        ldia_topic = pd.DataFrame(y, columns=['perplexity'], index=x)
        return ldia_topic
    def ldia_confirm(df,num):
        ldia = LDiA(n_components=num, learning_method='batch', evaluate_every=1, verbose=1, max_iter=50,
                    random_state=123)
        ldia = ldia.fit(df)
        columns = ['topic{}'.format(i) for i in range(ldia.n_components)]
        components_herb = pd.DataFrame(ldia.components_.T, index=df.columns, columns=columns)
        components_pres = ldia.transform(df)
        components_pres = pd.DataFrame(components_pres, index=df.index, columns=columns)
        return components_herb, components_pres

class alt_plot:
    def alt_plot(model,full_common_data):
        a = pd.DataFrame(model.wv.index_to_key, columns=['name'])
        b = pd.DataFrame(model.wv.vectors, index=a['name'])
        pca = PCA(n_components=2, random_state=123)
        pca = pca.fit(b)
        pca_vectr = pca.transform(b)
        full_common_data = full_common_data.set_index('herb')
        columns = ['topic{}'.format(i) for i in range(pca.n_components)]
        pca_topic = pd.DataFrame(pca_vectr, columns=columns, index=b.index)
        pca_matrix = pca_topic.round(3)
        pca_matrix = pca_matrix.join(full_common_data)
        pca_matrix = pca_matrix.reset_index()
        pca_matrix.rename(columns={'topic0':'Vector 1','topic1':'Vector 2'},inplace=True)
        return pca_matrix





