# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager
import seaborn as sns
import streamlit as st
import altair as alt
import sys
sys.path.append('.')
# %%
# Path: V-TCM\streamlit_app.py
# 全局设置
# streamlit
st.title('Vector-TCM V1.0.0')
tab1, tab2, tab3, tab4, tab5, tab6,tab7 = st.tabs(
    ["Descriptive statistics", "Difference between prescriptions", "Characteristic herbs and universal herbs",
     "LSA topic model", "LDiA topic model","Word2Vec model(beta)",
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
from library import conver, read_file,tf_idf, base_frame,dot_cos_cal,svd,ldia,alt_plot

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

    st.write('Note: Please delete the number in column 1 after the download is complete.You can understand the workflow of this program by uploading sample data.')
    st.write(
        'Note: When the program is running, there will be a little man doing sports in the upper right corner of the web page,don\`t refresh this page or do anything else until he stops.')
#%%
# 测试文件
# file=pd.read_csv("English example.csv")
#%%
def file_pre(file):
    if file!=None:
        txt = read_file.read(file)
    else:
        txt = pd.DataFrame(out1)
        col = txt.columns
        txt = txt.set_index(col[0])
    return txt
txt=file_pre(file)
# %%


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
idf_df = tf_idf.tf_idf_dataframe(tf_idf_dict)



with tab1:
    st.write('1.It involves a total of {} types of herbs'.format(total_herb_list))
    st.write('2.The total frequency of herbal use has reached {}'.format(total_herb_word_list))
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
        full_common_herb = full_common_data.copy()
        full_common_herb = conver(full_common_herb)
        st.download_button(
            label="Download full herb frequency data",
            data=full_common_herb.file,
            file_name='full_common_data.xlsx',
            mime='xlsx')
    # 密集矩阵下载
    if herb_dense_dataframe.empty == False:
        dense_dataframe = herb_dense_dataframe.copy()
        dense_dataframe = conver(dense_dataframe)
        st.download_button(
            label='Download dense matrix',
            data=dense_dataframe.file,
            file_name='dense matrix.xlsx')
    # tf-idf矩阵下载
    if idf_df.empty == False:
        tf_idf_matrix = conver(idf_df)
        st.download_button(
            label='Download tf_idf_matrix',
            data=tf_idf_matrix.file,
            file_name='tf_idf_matrix.xlsx')
        #close=st.button('Terminate descriptive statistics', key=3)
        #st.write('To ensure that the WebApp retains enough memory, try to terminate unneeded modules when appropriate')
        #if close:
        #    st.experimental_memo.clear()
#%%

with tab2:
    st.write('Focus on dot product and cosine similarity for a specific prescription')
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

#
#if st.button('Calculate the dot product between all prescriptions', key=9):
#    dense_dot_df = dot_cos_cal.dot(herb_dense_dataframe=herb_dense_dataframe)
#    dense_dot = conver(dense_dot_df)
#    st.download_button(
#        label='Download dot_product_matrix',
#        data=dense_dot.file,
#        file_name='dense dot product.xlsx')
#if st.button('Calculate the cosine similarity between all prescriptions', key=10):
#    cos_dot_df = dot_cos_cal.cos(herb_dense_dataframe=herb_dense_dataframe)
#    cos_dot = conver(cos_dot_df)
#    st.download_button(
#        label='Download cosine similarity matrix',
#        data=cos_dot.file,
#        file_name='cosine similarity.xlsx')
#st.write(
#    'Reminder: Calculating the dot product and cosine between all prescriptions can take a lot of time and cause the program to crash, depending on your dataset size')
#st.write(
#    'Reminder: We recommend that you start the process with the desktop app whenever possible, however, time consuming and system crashes are still possible roadblocks')


with tab3:
    tf_idf_sort = f.tf_idf_sort(idf_df=idf_df)
    st.success(
        "TF-IDF value calculation has been completed in the background.")
    num7 = st.select_slider(
        'please select the number of prescriptions you want to display.',
        options=range(1, 50, 1), key=7)
    idf_button_con = st.button('Continue', key=13)
    if idf_button_con:

            st.write('{} prescriptions that use more common herbs'.format(num7))
            st.table(tf_idf_sort.head(num7))


            st.write('{} prescriptions that use more rare herbs'.format(num7))
            st.table(tf_idf_sort.tail(num7))

    if tf_idf_sort.empty == False:
        tf_idf_matrix = conver(tf_idf_sort)
        st.download_button(
            label='Download tf_idf_matrix',
            data=tf_idf_matrix.file,
            file_name='tf_idf_matrix.xlsx')

with tab4:
    st.subheader('Topic classification based on Latent Semantic Analysis (LSA)')
    num4 = st.select_slider(
        'Please select the number of themes you wish to try',
        options=range(1, 100, 1), key=5)
    svd_button_pressed = st.button('Launch', key=5)
    if svd_button_pressed == True:
        if num4 < len(txt.index):
            lsa_topic=svd.svd_topic(num=num4,df=idf_df)
            st.line_chart(lsa_topic)
            with st.expander("See explanation"):
                st.write('Explained variance ratio: The amount of information extracted by the topic can be understood as the weight of different topics. The higher the weight, the more information the topic can extract from the document. The lower the weight, the less information the topic can extract from the document. The weight of a topic is the square root of the sum of the square of the singular values of the topic. The weight of a topic is the square root of the sum of the square of the singular values of the topic.')
                st.write('Cumulative explained variance ratio: Under the current number of topics, the total amount of information extracted by all topics, this indicator needs to be at least greater than 50%')
                st.write('Singular values: The singular values of the topic are the square root of the sum of the square of the singular values of the topic,determines the number of topics when the downtrend in this value begins to flatten')
        else:
            st.write(
                'Please select a smaller number,you cannot choose a number larger than the number of prescriptions in the dataset')

    with st.expander("Confirm the number of LSA classifications"):
        st.write(
            'If you confirm the number of topics you want to get based on the line chart, please fill in the blank and click "Continue" to get the specific topic matrix')
        num4_con = st.number_input('Enter the number of topics you have confirmed', step=1, format='%d', key=10)
        svd_button_con = st.button('Continue', key=10)
        if svd_button_con:
            pres_svd_topic = (svd.svd_confirm(num=num4_con, df=idf_df))[0]
            herb_svd_weight = (svd.svd_confirm(num=num4_con, df=idf_df))[1]
            st.table(pres_svd_topic.head(5))
            st.table(herb_svd_weight.head(5))
            st.success('The topic classification based on LSA is done,you can download this matrix')
            if pres_svd_topic.empty == False and herb_svd_weight.empty == False:
                pres_svd_topic = conver(pres_svd_topic)
                st.download_button(
                    label='Download svd topic matrix',
                    data=pres_svd_topic.file,
                    file_name='svd topic.xlsx')
                herb_svd_weight = conver(herb_svd_weight)
                st.download_button(
                    label='Download svd weight matrix',
                    data=herb_svd_weight.file,
                    file_name='svd herb weight.xlsx')

with tab5:
    st.subheader('Topic classification based on Latent Dirichlet Distribution (LDiA)')
    num5 = st.select_slider(
        'Please select the maximum number of themes you wish to try',
        options=range(1, 100, 1), key=6)
    ldia_button_pressed = st.button('Launch', key=10)
    if ldia_button_pressed == True:
        ldia_topic = ldia.ldia_topic(num=num5, df=herb_dense_dataframe)
        st.line_chart(ldia_topic)
        with st.expander("See explanation"):
            st.write(
                'Perplexity is an important reference indicator for determining the number of topics in the LDiA model. When perplexity is at its lowest point, we can take the value here as the number of reserved topics')
    st.write(
        'If you confirm the number of topics you want to get based on the line chart, please fill in the blank and click "Continue" to get the specific topic matrix')
    num5_con = st.number_input('Enter the number of topics you have confirmed', step=1, format='%d', key=11)
    ldia_button_con = st.button('Continue', key=11)
    if ldia_button_con:
        components_herb = (ldia.ldia_confirm(num=num5_con, df=herb_dense_dataframe))[0]
        components_pres = (ldia.ldia_confirm(num=num5_con, df=herb_dense_dataframe))[1]
        st.table(components_pres.head(5))
        st.table(components_herb.head(5))
        components_herb = conver(components_herb)
        components_pres = conver(components_pres)
        st.download_button(
            label='Download ldia topic matrix',
            data=components_pres.file,
            file_name='ldia topic.xlsx')

        st.download_button(
            label='Download ldia herb weight matrix',
            data=components_herb.file,
            file_name='ldia herb weight.xlsx')

        st.success('The topic classification based on LDiA is done,you can download this matrix in the "Download" tab')

with tab6:
    model=f.w2v(avg_len=avg_len)
    pca_matrix = alt_plot.alt_plot(model=model,full_common_data=full_common_data)
    w2v_data = alt.Chart(pca_matrix).mark_circle().encode(
        x='Vector 1', y='Vector 2', size='count', color='count', tooltip=['name', 'count']).interactive()
    st.altair_chart(w2v_data, use_container_width=True)

    op_w2v = st.radio('What function do you hope to achieve?',
                      ('Similar herbal search', 'Herbal analogy', 'Compatibility assessment'))
    if op_w2v == 'Similar herbal search':
        st.subheader('Similar herbal search')
        st.write('Please enter the name of the herb you want to search')
        search_herb = st.text_input('Enter the herb', key=12)
        search_button = st.button('Search', key=12)
        if search_button:
            feed_herb = model.wv.most_similar(positive=[search_herb], topn=10)
            feed_herb = pd.DataFrame(feed_herb, columns=['herb', 'similarity'])
            st.table(feed_herb)
    if op_w2v=='Herbal analogy':
        st.subheader('Herbal analogy')
        st.write('Please enter the name of the herb you want to search')
        st.write('If you want to directly compare the similarity of two herbs, please enter the herb name in Herb 1 and Herb 2')
        search_herb_p1 = st.text_input('Herb 1',key=13)
        search_herb_n1 = st.text_input('Herb 2',key=14)
        st.write('If you want to use the method of analogy to explore the law of paired combination of herbs, please also fill in Analogy Item')
        search_herb_p2 = st.text_input('Analogy Item',key=13)

        search_button = st.button('Search', key=13)

        if search_button:
            if len(search_herb_p2)==0:
                feed_herb=model.wv.similarity(search_herb_p1,search_herb_n1)
                st.write('The similarity of {} and {} is {}'.format(search_herb_p1,search_herb_p2,feed_herb))
            if len(search_herb_p2)>0:
                feed_herb=model.wv.most_similar(positive=[search_herb_p2,search_herb_p1],negative=[search_herb_n1],topn=10)
                feed_herb=pd.DataFrame(feed_herb,columns=['herb','vector_similarity'])
                feed_herb=feed_herb.sort_values(by='vector_similarity',ascending=False)
                best_match=feed_herb.iloc[0,0]
                st.write('Imitating the combination rule of {} and {}, {}is a more matching herb with {}'.format(search_herb_p1,search_herb_n1,best_match,search_herb_p2))
                st.write('Alternative herbs that can be paired with {} in the table below'.format(search_herb_p2))
                st.table(feed_herb)

    if op_w2v=='Compatibility assessment':
        st.subheader('Compatibility assessment')
        st.write('Please enter the herb list you want to assessment')
        input_herb = st.text_input('Use "," (English format) to separate the herbs',key=15)
        if st.button('Assessment', key=15):
            if len(input_herb)>0:
                input_herb_list = input_herb.split(',')
                feed_herb=model.wv.doesnt_match(input_herb_list)
                st.write('In this list of herbs, {} has the farthest vector distance from other herbs. Please evaluate whether the use of {} is reasonable in combination with the needs of clinical practice.'.format(feed_herb,feed_herb))



with tab7:
    st.write('Author information:')
    st.write('Name: Zhou Nan')
    st.write('Current situation: PhD student,Universiti Tunku Abdul Rahman(UTAR)')
    st.write('Mail_1:zhounan@1utar.my')
    st.write('Mail_2:zhounan2020@foxmail.com')
    st.write(
        'Due to Streamlit\'s IO capability limitations, this web application may not perform well on large data sets. If you think this web application cannot meet your needs, you can visit GitHub Public repository(https://github.com/ZhouNan2020/VectorTCM.git) to obtain PC platform applications, or directly contact the author by email for help.')



