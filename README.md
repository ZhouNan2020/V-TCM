# <center> Vector-TCM </center>
Vector-TCM is a platform for traditional Chinese medicine (TCM) data mining. It uses text vector as the basic data form and natural language processing (NLP) technology for TCM data mining.

**1. Original data form**  

The original data must be submitted as a text vector, as shown in the following example:

| index | text|
|-------|---|
|Prescription name|herb name1, herb name2, herb name3, ...|

Note: 1.1 Only .xlsx format worksheets are supported.  
1.2 the interval between herb names must be commas in English format. if your original data 
is not like this, it is recommended to use regular expressions to adjust it.  
1.3 There is no limit to the naming of the title row of the worksheet, but there must be this row.  
1.4 Provide sample data download, you can imitate the sample data to arrange your own data.  
1.5 You need to clean the data yourself, using regular expressions, manual or any other method you are familiar with.


**2.Start analysis**  

This software includes the following modules: Descriptive statistics, Prescription similarity, General analysis, LSA topic distribution, 
LDiA topic distribution, Word2Vec model.They perform different analysis tasks respectively.

Descriptive statistics: It is used in the same sense as descriptive statistics in any other analytical context, presenting the researcher with the most immediate information in the data set. It is a good starting point for any analysis.
  
Prescription similarity: It is used to calculate the similarity between prescriptions. Two vector distance calculation methods are mainly used here, dot product value and cosine similarity, where the dot product value reflects the absolute quantity of the same herbal medicine used in different prescriptions, and the cosine similarity reflects the relative similarity of different prescriptions, The maximum value is 1, which is exactly the same, and the minimum value is 0, which does not use any of the same herbs.
  
General analysis: This is a method of calculating rare herbal medicines and common herbal medicines in the data set based on the TF-IDF value. The average TF-IDF value of each prescription is obtained by calculating the TF-IDF value of herb, We can use it to tell whether a prescription's herbal composition is extremely rare, or very common. And further learn in this data set, what is the mainstream prescription composition.

LSA topic distribution and LDiA topic distribution: These are two topic classification methods commonly used in Natural Language Processing (NLP) to divide text into different categories. We still use them here to perform classification tasks, the purpose is to provide more alternative methods for the classification of TCM datasets (different from traditional two-step clustering and system clustering)
  
Word2Vec model: The Word2Vec model is based on the principle of word embedding, and constructs vectors differently from Bag-of-Words. This kind of word vector is generated based on context, and can calculate the context relationship of words more accurately. Based on this feature, the Word2Vec model can be used to discover the laws of the data set from a micro perspective, and you can experience it in the program.

