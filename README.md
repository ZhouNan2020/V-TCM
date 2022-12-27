# <center> Vector-TCM </center>
Vector-TCM is a platform for traditional Chinese medicine (TCM) data mining. It uses text vector as the basic data form and natural language processing (NLP) technology for TCM data mining.

1. Original data form
The original data must be submitted as a text vector, as shown in the following example:

| index | text|
|-------|---|
|Prescription name|herb name1, herb name2, herb name3, ...|

Note: 1.Only .xlsx format worksheets are supported.  
2.the interval between herb names must be commas in English format. if your original data 
is not like this, it is recommended to use regular expressions to adjust it.  
3.There is no limit to the naming of the title row of the worksheet, but there must be this row.  
4.Provide sample data download, you can imitate the sample data to arrange your own data.  


2. Data preprocessing
<kbd>
![img.png](img.png)
</kbd>
You can select different modules at the top to start different analysis.

2.1 Descriptive statistics
<kbd>
![img_1.png](img_1.png)
</kbd>
The role of this module is no different from conventional descriptive statistics and is used to understand the basic information of the data set.

2.2 Prescription similarity
<kbd>
![img_2.png](img_2.png)
</kbd>
The Prescription similarity module is used to analyze the similarity between prescriptions in the data set. The algorithm of the module is based on the dot product value and cosine value. Due to the limitation of computing power, in the WebAPP version of the Vector-TCM, it is not possible to directly calculate the similarity of prescriptions in the entire data set. You need to select a part of the prescription calculation.
