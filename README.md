# Sentiment Analysis Model

## Summary | Problem Statement

### Summary :
NLP project to target real word customer's issue based on there feedbacks.

### Problem Statement :
Any business is obliged to understand clients — their needs, their opinions, their satisfaction with the product. In case of large web-based companies we need to analyse hundreds of thousands or even millions of opinions to different products, and simply searching for pre-defined “good” or “bad” words in the comments is not enough. With rise of machine learning, in particular, deep neural networks, sentiment analysis — the problem of understanding the emotional tone of a text has been solved with very high accuracy.The problem statement for this project was client want to segrigate the customers based on there reviews and raw data so they can target a set of customers for further analysis.

## Table of Contents
  - [Analysis of Text Data](#Analysis-of-Text-Data)
  - [Data Cleaning](#data-cleaning)
  - [Data Preprocessing](#data-preprocessing)
  - [Model Selection](#model-selection-and-evaluations)
  - [Deal with imbalanced data](#deal-with-imbalanced-data)
  - [Flask and Api](#flask-and-api)
  - [Api Testing](#api-testing)

## Basic Analysis
### Analysis of Text Data
 
>>
    * Checking average of character each sentence have.
![foxdemo](https://github.com/SarkarPriyanshu/Sentiment_Predict/blob/master/Graphs/img1.png?raw=true)
    
    * The average word length ranges
![foxdemo](https://github.com/SarkarPriyanshu/Sentiment_Predict/blob/master/Graphs/img2.png?raw=true)

    * the number of words appearing in each news highlights.
![foxdemo](https://github.com/SarkarPriyanshu/Sentiment_Predict/blob/master/Graphs/img3.png?raw=true)    
    
    * Which stopwords occur frequently in our text, let’s inspect which words other than these stopwords occur frequently?
![foxdemo](https://github.com/SarkarPriyanshu/Sentiment_Predict/blob/master/Graphs/img4.png?raw=true)
    
    * Top 50 most occuring words
![foxdemo](https://github.com/SarkarPriyanshu/Sentiment_Predict/blob/master/Graphs/img5.png?raw=true)
    
    * Bigram and trigram 
![foxdemo](https://github.com/SarkarPriyanshu/Sentiment_Predict/blob/master/Graphs/img6.png?raw=true)
![foxdemo](https://github.com/SarkarPriyanshu/Sentiment_Predict/blob/master/Graphs/img6.1.png?raw=true)


## Data Cleaning
>   After analying have to deal with multiple sub problems
> ### Sub Problems we faced working with this data
>>
    * Removing special charecters, symbols and many noisy raw text data.
    * Need to remove stopwords.
    * Converting text to lowercase for narmalisation 
    * Converting emogi if present to utf-8 format.
>

## Data Preprocessing
### Preprocessing steps

>   After analying have to deal with multiple sub problems
> ### Steps we follow in preprocessing steps
>>
    * Normalisation of text
    * Tokenize words and further clean-up text
    * Removing stopwords
    * Applying bigram and trigram
    * Stemming and lemmatisation
    * Sentiment Predictor using flair library

### Observation after preprocessing
>>
![foxdemo](https://github.com/SarkarPriyanshu/Sentiment_Predict/blob/master/Graphs/img7.png?raw=true)
    
    * Data we produce using flair library was imbalanced so we have applied various methods later for better predictive model.

## Model Selection and Evaluations
> ### Logistic Model and evaluation
>>
        Observation Logistic Regression:
![foxdemo](https://github.com/SarkarPriyanshu/Sentiment_Predict/blob/master/Graphs/img8.png?raw=true)
    
    * G-Mean test: 0.39774579483764383
    * Dominance test: 0.830462250300307
    * ROC-AUC test: 0.8038083374115763

![foxdemo](https://github.com/SarkarPriyanshu/Sentiment_Predict/blob/master/Graphs/img9.png?raw=true)
        
> ### GaussianNB and evaluation
>>
     Observation Support GaussianNB:
![foxdemo](https://github.com/SarkarPriyanshu/Sentiment_Predict/blob/master/Graphs/img10.png?raw=true)
     
        * G-Mean test: 0.20328966895744396
        * Dominance test: 0.9563242425590603
        * ROC-AUC test: 0.7552564844062819
            
![foxdemo](https://github.com/SarkarPriyanshu/Sentiment_Predict/blob/master/Graphs/img11.png?raw=true)

![foxdemo](https://github.com/SarkarPriyanshu/Sentiment_Predict/blob/master/Graphs/img12.png?raw=true)
  
## Deal with imbalanced data
> ### Under Samplying Techniques 
>>
    * Random Under samplying
    * NearMiss
        * NearMiss version 1
        * NearMiss version 2
        * NearMiss version 3

> ### Over Samplying Techniques
    * Random Over Samplying
    * Random Over Samplying with smoothing
    * SMOTE
    * ADASYN
    
> ### Observation
    * From the obove observation we got logistic regression is best performer for predicting sentiment for our dataset and NearMiss under sampling is giving best result for handling unbalance dataset
    
![foxdemo](https://github.com/SarkarPriyanshu/Sentiment_Predict/blob/master/Graphs/imglast.png?raw=true)
![foxdemo](https://github.com/SarkarPriyanshu/Sentiment_Predict/blob/master/Graphs/img13.png?raw=true)
    
    * Precision:  [0.84778226 0.78287732]
    * Recall:  [0.77368905 0.85452794]
    * f-score:  [0.80904281 0.81713496]
    * Support:  [1087 1038]
    * ROC-AUC test: 0.8877680345580012
    
## Flask and Api
>>
  * [Server code](https://github.com/SarkarPriyanshu/Sentiment_Predict/tree/master/Server) using [Streamlit](https://streamlit.io/)
    
## Api Testing
>>
  * For Testing and documentaion we used [Postman](https://www.postman.com/)
  * For document [Click Here](https://github.com/SarkarPriyanshu/CreditScore/blob/main/Credit%20score%20classification%20APIs.postman_collection.json)

### [Sponsor By me]([https://github.com/SarkarPriyanshu])

## Technologies Used

- [Streamlit](https://streamlit.io/), for backend
- [Jupyter Notebook](https://jupyter.org/), for Cleaning, Model traning & Evaluation
- [Postman](https://www.postman.com/), for Testing & Documentaions

## Chao

A passion project by Priyanshu Sarkar here is my [Github](https://github.com/SarkarPriyanshu) and  [CodeSandBox](https://codesandbox.io/u/SarkarPriyanshu)
