# Fake News Detection Using Machine Learning

## Introduction
Fake news has become a major issue in the digital world due to the rapid spread of information through social media and online platforms. Fake news refers to false or misleading information presented as real news. It can influence public opinion and create confusion among people. Detecting fake news manually is difficult because of the huge amount of content available online.

This project aims to build a Fake News Detection System using Machine Learning techniques. The system analyzes the text of a news article and predicts whether the news is fake or real. By using Natural Language Processing (NLP) and machine learning algorithms, the model learns patterns from labeled datasets and classifies the news accordingly.

## Abstract
The Fake News Detection project uses machine learning to classify news articles as fake or real. A dataset containing fake and real news articles is used to train the model. The text data is converted into numerical form using TF-IDF (Term Frequency–Inverse Document Frequency) vectorization.

A Logistic Regression classifier is trained on the processed data to identify patterns in fake and real news articles. The model is evaluated using accuracy metrics to measure its performance. The trained model is saved and integrated into a simple web interface using Streamlit. Users can enter news text and the system will predict whether the news is fake or real.

This project demonstrates how machine learning and natural language processing can be applied to solve real-world problems such as misinformation detection.

## Tools Used
The following tools and technologies were used in this project:

Programming Language  
Python

Development Environment  
Visual Studio Code (VS Code)

Libraries  
Pandas – for loading and processing data  
NumPy – for numerical operations  
Scikit-learn – for machine learning algorithms and model evaluation  
TF-IDF Vectorizer – for converting text into numerical features  
Streamlit – for building the web interface  
Pickle – for saving and loading trained models

Dataset Source  
Kaggle Fake and Real News Dataset

## Steps Involved in Building the Project

Step 1: Dataset Collection  
The dataset containing fake and real news articles was downloaded from Kaggle.

Step 2: Data Preprocessing  
Both datasets were combined and labeled. Fake news was assigned label 0 and real news was assigned label 1.

Step 3: Feature Extraction  
TF-IDF vectorization was used to convert text data into numerical form so that it could be used by machine learning models.

Step 4: Train-Test Split  
The dataset was divided into training data (80%) and testing data (20%) to evaluate model performance.

Step 5: Model Training  
A Logistic Regression classifier was used to train the machine learning model using the training dataset.

Step 6: Model Evaluation  
The model was tested using the testing dataset and accuracy was calculated to measure performance.

Step 7: Model Saving  
The trained model and TF-IDF vectorizer were saved using Pickle so that they can be reused without retraining.

Step 8: Web Application Development  
A simple web application was built using Streamlit where users can enter news text and get predictions instantly.

## Conclusion
The Fake News Detection system successfully classifies news articles as fake or real using machine learning techniques. The Logistic Regression model combined with TF-IDF feature extraction achieved high accuracy in detecting fake news.

This project demonstrates the practical use of Natural Language Processing and machine learning in solving real-world problems such as misinformation detection. The Streamlit web interface allows users to easily interact with the model and test predictions in real time. Future improvements can include using advanced deep learning models and deploying the system online for wider use.