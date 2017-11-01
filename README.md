# sentiment_analysis
Sentiment Analysis with CNN, RNN and VADER

## Introduction

Sentiment Analysis is a very basic task in NLP. When I learnt Text Analytics at school, I have built 2 sentiment analysis models, one with Naive Bayes, and the other with VADER. The result showed that VADER was better. </br>
I this project, I built three models, RNN, CNN and VADER to do sentiment analysis, and conpare the results.

## Methodology

1. Load dataset from keras.datasets.imdb
> Dataset of 25,000 movies reviews from IMDB, labeled by sentiment (positive/negative). Reviews have been preprocessed, and each review is encoded as a sequence of word indexes (integers). 
2. Preprocess data (pad or truncate to maximum 500 words per review)
3. Build models (include word embedding), train and evaluate the model.
4. For VADER, map indexes back to original words, and feed to VADER. (VADER is a rule and lexicon based method, so no need to train.)

## Result
1. CNN: 0.8864 
2. RNN: 0.8960 (Take much longer time than using CNN)
3. VADR: 0.6984

## Analysis
The result of RNN is better than that of CNN. I think the reason is that RNN keep the locality of the word, while CNN is more like bag-of-word that do not maintain locality info. RNN is very slow becasue Recurrent neural networks scale poorly due to the intrinsic difficulty in parallelizing their state computations. VADER do not perform as well as neural network on large dataset, becasue VADER is rule based and it does not learn, the performance is static. 

## References
http://www.wildml.com/2015/11/understanding-convolutional-neural-networks-for-nlp/ </br>
https://machinelearningmastery.com/predict-sentiment-movie-reviews-using-deep-learning/#comment-418664 </br>
https://machinelearningmastery.com/sequence-classification-lstm-recurrent-neural-networks-python-keras/
