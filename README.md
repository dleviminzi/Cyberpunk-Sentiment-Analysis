The goal of this project is to analyze community sentiment for some AAA game 
releases this year. I am using posts from the 15 days before and after each 
release, which I acquired using the psaw/praw APIs. Then sentiment analysis 
will be performed on all of these posts/comments, using a model built from 
steam reviews. 

Currently the model uses the HuggingFace transformer library. More specifically,
it is using their implementation of BERT(sequence classification).

NOTE: Things are very messy at the moment. They will be cleaned post training.
