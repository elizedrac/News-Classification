# News Source Classification Project

## Overview
This project investigates the task of classifying news headlines by source, focusing on distinguishing between Fox News and NBC. Although this is a binary classification problem, it is challenging due to overlapping topics and vocabulary across outlets, making simple keyword-based approaches insufficient.

The project explores how different text representations and model choices affect classification performance, ranging from linear models using bag-of-words features to neural and pretrained transformer-based models.

## Objectives
The main goals of this project are:
- To evaluate how well different modeling approaches can distinguish between news sources using headline text
- To compare simple linear models against more complex sequential and pretrained models
- To analyze the trade-offs between accuracy, computational cost, and model complexity
- To select a final model that balances performance and efficiency

## Project Structure
The main notebook handles:
- Data loading and preprocessing from scraped news URLs
- Creation of multiple preprocessing variants to study their impact on performance
- Training and evaluation of several models:
  - Logistic Regression (baseline)
  - Linear SVM with TF-IDF features
  - LSTM / biLSTM models
  - DistilBERT
- Quantitative evaluation using accuracy
- Qualitative analysis of learned representations via embedding visualizations

In addition, the data collection notebook scrapes headlines using two approaches: extracting headline text directly from URL slugs and scraping headline content from the article webpages themselves. Both data sources are used in experimentation to compare their impact on model performance.

A separate submission notebook and directory contain the finalized pretrained model and evaluation pipeline.

## Models and Findings
We begin with a linear regression–based classifier using TF-IDF features as a baseline. Building on this, we find that a linear SVM substantially improves performance by learning a more robust decision boundary in high-dimensional feature space. Sequential models such as LSTMs are able to capture contextual information and show clearer clustering in embedding space, but do not consistently outperform the SVM on this task. DistilBERT achieves the highest accuracy overall, benefiting from large-scale pretraining, but at a significantly higher computational cost.

Overall, the linear SVM provides the best balance between accuracy, stability, and efficiency for this dataset.

## Conclusions
Our experiments suggest that for short news headlines and a binary source classification task, strong linear models paired with effective feature representations can be competitive with more complex neural approaches. While deeper models offer richer representations, their additional complexity does not always translate into proportional performance gains. This highlights the importance of matching model complexity to the structure and scale of the problem.

## Evaluating the Final Model
The final pretrained SVM model used for submission is contained in the submission notebook and directory. To evaluate the best-performing model, please navigate to the submission folder and follow the instructions provided in the submission README. That README contains step-by-step instructions for installing dependencies and running the evaluation script on the provided dataset.
