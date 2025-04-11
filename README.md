# Crime Detection and Prevention System

<div>
  <p>Watch the project walkthrough video to understand the system's capabilities and implementation details.</p>
  <a href="https://drive.google.com/file/d/1stmzkmcMO6zBVvZkSiEwg504OaFKH-I_/view?usp=sharing">
    <button style="background-color: #4CAF50; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; font-size: 16px; margin: 10px 0;">
      Project Walkthrough Video
    </button>
  </a>
</div>

## Project Overview

This project implements an advanced crime detection and prevention system using machine learning and deep learning techniques. The system analyzes text conversations to identify potential criminal activities and provides appropriate responses.

## System Architecture

### Crime Project Flow

<div align="center">
  <img src="assets/CrimeProjectFlow-3.png" alt="Crime Project Flow" width="90%">
  <p><em>Figure 3: Overall system architecture and data flow for crime detection</em></p>
</div>

### SEER-BERT Architecture

<div align="center">
  <img src="assets/SEER-BERT-2.png" alt="SEER-BERT Architecture" width="85%">
  <p><em>Figure 4: SEER-BERT architecture for enhanced text classification</em></p>
</div>

## Model Performance

We evaluated multiple machine learning and deep learning models to find the optimal solution for crime detection.

### Machine Learning Models

The following image shows the TOPSIS scores for various machine learning models:

<div align="center">
  <img src="assets/TOPSIS_scores_Machine Learning.png" alt="Machine Learning Models TOPSIS Scores" width="80%">
  <p><em>Figure 1: Performance comparison of different machine learning models using TOPSIS analysis</em></p>
</div>

### Deep Learning Models

Similarly, we evaluated deep learning models:

<div align="center">
  <img src="assets/TOPSIS_scores_Deep Learning.png" alt="Deep Learning Models TOPSIS Scores" width="80%">
  <p><em>Figure 2: Performance comparison of different deep learning architectures using TOPSIS analysis</em></p>
</div>

## Product Flows

### Product 1 Flow

<div align="center">
  <img src="assets/Product1_Flow.png" alt="Product 1 Flow" width="80%">
  <p><em>Figure 5: Implementation flow for the first product variant</em></p>
</div>

### Product 2 Flow

<div align="center">
  <img src="assets/Product2_Flow.png" alt="Product 2 Flow" width="80%">
  <p><em>Figure 6: Implementation flow for the second product variant</em></p>
</div>

## Requirements

The project requires several Python libraries as specified in the requirements.txt file, including:
- numpy
- pandas
- matplotlib
- seaborn
- nltk
- scikit-learn
- tensorflow
- tensorflow-hub
- transformers

## Results

Our best model achieved over 56% accuracy on the test dataset, which is significant considering the complexity of crime detection from text conversations.

## Future Work

- Implement real-time monitoring capabilities
- Expand the dataset to include more diverse criminal conversation patterns
- Integrate with messaging platforms for proactive crime prevention
