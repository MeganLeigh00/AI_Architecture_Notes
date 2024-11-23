# AI Model Architectures

## 1. Linear Models
- **Linear Regression**: Predicts a continuous output by fitting a linear relationship between input features and the target.
- **Logistic Regression**: Used for binary classification by applying a logistic function to model probabilities.

## 2. Decision Trees
- **Decision Tree**: A tree-like structure where decisions are made based on feature splits, useful for both classification and regression.
- **Random Forest**: An ensemble of decision trees that improves accuracy by averaging multiple trees' predictions.
- **Gradient Boosting**: Sequentially builds trees to minimize errors by adjusting for mistakes made by previous trees.

## 3. Support Vector Machines (SVM)
- **SVM**: Finds the hyperplane that best separates data into classes, useful for both linear and non-linear classification using kernels.

## 4. k-Nearest Neighbors (k-NN)
- **k-NN**: A simple algorithm that classifies data points based on the majority class of the k-nearest data points in feature space.

## 5. Neural Networks (NN)
- **Feedforward Neural Networks (FNN)**: A basic neural network where information flows in one direction from input to output.
- **Convolutional Neural Networks (CNN)**: Designed for image processing, CNNs use convolutional layers to automatically detect spatial features.
- **Recurrent Neural Networks (RNN)**: Suitable for sequence data (e.g., time series, text), RNNs use loops to retain information from previous steps.
- **Long Short-Term Memory (LSTM)**: A type of RNN designed to retain information over long sequences, addressing the vanishing gradient problem.

## 6. Transformers
- **Transformer**: Used in NLP, transformers handle sequences with attention mechanisms that weigh the importance of each input element, enabling parallel processing of data.
- **BERT**: A transformer model pre-trained for bidirectional language understanding, commonly used for NLP tasks like text classification or Q&A.
- **GPT**: A transformer-based model specialized in text generation, using a unidirectional (left-to-right) language model.

## 7. Autoencoders
- **Autoencoder**: A neural network used for unsupervised learning, compressing data into a latent space (encoder) and reconstructing it (decoder).
- **Variational Autoencoder (VAE)**: A probabilistic variant of autoencoders, useful for generating new data samples by learning the distribution of the input data.

## 8. Generative Adversarial Networks (GANs)
- **GAN**: Composed of two neural networks (generator and discriminator) that compete, with the generator creating data and the discriminator distinguishing real from generated data, often used for image generation.

## 9. Reinforcement Learning (RL)
- **Q-Learning**: A model-free RL method that learns the value of actions in states to maximize cumulative reward.
- **Deep Q-Network (DQN)**: Uses a neural network to approximate Q-values, enabling RL in high-dimensional spaces like games.

---

# Machine Learning Techniques

## Supervised Learning Techniques
- **Linear Regression**
  - Example: Predicting house prices based on square footage.
  - Metrics: Mean Squared Error (MSE), R-squared (R²).
- **Polynomial Regression**
  - Example: Modeling the growth of a company’s revenue over time (non-linear).
  - Metrics: Mean Absolute Error (MAE), Root Mean Squared Error (RMSE).
- **Logistic Regression**
  - Example: Predicting whether an email is spam or not spam.
  - Metrics: Accuracy, Precision, Recall, F1 Score, AUC-ROC.
- **Support Vector Machines (SVM)**
  - Example: Classifying handwritten digits.
  - Metrics: Accuracy, F1 Score, Confusion Matrix.
- **Decision Trees**
  - Example: Predicting loan approval based on applicant attributes.
  - Metrics: Accuracy, Precision, Recall, F1 Score, Gini Impurity.
- **Random Forest**
  - Example: Predicting customer churn based on user behavior.
  - Metrics: Accuracy, Feature Importance, AUC-ROC, Out-of-Bag Error.
- **k-Nearest Neighbors (k-NN)**
  - Example: Classifying a tumor as benign or malignant based on attributes.
  - Metrics: Accuracy, Precision, F1 Score.
- **Naive Bayes**
  - Example: Text classification for sentiment analysis.
  - Metrics: Accuracy, Precision, Recall, Log-Loss.
- **Gradient Boosting (XGBoost, CatBoost)**
  - Example: Predicting credit card fraud.
  - Metrics: AUC-ROC, Precision, Recall, F1 Score.
- **Multilayer Perceptron (MLP)**
  - Example: Predicting stock prices from historical data.
  - Metrics: MSE, RMSE, R².
- **Convolutional Neural Networks (CNN)**
  - Example: Image recognition (e.g., identifying objects in pictures).
  - Metrics: Accuracy, Precision, F1 Score, Top-5 Accuracy.
- **Recurrent Neural Networks (RNN)**
  - Example: Predicting the next word in a sentence (text generation).
  - Metrics: Perplexity, Cross-Entropy Loss.

## Unsupervised Learning Techniques
- **k-Means Clustering**
  - Example: Grouping customers by purchasing behavior.
  - Metrics: Silhouette Score, Inertia, Davies-Bouldin Index.
- **Hierarchical Clustering**
  - Example: Gene expression data clustering in bioinformatics.
  - Metrics: Cophenetic Correlation Coefficient, Silhouette Score.
- **DBSCAN**
  - Example: Identifying geographical clusters of events (e.g., earthquakes).
  - Metrics: Silhouette Score, Adjusted Rand Index.
- **Gaussian Mixture Models (GMM)**
  - Example: Identifying customer segments in marketing.
  - Metrics: Bayesian Information Criterion (BIC), Log-Likelihood.
- **Principal Component Analysis (PCA)**
  - Example: Reducing dimensionality for image compression.
  - Metrics: Explained Variance Ratio, Reconstruction Error.
- **t-SNE**
  - Example: Visualizing high-dimensional data (e.g., word embeddings).
  - Metrics: Visual assessment (no direct numerical evaluation).
- **Autoencoders**
  - Example: Detecting anomalies in network traffic.
  - Metrics: Reconstruction Error, Mean Squared Error.
- **Isolation Forest**
  - Example: Detecting outliers in financial transactions.
  - Metrics: Precision, Recall, AUC-ROC, F1 Score.
- **Apriori Algorithm**
  - Example: Market basket analysis to find product associations (e.g., customers buying bread often buy butter).
  - Metrics: Support, Confidence, Lift.
- **Generative Adversarial Networks (GANs)**
  - Example: Generating realistic-looking images.
  - Metrics: Inception Score (IS), Frechet Inception Distance (FID).
- **Variational Autoencoders (VAE)**
  - Example: Generating synthetic images or data points for augmentation.
  - Metrics: Reconstruction Error, Log-Likelihood.

## Semi-Supervised Learning
- **Self-Training**
  - Example: Classifying medical images with limited labeled data.
  - Metrics: Accuracy, Precision, F1 Score (on the labeled set).
- **Co-Training**
  - Example: Document classification using two different feature sets.
  - Metrics: Accuracy, AUC-ROC, F1 Score.

---

# Evaluation Metrics Overview
- **Accuracy**: Percentage of correct predictions (useful for balanced classes).
- **Precision**: Proportion of true positives out of predicted positives (important for reducing false positives).
- **Recall**: Proportion of true positives out of actual positives (important for reducing false negatives).
- **F1 Score**: Harmonic mean of precision and recall (useful for imbalanced classes).
- **AUC-ROC**: Measures the area under the Receiver Operating Characteristic curve (useful for binary classification).
- **Mean Squared Error (MSE)**: Measures average squared differences between actual and predicted values (for regression).
- **Silhouette Score**: Evaluates how similar an object is to its own cluster vs. others (for clustering).