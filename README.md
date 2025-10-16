🎵 Spotify Genre Classification using ML & ANN

This project builds a genre classification system using both Classical Machine Learning models (SVM, Perceptron, Random Forest) and a Neural Network (PyTorch).
The system learns to predict a song’s genre based on its audio features from the Spotify dataset.

1.Classical Machine Learning Models

Perceptron
SVM (RBF, Sigmoid, Polynomial)
Random Forest

Each model is evaluated using 5-Fold Stratified Cross-Validation for:
Scaled and Unscaled feature inputs
Mean and Standard Deviation of accuracy
Results are saved as saved_models/cv_results_table.csv.

2.Artificial Neural Network (PyTorch)

A two-layer feedforward neural network trained for 100 epochs:
Hidden layer: 128 neurons
Activation: ReLU
Output: LogSoftmax
Loss Function: Negative Log Likelihood (NLLLoss)
Optimizer: Adam (with weight decay)
Regularization: Dropout (0.3)
The ANN is trained using mini-batch gradient descent (implemented directly in PyTorch).
Performance metrics (loss and accuracy) are plotted for both train and test sets.

Dependencies

This project requires:
pandas
numpy
scikit-learn
matplotlib
torch
joblib
pickle


💾 Dataset Requirement

The script expects a dataset in CSV format with Spotify audio features.
📍 Kaggle Path(Recommended)
If running in Kaggle, ensure your dataset is uploaded as an input dataset with one of the following paths:
/kaggle/input/-spotify-tracks-dataset/dataset.csv[depends on kagle user]


🖥️ Local System Alternative

If running locally, download the same dataset and update the paths:
data_path = "path/to/your/dataset.csv"


🧪 Constraints and Notes

Ensure your dataset includes:
track_genre (target column)
Feature columns like danceability, energy, tempo, etc.
The script assumes no missing values after cleaning.
For consistent results, all random states are fixed (random_state=42).


✨ Author

Kushal D — 3rd year CSE
Mrunmayi Sandeep mohite-3rd year CSE
Project: Playlist Curator and Genre Classifier


