# News Source Classification (SVM)

## Model
Linear SVM (scikit-learn) with TF-IDF vectorization, used to classify news headlines as FoxNews (0) or NBC (1).

## Files
- model.py: model definition and prediction logic
- preprocess.py: data loading and preprocessing from URLs
- weights/:
    - best_svm_model.pkl: pretrained SVM model
    - tfidf_vectorizer.joblib: pretrained TF-IDF vectorizer
- requirements.txt: dependencies

model.py assumes best_svm_model.pkl and tfidf_vectorizer.joblib are located in the weights/ directory.

## Dataset Format
The input CSV must contain a url column. Labels are derived directly from the URL domain:
- foxnews.com -> 0 (FoxNews)
- nbcnews.com -> 1 (NBC)

The preprocessing pipeline extracts text from URLs and applies minimal preprocessing (lowercasing only).

## Running the Code

Setup:
pip install -r requirements.txt

Evaluation:
The model is pretrained; no training is performed at evaluation time.
To run inference and compute accuracy, execute:
python eval_project_b.py --model model.py --preprocess preprocess.py --csv <INPUT_CSV>

Replace <INPUT_CSV> with the path to the provided evaluation CSV file with the only column being 'url'.

## Evaluation Procedure
The evaluator performs the following steps:
1. Calls preprocess.prepare_data(csv_path) to load and preprocess data
2. Initializes the model via get_model() in model.py
3. Loads pretrained weights from the weights/ directory
4. Runs inference using model.predict(batch) to compute accuracy

Output: classification accuracy (fraction of correctly predicted examples).
