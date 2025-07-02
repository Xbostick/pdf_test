# 20 Newsgroups Classification Project

This project classifies text documents from the [20 Newsgroups dataset](https://scikit-learn.org/stable/datasets/real_world.html#the-20-newsgroups-text-dataset) into one of 20 categories (e.g., `sci.space`, `rec.sport.hockey`) using machine learning models. It includes a Flask web server to upload PDF files and predict their category, along with tests to ensure the code works correctly.

## Features
- **Text Preprocessing**: Cleans text (removes URLs, emails, punctuation) and converts it to numerical features using TF-IDF and PCA.
- **Models**: Supports three models:
  - Logistic Regression (simple and fast)
  - Linear Neural Network (basic deep learning)
  - Convolutional Neural Network (CNN, advanced deep learning)
- **Web Interface**: Upload a PDF via a Flask server to get its predicted category.
- **Tests**: Simple unit tests to check key functions.

## Requirements
To run this project, you need Python 3.8+ and the following packages:
```
numpy
torch
scikit-learn
joblib
nltk
pdfminer.six
flask
pytest
matplotlib
tqdm
```

Install them using:
```bash
pip install numpy torch scikit-learn joblib nltk pdfminer.six flask pytest matplotlib tqdm
```

## Setup
1. Clone the repository:
   ```bash
   git clone <your-repository-url>
   cd <repository-name>
   ```
2. Install the required packages (see above).

## Usage
### Training Models
1. Open the `example.ipynb` notebook in Jupyter:
   ```bash
   jupyter notebook example.ipynb
   ```
2. Run the cells to:
   - Load the 20 Newsgroups dataset.
   - Preprocess the data (creates `models/Preprocessor_pretrained.pkl`).
   - Train a model (Logistic Regression, Linear, or CNN).
   - Save the model to the `models` folder.
   - Test predictions on a sample.

   Example: To train a Logistic Regression model, run the cell with:
   ```python
   model = train(x_train, x_test, y_train, y_test, model_type="LogReg")
   ```

### Running the Web Server
1. Start the Flask server:
   ```bash
   python server.py
   ```
2. Open your browser and go to `http://127.0.0.1:5000`.
3. Upload a PDF file.
4. The server will predict its category using the pretrained model.

### Running Tests
1. Ensure `pytest` is installed.
2. Run the tests:
   ```bash
   pytest test_newsground.py -v
   ```
3. The tests check key functions like text cleaning, model predictions, and file uploads. You should see output like `9 passed` if everything works.

## File Structure
- `labels_newsground.py`: List of 20 category labels (e.g., `alt.atheism`, `rec.autos`).
- `dl_functions.py`: Code to train and predict with Logistic Regression, Linear, and CNN models.
- `text_helper.py`: Functions to clean text, preprocess data, and convert PDFs to features.
- `server.py`: Flask server to upload PDFs and predict categories.
- `example.ipynb`: Jupyter notebook showing how to load data, train models, and test predictions.
- `test_newsground.py`: Unit tests for key functions and models.
- `models/`: Folder to store trained models (created during training).
- `files/`: Folder for uploaded PDFs (created during server use).
- `pages/`: Folder with HTML templates for the Flask server (e.g., `download_page.html`).

## Example
To train a CNN model and test it:
1. Run `example.ipynb` cells to load data and train the CNN:
   ```python
   model = train(x_train, x_test, y_train, y_test, model_type="CNN", epoch=10)
   ```
2. Predict a sample:
   ```python
   sample = preprocessor.preprocess(newsgroups_test.data[0])
   prediction = predict(sample, model_type="CNN")
   print(f"Predicted class: {labels_newsground[prediction[0]]}")
   ```

## Notes
- Ready to use models storing [here](https://pages.github.com/)
- If tests fail due to missing models, train the models first using `example.ipynb`.
- The Flask server uses the Linear model by default for predictions. Edit `server.py` to use a different model.
