"""
Simple unit tests for complex functions in the 20 Newsgroups project, including LinearModel and CNNModel.
Designed for beginners, focusing on key functionality with minimal setup.
"""

import pytest
import numpy as np
import torch
from labels_newsground import labels_newsground
from train import dl_data_preprocess, train, predict, Linear_model, CNN_model
from text_helper import clean_text, preprocess_corpus, tokenize_text_inference
from server import allowed_file
from pathlib import Path

# Tests for train.py
def test_dl_data_preprocess():
    """Test if dl_data_preprocess converts data to tensors and encodes labels."""
    x_train = np.array([[1, 2], [3, 4]])
    x_test = np.array([[5, 6]])
    y_train = np.array([0, 1])
    y_test = np.array([0])

    x_train_out, x_test_out, y_train_out, y_test_out = dl_data_preprocess(x_train, x_test, y_train, y_test)

    assert isinstance(x_train_out, torch.Tensor), "x_train should be a tensor"
    assert isinstance(x_test_out, torch.Tensor), "x_test should be a tensor"
    assert x_train_out.shape == (2, 2), "x_train shape should be (2, 2)"
    assert x_test_out.shape == (1, 2), "x_test shape should be (1, 2)"
    assert y_train_out.shape == (2, 2), "y_train should be one-hot encoded with 2 classes"
    assert y_test_out.shape == (1,), "y_test should match input shape"

def test_train_logreg():
    """Test if train function works with LogisticRegression."""
    x_train = np.random.rand(5, 3)
    x_test = np.random.rand(2, 3)
    y_train = np.array([0, 1, 0, 1, 0])
    y_test = np.array([0, 1])

    model = train(x_train, x_test, y_train, y_test, model_type="LogReg", save=None)

    assert model is not None, "Model should be created"

def test_predict():
    """Test if predict function works with a simple input (assumes model exists)."""
    try:
        tokens = np.random.rand(1, 500)
        prediction = predict(tokens, model_type="LogReg", save="./models/")
        assert isinstance(prediction, np.ndarray), "Prediction should be a numpy array"
        assert prediction.shape == (1,), "Prediction should be for one sample"
    except FileNotFoundError:
        pytest.skip("Model file not found, skipping predict test")

def test_linear_model():
    """Test if LinearModel initializes and processes input correctly."""
    num_labels = 3
    model = Linear_model(num_labels)
    input_data = torch.rand(2, 500)  # Batch of 2 samples, 500 features

    output = model(input_data)

    assert isinstance(model, torch.nn.Module), "LinearModel should be a PyTorch module"
    assert output.shape == (2, num_labels), f"Output shape should be (2, {num_labels})"
    assert torch.allclose(output.sum(dim=1), torch.tensor(1.0)), "Softmax output should sum to 1"

def test_cnn_model():
    """Test if CNNModel initializes and processes input correctly."""
    num_labels = 3
    model = CNN_model(num_labels)
    input_data = torch.rand(2, 500)  # Batch of 2 samples, 500 features

    output = model(input_data.unsqueeze(dim = 1))

    assert isinstance(model, torch.nn.Module), "CNNModel should be a PyTorch module"
    assert output.shape == (2, num_labels), f"Output shape should be (2, {num_labels})"
    assert torch.allclose(output.sum(dim=1), torch.tensor(1.0)), "Softmax output should sum to 1"

# Tests for text_helper.py
def test_clean_text():
    """Test if clean_text removes URLs, emails, and punctuation."""
    text = "Hello https://example.com and user@example.com! 123"
    cleaned = clean_text(text)
    expected = "hello and "
    assert cleaned == expected, f"Expected '{expected}', got '{cleaned}'"
    assert "https" not in cleaned, "URLs should be removed"
    assert "@" not in cleaned, "Emails should be removed"
    assert "123" not in cleaned, "Numbers should be removed"

def test_preprocess_corpus():
    """Test if preprocess_corpus cleans and processes text."""
    corpus = ["Hello world!", "Test 123."]
    processed = preprocess_corpus(corpus)
    assert len(processed) == 2, "Should process two documents"
    assert isinstance(processed[0], str), "Output should be strings"
    assert "123" not in processed[1], "Numbers should be removed"
    assert "!" not in processed[0], "Punctuation should be removed"

def test_tokenize_text_inference():
    """Test if tokenize_text_inference processes a PDF (emulated preprocessor)."""
    class EmulPreprocessor:
        def preprocess(self, corpus):
            return np.array([[0.1, 0.2]])

    preprocessor = EmulPreprocessor()
    path = Path("./files/test.pdf")  # Dummy path, won't be used
    if not path.exists():
        pytest.skip("test.pdf file not found, skipping tokenize test")

        result = tokenize_text_inference(path, preprocessor)
        assert isinstance(result, np.ndarray), "Output should be a numpy array"
        assert result.shape == (1, 2), "Output shape should match preprocessor"
        

if __name__ == "__main__":
    pytest.main([__file__, "-v"])