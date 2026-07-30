import os

from ABPR import AutoBayesianPolynomialRegression

# Project path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# Data paths
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
TRAIN_FILE = os.path.join(DATA_DIR, "train.csv")
TEST_FILE = os.path.join(DATA_DIR, "test.csv")
MODEL_FILE = os.path.join(MODELS_DIR, "abpr_model.pkl")
TARGET_COLUMN = ["Y1", "Y2"]

# Model configs
CONFIGS = {
    "max_degree": 10,     
    "max_iter": 100,                 
    "evidence_tol": 1e-3,
    "latent_lr": 0.0,
}

def main():
    # Initialize the model
    print("Initializing Model")
    model = AutoBayesianPolynomialRegression(CONFIGS)

    # Train the model directly from the CSV
    print("Training Model")
    model.fit(TRAIN_FILE, TARGET_COLUMN)

    # Print out what the model learned
    print("Model Summary")
    print(model.summary())

    # Make predictions on the test file
    print("Testing Predictions")
    y_pred, y_std = model.predict(TEST_FILE, TARGET_COLUMN, return_std=True)

    # Save the model
    print("Saving Model")
    model.save(MODEL_FILE)

    print("Process completed successfully!")  

if __name__ == "__main__":
    main()