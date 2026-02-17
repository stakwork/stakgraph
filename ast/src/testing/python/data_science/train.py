
import torch
import pandas as pd
from .model import FraudDetector
from .pipeline import DataPipeline, clean_data

def train_model(data_path: str, epochs: int = 10):
    """
    Train the fraud detection model.
    """
    # Load data
    df = pd.read_csv(data_path)
    df = clean_data(df)

    # Preprocess
    pipeline = DataPipeline()
    features = pipeline.extract_features(df)
    
    # Initialize model
    model = FraudDetector(input_dim=features.shape[1])
    optimizer = torch.optim.Adam(model.parameters())

    # Training loop
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        # Mock forward pass
        output = model(torch.tensor(features))
        loss = torch.mean(output) # Mock loss
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch}: Loss {loss.item()}")

if __name__ == "__main__":
    train_model("data/transactions.csv")
