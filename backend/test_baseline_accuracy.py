import torch
import os
import sys

# Import necessary functions from existing scripts
from train import get_model, get_cifar10_loaders, evaluate, get_input_size

def main():
    # Allow model path to be passed as an argument, defaulting to resnet18_baseline.pth
    model_path = "../models/resnet18_baseline.pth"
    if len(sys.argv) > 1:
        model_path = sys.argv[1]

    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        sys.exit(1)

    # Extract model name from filename (e.g., 'resnet18_baseline.pth' -> 'resnet18')
    filename = os.path.basename(model_path)
    model_name = filename.split('_baseline')[0].split('.pth')[0]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading '{model_name}' from {model_path} onto {device}...")

    # Load the model architecture
    try:
        model = get_model(model_name, num_classes=10, pretrained=False)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Load the trained weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    # Set up data loaders
    input_size = get_input_size(model_name)
    print(f"Setting up CIFAR-10 testing data (input size: {input_size}x{input_size})...")
    
    # We only need the validation/testing loader
    _, test_loader = get_cifar10_loaders(
        input_size=input_size, 
        batch_size=128,
        num_workers=2, 
        data_dir='./data'
    )

    # Evaluate accuracy
    print("Evaluating Test Accuracy (this might take a minute)...")
    accuracy = evaluate(model, test_loader, device)
    
    print(f"\n{'-'*40}")
    print(f"Model: {filename}")
    print(f"Test Accuracy: {accuracy:.2f}%")
    print(f"{'-'*40}")

if __name__ == '__main__':
    main()
