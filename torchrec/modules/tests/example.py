import torch
import torch.nn as nn
import nvtx

# Define a simple model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(1000, 100)

    def forward(self, x):
        return self.fc(x)

# Annotate initialization with NVTX
@nvtx.annotate("Initialize Model", color="blue")
def initialize_model():
    model = SimpleModel().cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    return model, optimizer

# Annotate training loop with NVTX
@nvtx.annotate("Training Loop", color="green")
def train(model, optimizer):
    criterion = nn.MSELoss()
    for i in range(5):  # Small number of iterations for demonstration
        with nvtx.annotate(f"Iteration {i}", color="yellow"):
            inputs = torch.randn(64, 1000).cuda()
            targets = torch.randn(64, 100).cuda()

            # Forward pass
            with nvtx.annotate("Forward Pass", color="red"):
                outputs = model(inputs)

            # Compute loss
            with nvtx.annotate("Compute Loss", color="purple"):
                loss = criterion(outputs, targets)

            # Backward pass
            with nvtx.annotate("Backward Pass", color="orange"):
                loss.backward()

            # Optimizer step
            with nvtx.annotate("Optimizer Step", color="cyan"):
                optimizer.step()
                optimizer.zero_grad()

if __name__ == "__main__":
    model, optimizer = initialize_model()
    train(model, optimizer)
