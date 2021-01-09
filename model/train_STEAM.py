from data_loader import get_iterators
from model import BERT
from train import train
import torch.optim as optim


if __name__ == "__main__":
    print("Tokenizing data and creating batch iterators...")
    t_loader, v_loader = get_iterators()
    print("Done.")

    model = BERT().to('cuda:0')

    optimizer = optim.Adam(model.parameters(), lr=2e-5)

    print("Training has begun...")
    train(model=model, optimizer=optimizer, t_loader=t_loader, v_loader=v_loader, num_epochs=1)
    print("Done.")
