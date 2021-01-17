from data_loader import get_iterators
from model import BERT
from train import train
from torch import optim, load


if __name__ == "__main__":
    print("Tokenizing data and creating batch iterators...")
    t_loader, v_loader = get_iterators()
    print("Done.")

    model = BERT().to('cuda:0')
    optimizer = optim.Adam(model.parameters(), lr=2e-5)

    try:
        print("Loading from checkpoint...")
        checkpoint = load('./checkpoints/modelchkpt.pt',  map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        best_loss = checkpoint['valid_loss']
        print("Done.")

        print("Training has begun...")
        train(model=model, optimizer=optimizer, t_loader=t_loader, 
              v_loader=v_loader, num_epochs=3-epoch, best_loss=best_loss)
        print("Done.")

    except FileNotFoundError:
        print("Training has begun...")
        train(model=model, optimizer=optimizer, t_loader=t_loader, 
              v_loader=v_loader, num_epochs=3)
        print("Done.")

