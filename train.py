import torch
from torch import nn
from timeit import default_timer as timer
from torch.utils.data import DataLoader
from src.datasets import Fashion
from src.train_using_pytorch import CNN
from helper_functions import accuracy_fn, print_train_time

def train (train_data_loader, test_data_loader):
    torch.manual_seed(42)
    time_train_start_model = timer()
    epochs = 3
    model = CNN(input_shape=784, hidden_units=10, output_shape=len(Fashion.class_name()))
    model.to('cpu')
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params=model.parameters(), lr=0.1)
    for epoch in range(epochs):
        print (f"Epoch: {epoch}")
        train_loss = 0
        for batch, (X, y) in enumerate(train_data_loader):
            y_pred = model(X)
            loss = loss_fn(y_pred, y)
            train_loss += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if batch % 400 == 0:
                print (f"Looked at {batch * len(X)}/{len(train_data_loader.dataset)} batch")
        train_loss /= len(train_data_loader)
        ##test
        test_loss, test_acc = 0, 0
        with torch.inference_mode():
            for X, y in test_data_loader:
                test_pred = model(X)
                test_loss += loss_fn(test_pred, y)
                test_acc += accuracy_fn(y_true=y, y_pred=test_pred.argmax(dim=1))
            test_loss /= len(test_data_loader)
            test_acc /= len(test_data_loader)
        print (f"\nTrain loss :{train_loss:.5f} | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%\n")
    time_train_end_model = timer()
    print_train_time(start= time_train_start_model,end= time_train_end_model, device=str(next(model.parameters()).device))

if __name__ == '__main__':
    BATCH_SIZE = 32
    train_data_loader = DataLoader(Fashion.load_data_train(), batch_size=BATCH_SIZE, shuffle=True)
    test_data_loader = DataLoader(Fashion.load_data_test(), batch_size=BATCH_SIZE, shuffle=False)
    train(train_data_loader, test_data_loader)



