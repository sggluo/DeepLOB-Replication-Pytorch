import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from torchmetrics import Accuracy, Precision, Recall
from torchmetrics.classification import MulticlassF1Score



def train_and_evaluate(model, train_loader, val_loader, criterion,
    optimizer, n_epochs, device='cpu', scheduler=None,
                 early_stop=False, patience=5, path=None):

    device = torch.device(device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        print('Cuda is not available.')
        device = torch.device('cpu')
    print(f'The device being used is {device}.')

    model = model.to(device)
    train_loss_history = []
    val_loss_history = []
    best_val = np.inf
    counter = 0
    best_epoch = 0

    for i in range(1,n_epochs+1):
        model.train()
        total_samples = 0
        total_train_loss = 0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            y_hat = model(x)
            loss = criterion(y_hat,y)
            loss.backward()
            optimizer.step()

            total_samples += len(y)
            total_train_loss += loss.item()*len(y)

        train_loss_history.append(total_train_loss/total_samples)

        model.eval()
        with torch.no_grad():
            total_val_loss = 0
            total_samples = 0
            for x,y in val_loader:
                x = x.to(device)
                y = y.to(device)
                y_hat = model(x)
                loss = criterion(y_hat,y)
                total_samples += len(y)
                total_val_loss += loss.item()*len(y)

            val_loss =  total_val_loss / total_samples
            val_loss_history.append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), path)
            best_epoch = i
            counter = 0
        else:
            counter += 1

        if i%5==0:
            print(f'At epoch {i}, train loss: {train_loss_history[-1]:.4f} | val loss: {val_loss:.4f}')

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLRONPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        if early_stop:
            if counter > patience:
                    print(f'Val loss stops decreasing > {counter} epochs.')
                    print(f'The best val loss is {best_val:.4f} at epoch {best_epoch}.')
                    print('Early stopping is thus triggered.')
                    break

    print(f'The best val loss is {best_val:.4f} at epoch {best_epoch}.')
    return  train_loss_history, val_loss_history

def test_model(model, test_loader, criterion, device, num_classes=3):
    model.eval()
    accuracy = Accuracy(task='multiclass', num_classes=num_classes, average='micro' ).to(device)
    precision = Precision(task='multiclass', num_classes=num_classes, average='macro' ).to(device)
    recall = Recall(task='multiclass', num_classes=num_classes, average='macro' ).to(device)
    f1_score = MulticlassF1Score(num_classes=num_classes, average='macro').to(device)

    total_samples = 0
    total_loss = 0
    all_y_hat = []
    all_y = []
    with torch.no_grad():
        for x,y in test_loader:
            x = x.to(device)
            y = y.to(device)
            y_hat = model(x)
            loss = criterion(y_hat, y)
            total_loss += len(y)*loss.item()
            total_samples += len(y)
            accuracy.update(y_hat, y)
            precision.update(y_hat, y)
            recall.update(y_hat, y)
            f1_score.update(y_hat, y)
            all_y_hat.append(y_hat.argmax(dim=1).to('cpu'))
            all_y.append(y.to('cpu'))

    loss_av = total_loss / total_samples
    acc = accuracy.compute().item()
    prec = precision.compute().item()
    rec = recall.compute().item()
    f1 = f1_score.compute().item()

    all_y = torch.cat(all_y).squeeze().numpy()
    all_y_hat = torch.cat(all_y_hat).squeeze().numpy()
    cm = confusion_matrix(all_y, all_y_hat, labels=[0,1,2])
    print(f'Accuracy: {acc}')
    print(f'Precision: {prec}')
    print(f'Recall: {rec}')
    print(f'F1 Score: {f1}')
    print(cm)

