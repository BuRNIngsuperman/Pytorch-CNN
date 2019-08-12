import torch
import torch.nn as nn
from torch.autograd import Variable


# Train the model
def Train_mode(model, data_train, criterion, optimizer):
    model.train()
    for batch, (inputs, targets) in enumerate(data_train):
        inputs = Variable(inputs.cuda())
        targets = Variable(targets.cuda())
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        # update weights
        optimizer.step()


def get_train_loss(model, data_train, criterion):
    model.eval()
    train_loss = 0
    train_acc = 0
    total_size = 0
    for batch, (inputs, targets) in enumerate(data_train):
        with torch.no_grad():
            inputs = Variable(inputs.cuda())
            targets = Variable(targets.cuda())
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            train_loss = train_loss + loss.item()
            _, pred = outputs.max(1)
            total_size += targets.size(0)
            train_acc += pred.eq(targets).sum().item()

    train_acc = train_acc / total_size
    train_loss = train_loss / (batch + 1)
    return train_acc, train_loss


def get_test_loss(model, data_test, criterion):
    model.eval()
    test_loss = 0
    test_acc = 0
    total_size = 0
    with torch.no_grad():
        for batch,(inputs,targets) in enumerate(data_test):
            inputs = Variable(inputs.cuda())
            targets = Variable(targets.cuda())
            outputs = model(inputs)
            loss = criterion(outputs,targets)
            test_loss = test_loss+loss
            total_size = total_size+targets.size(0)
            _, test_pred = torch.max(outputs.data,1)
            test_acc += test_pred.eq(targets).sum().item()

        test_acc = test_acc/total_size
        test_loss = test_loss/(batch+1)

    return test_acc, test_loss
