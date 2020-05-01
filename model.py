# #####
# model.py
# -----
# Author: Glenn Dawson (2020)
# -----
# Using transfer learing to classify chest X-ray images as being:
#  * COVID-19
#  * Healthy
#  * Non-COVID pneumonia
# Four base models are used:
#  * VGG-16
#  * Mobilenet_V2
#  * ResNet-50
#  * ResNeXt-101
# Training dataset:
#  * 259 COVID-19
#  * 540 Healthy
#  * 540 Non-COVID pneumonia
# Testing dataset:
#  * 28 COVID-19
#  * 60 Healthy
#  * 60 Non-COVID pneumonia
# Results:
#  * COVID-19 false negative rate: 0/28
#  * COVID-19 false positive rate: 3/120
# #####

from __future__ import print_function, division

from timeit import default_timer as timer
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
BATCH_SIZE = 16
EPOCHS = 10
FROZEN = True
SCHEDULER = True
MILESTONES = [5, 8]
INITIAL_LR = 0.001
MODEL = 'mobilenet'
FILENAME = 'covid-' + MODEL + '-model.pt'
CRITERION = nn.CrossEntropyLoss()
SHOW_PLOTS = True
EARLY = False
CHECK_CONFUSION = True


def main():
    train_ldr, val_ldr = load_data()
    if not CHECK_CONFUSION:
        model = make_model(n_classes=3)
        model, history = train(model, train_ldr, val_ldr, FILENAME)
        get_confusion_matrix(val_ldr, model=model)
        del model
        del train_ldr
        del val_ldr
        torch.cuda.empty_cache()
        if SHOW_PLOTS:
            plot(history)
    else:
        get_confusion_matrix(val_ldr, pretrained=True)
        

def train(model, train_ldr, val_ldr, save_filename):
    if torch.cuda.is_available():
        print(f'Training on {torch.cuda.get_device_name()}.\n')
    else:
        print(f'Training on CPU.\n')
    # Number of epochs already trained (if using loaded in model weights)
    try:
        print(f'Model has been trained for: {model.epochs} epochs.\n')
    except:
        model.epochs = 0
        print(f'Starting training from scratch.\n')

    print(f'Training on {len(train_ldr.dataset)} samples.')
    print(f'Validation on {len(val_ldr.dataset)} samples.')
    print(f'Number of classes: {model.n_classes}\n')

    model = model.to(device)
    max_epochs_stop = 3
    epochs_no_improve = 0
    val_loss_min = np.Inf
    val_best_acc = 0
    val_best_epoch = 0
    history = []
    optimizer = optim.Adam(model.parameters(), lr=INITIAL_LR)
    overall_start = timer()
    if SCHEDULER:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=MILESTONES,
            gamma=0.1
        )
    for epoch in range(1, EPOCHS + 1):
        train_loss = 0.0
        val_loss = 0.0

        train_acc = 0.0
        val_acc = 0.0

        model.train()
        start = timer()

        # Training loop
        for i, (data, target) in enumerate(train_ldr):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)

            # Backprop
            loss = CRITERION(output, target)
            loss.backward()
            optimizer.step()

            # Tracking
            train_loss += loss.item() * data.size(0)
            _, pred = torch.max(output, dim=1)
            correct = pred.eq(target.data.view_as(pred))
            accuracy = torch.mean(correct.type(torch.FloatTensor))
            train_acc += accuracy.item() * data.size(0)

            print(f'Epoch: {epoch}\t{100 * (i + 1) / len(train_ldr):.2f}% '
                  f'complete. {timer() - start:.2f} seconds elapsed in '
                  f'epoch.',
                  end='\r')

        # End-of-epoch validation
        else:
            model.epochs += 1
            with torch.no_grad():
                model.eval()

                # Validation loop
                for data, target in val_ldr:
                    data, target = data.to(device), target.to(device)
                    output = model(data)

                    loss = CRITERION(output, target)
                    val_loss += loss.item() * data.size(0)

                    _, pred = torch.max(output, dim=1)
                    correct = pred.eq(target.data.view_as(pred))
                    accuracy = torch.mean(correct.type(torch.FloatTensor))
                    val_acc += accuracy.item() * data.size(0)

                # Average loss
                train_loss = train_loss / len(train_ldr.dataset)
                val_loss = val_loss / len(val_ldr.dataset)

                # Average accuracy
                train_acc = train_acc / len(train_ldr.dataset)
                val_acc = val_acc / len(val_ldr.dataset)

                history.append([train_loss, val_loss, train_acc, val_acc])
                print(f'\n\t\tTraining Loss: {train_loss:.4f} \t\t'
                      f'Validation Loss: {val_loss:.4f}')
                print(f'\t\tTraining Accuracy: {100 * train_acc:.2f}%\t'
                      f'Validation Accuracy: {100 * val_acc:.2f}%')

                if val_loss < val_loss_min:
                    torch.save(model, FILENAME)

                    # Tracking
                    epochs_no_improve = 0
                    val_loss_min = val_loss
                    val_best_acc = val_acc
                    val_best_epoch = epoch
                elif EARLY:
                    epochs_no_improve += 1
                    if epochs_no_improve >= max_epochs_stop:
                        total_time = timer() - overall_start
                        print(f'\nEarly Stopping! Total epochs: {epoch}. Best '
                              f'epoch: {val_best_epoch} with loss:'
                              f' {val_loss_min:.2f} and acc: '
                              f'{100 * val_best_acc:.2f}%')
                        print(f'{total_time:.2f} total seconds elapsed. '
                              f'Average of {total_time / (epoch + 1):.2f} '
                              f'seconds per epoch.')

                        # Load the best state dict
                        model = torch.load(FILENAME)

                        # Attach the optimizer
                        model.optimizer = optimizer

                        # Format history
                        history = pd.DataFrame(history,
                                               columns=['train_loss',
                                                        'val_loss',
                                                        'train_acc',
                                                        'val_acc'])

                        return model, history
        if SCHEDULER:
            scheduler.step()
            
    # Load the best state dict
    model = torch.load(FILENAME)

    # Attach the optimizer
    model.optimizer = optimizer

    # Record overall time and print out stats
    total_time = timer() - overall_start
    print(f'\nBest epoch: {val_best_epoch} with loss: {val_loss_min:.2f} '
          f'and acc: {100 * val_best_acc:.2f}%')
    print(f'{total_time:.2f} total seconds elapsed. Average of '
          f'{total_time / (EPOCHS):.2f} seconds per epoch.'
    )

    # Format history
    history = pd.DataFrame(history,
                           columns=['train_loss', 'val_loss',
                                    'train_acc', 'val_acc'])

    return model, history


def make_model(n_classes=2):
    if MODEL == 'vgg16':
        model = models.vgg16(pretrained=True, progress=True)
    elif MODEL == 'mobilenet':
        model = models.mobilenet_v2(pretrained=True, progress=True)
    elif MODEL == 'resnet50':
        model = models.resnet50(pretrained=True, progress=True)
    elif MODEL == 'resnext101':
        model = models.resnext101_32x8d(pretrained=True, progress=True)
    else:
        raise ValueError('Invalid pretrained model.')

    model.n_classes = n_classes
    if FROZEN:
        for param in model.parameters():
            param.requires_grad = False

    final = nn.Sequential(
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(1000, 1000),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(1000, 750),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(750, 750),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(750, 500),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(500, model.n_classes)
    )

    if MODEL == 'vgg16':
        model.classifier[6] = nn.Sequential(
            nn.Linear(4096, model.n_classes)
        )
    elif MODEL == 'mobilenet':
        model.classifier = nn.Sequential(
            nn.Linear(1280, 1000),
            final
        )
    elif MODEL in ['resnet50', 'resnext101']:
        model.fc = nn.Sequential(
            nn.Linear(model.fc.in_features, 1000),
            final
        )

    print(f'Transfer learning from {MODEL}.')
    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')

    return model


def load_data():
    if MODEL == 'vgg16':
        resize = transforms.Resize((900, 800))
    elif MODEL == 'mobilenet':
        resize = transforms.Resize((800, 600))
    elif MODEL == 'resnet50':
        resize = transforms.Resize((800, 600))
    elif MODEL == 'resnext101':
        resize = transforms.Resize((600, 400))
    else:
        raise ValueError('Invalid pretrained model.')

    tfs = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        resize,
        transforms.ToTensor()
    ])

    train_ldr = torch.utils.data.DataLoader(
        datasets.ImageFolder(
            root='./Data/Train',
            transform=tfs
        ),
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    test_ldr = torch.utils.data.DataLoader(
        datasets.ImageFolder(
            root='./Data/Test',
            transform=tfs
        ),
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    return train_ldr, test_ldr


def get_confusion_matrix(val_ldr, model=None, pretrained=False):
    if pretrained:
        model = torch.load(FILENAME)
    elif model is None:
        raise ValueError('No model specified.')

    model = model.to(device)
    confusion = torch.zeros(model.n_classes, model.n_classes)
    with torch.no_grad():
        for data, targets in val_ldr:
            data, targets = data.to(device), targets.to(device)
            output = model(data)
            _, preds = torch.max(output, 1)
            for target, pred in zip(targets.view(-1), preds.view(-1)):
                confusion[pred.long(), target.long()] += 1
    print(MODEL)
    print(confusion)
    
    del model
    del val_ldr
    torch.cuda.empty_cache()


def plot(history):
    # Validation plot
    plt.figure()
    for c in ['train_loss', 'val_loss']:
        plt.plot(history[c], label=c)
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Average Cross Entropy Loss')
    plt.title('Training and Validation Losses')

    # Accuracy plot
    plt.figure()
    for c in ['train_acc', 'val_acc']:
        plt.plot(history[c] * 100, label=c)
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Average Accuracy')
    plt.title('Training and Validation Accuracy')

    plt.show()


if __name__ == '__main__':
    main()
