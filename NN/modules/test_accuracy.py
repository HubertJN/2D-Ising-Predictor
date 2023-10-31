import torch
import numpy as np
from math import log10, floor

# finding exponent of number
def find_exp(number) -> int:
    base10 = log10(abs(number))
    return floor(base10)

def test_accuracy(net, testloader, selection, test_batch_size, output_label=None, device="cpu"):
    correct = 0
    total = 0
    with torch.no_grad():
        if np.any(output_label) == None: # if diff_array is not given, then return accuracy and do not store diff_array
            for i, (images, labels) in enumerate(testloader):
                images, labels = images.to(device), (labels).to(device)
                outputs = net(images)
                predicted = outputs
                total += labels.size(0)
                for j, (k, l) in enumerate(zip(predicted,labels)):
                    diff = abs(k-l[0]).item()
                    correct += (diff < 0.05)
                    #correct += (abs(predicted.item() - labels.item()) < 0.01)
        else: # if diff_array is given, then return accuracy and store diff_array
            for i, (images, labels) in enumerate(testloader):
                images, labels = images.to(device), (labels).to(device)
                outputs = net(images)
                predicted = outputs
                total += labels.size(0)
                for j, (k, l) in enumerate(zip(predicted,labels)):
                    diff = abs(k-l[selection]).item()
                    correct += (diff < 10**find_exp(l[selection].item()))
                    output_label[i*test_batch_size+j, 0] = l[0].item()
                    output_label[i*test_batch_size+j, 1] = l[1].item()
                    output_label[i*test_batch_size+j, 2] = l[2].item()
                    output_label[i*test_batch_size+j, 3] = l[3].item()
                    output_label[i*test_batch_size+j, 4] = l[4].item()
                    output_label[i*test_batch_size+j, 5] = l[5].item()
                    output_label[i*test_batch_size+j, 6] = l[6].item()
                    output_label[i*test_batch_size+j, 7] = l[7].item()
                    output_label[i*test_batch_size+j, 8] = l[8].item()
                    output_label[i*test_batch_size+j, 9] = l[9].item()
                    output_label[i*test_batch_size+j, 10] = k.item()

    return correct / total, output_label
