import torch
import numpy as np
from math import log10, floor

# finding exponent of number
def find_exp(number) -> int:
    base10 = log10(abs(number))
    return floor(base10)

def test_accuracy(net, testloader, test_batch_size, output_label, device="cpu"):
    correct = 0
    total = 0
    with torch.no_grad():
        for i, (images, extra, labels) in enumerate(testloader):
            images, extra, labels = images.to(device), extra.to(device), labels.to(device)
            outputs = net(images, extra)
            predicted = outputs
            total += labels.size(0)
            for j, (k, l) in enumerate(zip(predicted,labels)):
                diff = abs(k-l[0]).item()
                output_label[i*test_batch_size+j, 0] = l[0].item()
                output_label[i*test_batch_size+j, 1] = l[1].item()
                output_label[i*test_batch_size+j, 2] = l[2].item()
                output_label[i*test_batch_size+j, 3] = l[3].item()
                output_label[i*test_batch_size+j, 4] = k.item()

    return output_label
