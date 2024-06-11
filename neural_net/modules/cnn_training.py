import torch
import numpy as np

def net_training(epochs, net, device, loss_func, optimizer, scheduler, train_loader, val_loader):
    train_loss_arr = np.zeros(epochs)
    val_loss_arr = np.zeros(epochs) 

    for epoch in range(1, epochs+1):
        cum_num = 0
        cum_loss = 0

        # training loop
        net.train()
        for i, (features, labels, _) in enumerate(train_loader):
            # sending data to device
            features = features.to(device)
            labels = labels.to(device)

            # performing forward pass
            predictions = net(features)

            # calculating loss
            loss = loss_func(labels, predictions[:,0], predictions[:,1])

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # cumulative loss across batch for terminal display
            cum_loss += loss.item()
            cum_num += 1

        scheduler.step()

        train_loss = cum_loss/cum_num
        train_loss_arr[epoch-1] = train_loss

        cum_num = 0
        cum_loss = 0

        # validation loop
        net.eval()
        for i, (features, labels, _) in enumerate(val_loader):
            # sending data to device
            features = features.to(device)
            labels = labels.to(device)

            # performing forward pass
            predictions = net(features)

            # calculating loss
            loss = loss_func(labels, predictions[:,0], predictions[:,1])

            # cumulative loss across batch for terminal display
            cum_loss += loss.item()
            cum_num += 1

        val_loss = cum_loss/cum_num
        val_loss_arr[epoch-1] = val_loss
        
        if epoch%10 == 0 or epoch == 1:
            print (f"Epoch [{epoch:05d}/{epochs:05d}], Train Loss: {train_loss:.8f}, Val Loss: {val_loss:.8f}")

    return net, train_loss_arr, val_loss_arr
        
