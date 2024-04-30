import torch
import numpy as np
import time

def gnn_training(epochs, net, device, loss_func, optimizer, scheduler, train_loader, val_loader):
    train_loss_arr = np.zeros(epochs)
    val_loss_arr = np.zeros(epochs)
    time_taken = 0

    initial_time = time.time()
    begin = initial_time

    for epoch in range(1, epochs+1):
        cum_num = 0
        cum_loss = 0

        # training loop
        net.train()
        for i, batch in enumerate(train_loader):
            # splitting batch
            features = batch.x
            edge_index = batch.edge_index
            labels = batch.y

            # performing forward pass
            predictions = net(features, edge_index, len(batch))

            # calculating loss
            loss = loss_func(labels, predictions[:,0], predictions[:,1])

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # cumulative loss across batch for terminal display
            cum_loss += loss.item()
            cum_num += 1

        train_loss = cum_loss/cum_num
        train_loss_arr[epoch-1] = train_loss

        cum_num = 0
        cum_loss = 0

        # validation loop
        net.eval()
        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                # splitting batch
                features = batch.x
                edge_index = batch.edge_index
                labels = batch.y

                # performing forward pass
                predictions = net(features, edge_index, len(batch))

                # calculating loss
                loss = loss_func(labels, predictions[:,0], predictions[:,1])

                # cumulative loss across batch for terminal display
                cum_loss += loss.item()
                cum_num += 1

        val_loss = cum_loss/cum_num
        val_loss_arr[epoch-1] = val_loss
        
        scheduler.step()

        if epoch%10 == 0 or epoch == 1:
            end = time.time()
            print (f"Epoch [{epoch:05d}/{epochs:05d}], Train Loss: {train_loss:+.8f}, Val Loss: {val_loss:+.8f}, Time Taken: {end-begin:.1f} seconds, Learning Rate: {scheduler.get_last_lr()[0]:.2E}")
            begin = time.time()

    
    final_time = time.time()
    time_taken = final_time - initial_time

    return net, train_loss_arr, val_loss_arr, time_taken
        
