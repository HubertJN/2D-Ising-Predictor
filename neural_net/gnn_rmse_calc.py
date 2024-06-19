import matplotlib.pyplot as plt
import numpy as np
import warnings
import sys
warnings.filterwarnings("ignore")
np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(linewidth=np.nan)

num_runs = 144
rmse = np.zeros([num_runs,3])
for run in range(1,num_runs+1):
    hyper_dict = {}
    data = np.load("./plotting_data/gnn/gnn_prediction_actual_%d.npy" % run)
    with open('./plotting_data/gnn/hyperparameters_%d.txt' % run, 'r') as f:
        for line in f:
           (key, val) = line.split(" = ")
           hyper_dict[key] = val.rstrip()
    f.close()
    
    alpha = data[:,1]
    beta = data[:,2]
    expectation = alpha/(alpha+beta)
    rmse[run-1, 0] = int(hyper_dict["k_edge"])
    rmse[run-1, 1] = int(hyper_dict["hidden_n"])
    rmse[run-1, 2] = np.sqrt(np.mean((data[:,0]-expectation)**2))
    # TEMPORARY FOR ADDING RMSE TO HYPERPARAMETER FILES
    hyper_dict['rmse'] = rmse[run-1, 2]
    with open("./plotting_data/gnn/hyperparameters_%d.txt" % run, 'w') as f:  
        for key, value in hyperparameters.items():  
            f.write('%s = %s\n' % (key, value))
    ##################################################

np.save("figures/gnn/rmse.npy", rmse)

