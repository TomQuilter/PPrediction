import torch
import math
import numpy as np
import matplotlib.pyplot as plt
from typing import Union

from models.IterativeModel import IterativeModel
from utils.metric_utils.calc_metric import calc_acc

class AbilityDifficultyVariationalInference(IterativeModel):
    def __init__(self, model_params):
        super().__init__(model_params)


    def train(self, train_ts, val_ts, test_ts, S, Q, rate, iters, init, step_size):
        acc_arr_size = math.ceil(iters/step_size)
        train_nll_arr, val_nll_arr, test_nll_arr = np.zeros(iters), np.zeros(acc_arr_size), np.zeros(acc_arr_size)
        train_acc_arr, val_acc_arr, test_acc_arr = np.zeros(acc_arr_size), np.zeros(acc_arr_size), np.zeros(acc_arr_size)

        # Randomly ### initialise random student ####, question parameters
        # bs = torch.zeros(S, requires_grad=True)
        # bq = torch.zeros(Q, requires_grad=True)


        #bs = torch.randn(S, requires_grad=True, generator=self.rng)  ## Creates a Bs for each student
        #print("bs1", bs)
 
        ## initialise Mu and sigma for each student
        Ms = torch.zeros(S, requires_grad=True)
        Ss = torch.ones(S, requires_grad=True)    
        
        # Draw 1 sample for each (mean, std_dev) pair using the reparameterization trick
        epsilon = torch.randn(S)  # samples from a standard normal (mean=0, std=1)
        bs = Ms + Ss * epsilon  # reparameterization trick
        print("bs2", bs)

        bq = torch.randn(Q, requires_grad=True, generator=self.rng)

        last_epoch = iters
        prev_nll = 0
        for epoch in range(iters):
            params = {'bs': bs, 'bq': bq} # {'bs': bs,'Ms': Ms, 'Ss': Ss, 'bq': bq} # {'bs': bs, 'bq': bq}

            train_nll = self.calc_nll(train_ts, params)
            train_nll.backward()
            
            if epoch % step_size == 0:
                val_nll = self.calc_nll(val_ts, params)
                test_nll = self.calc_nll(test_ts, params)

                # terminate at last iteration if test nll of this iter greater than test nll of last iter
                if epoch != 0 and test_nll > prev_nll:
                    last_epoch = epoch
                    break
                
                val_nll_arr[epoch//step_size] = val_nll
                test_nll_arr[epoch//step_size] = test_nll
        
                train_acc = calc_acc(train_ts[0], self.predict(train_ts, params)[1])
                val_acc = calc_acc(val_ts[0], self.predict(val_ts, params)[1])
                test_acc = calc_acc(test_ts[0], self.predict(test_ts, params)[1])
                train_acc_arr[epoch//step_size], val_acc_arr[epoch//step_size], test_acc_arr[epoch//step_size] = train_acc, val_acc, test_acc

                self.print_iter_res(epoch, train_nll, val_nll, test_nll, train_acc, val_acc, test_acc)

            # Gradient descent
            with torch.no_grad():
                bs -= rate * bs.grad  ## All Bs's updated, needs to be Mu and sigma now
                #Ms -= rate * Ms.grad
                #Ss -= rate * Ss.grad
                bq -= rate * bq.grad

            # Zero gradients after updating
            bs.grad.zero_()
            #Ms.grad.zero_()
            #Ss.grad.zero_()
            bq.grad.zero_()

            train_nll_arr[epoch] = train_nll
            prev_nll = test_nll

        history = {'avg train nll': np.trim_zeros(train_nll_arr, 'b')/train_ts.shape[1], 
                    'avg val nll': np.trim_zeros(val_nll_arr, 'b')/val_ts.shape[1],
                    'avg test nll': np.trim_zeros(test_nll_arr, 'b')/test_ts.shape[1], 
                    'train acc': np.trim_zeros(train_acc_arr, 'b'),
                    'val acc': np.trim_zeros(val_acc_arr, 'b'),
                    'test acc': np.trim_zeros(test_acc_arr, 'b')}    ### Bs = mu + here to change params to Ms and SS?
        params = {'bs': bs, 'bq': bq} # {'bs': bs,'Ms': Ms, 'Ss': Ss, 'bq': bq} # {'bs': bs, 'bq': bq}
        return params, history, last_epoch
 
    def calc_probit(self, data_ts, params):
        # Draw 1 sample for each (mean, std_dev) pair using the reparameterization trick
        #epsilon = torch.randn(S)  # samples from a standard normal (mean=0, std=1)
        #bs = Ms + Ss * epsilon  # reparameterization trick
        #print("bs2", bs)
       # print("params['Ms']",params['Ms'])
        #print("params['Ss']",params['Ss'])  
        bs_data = torch.index_select(params['bs'], 0, data_ts[1])  ### Grab params['bs'] , grad MuS and SS and them generate no!?  
        print("bs_data") 
        print(bs_data)
        ## EVERY time u want Bs (for all students) u take a draw and get a 1 by S vector
        ## So replace Bs with Mu and Sigma and generate the Bs's
        ## select with data_ts[1]) an issue here?? No just takes a certain bs value - bs is just a 1 by S list of numbers
        bq_data = torch.index_select(params['bq'], 0, data_ts[2])

        probit_correct = torch.sigmoid(bs_data + bq_data)

        return probit_correct

    def calc_nll(self, data_ts, params):
        probit_correct = self.calc_probit(data_ts, params)
        nll = -torch.sum(data_ts[0]*torch.log(probit_correct) + (1-data_ts[0])*torch.log(1-probit_correct)) ## NEED KL TERM TOO
        return nll


    def predict(self, data_ts, params):
        probit_correct = self.calc_probit(data_ts, params)
        predictions = (probit_correct>=0.5).float()
        return probit_correct, predictions
