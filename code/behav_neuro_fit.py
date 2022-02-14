# -*- coding: utf-8 -*-

"""
Authors: Zheng Wang
input: emp sc emp eeg stimulus and behavior data
outputs: sim eeg and behavior (fitted)
"""

import numpy as np # for numerical operations
import pandas as pd # for data manipulation
import torch
import torch.optim as optim


from torch.nn.parameter import Parameter
from sklearn.metrics.pairwise import cosine_similarity

class OutputJR():
    mode_all = ['train', 'test']
    stat_vars_all = ['m', 'v']
    def __init__(self, state_names, node_size, param, fit_weights=False):
        self.loss = np.array([])
        
        for name in state_names+['EEG']:
           for m in self.mode_all:
               setattr(self, name+'_'+m,np.array([]))
        
        vars = [a for a in dir(param) if not a.startswith('__') and not callable(getattr(param, a))]
        for var in vars:
            if np.any(getattr(param, var)[1]> 0):
                if var != 'std_in':
                    setattr(self, var, np.array([]))
                    for stat_var in self.stat_vars_all:
                        setattr(self, var+'_'+stat_var, np.array([]))
                else:
                    setattr(self, var, np.array([]))
        if  fit_weights == True:
            self.weights = np.array([])
        self.leadfield = np.array([])
    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
        
class ParamsJR():
    
    def __init__(self, **kwargs):
        for var in kwargs:
            setattr(self, var, kwargs[var])
def sys2nd(A, a,  u, x, v):
    return A*a*u -2*a*v-a**2*x

def sigmoid(x, vmax, v0, r):
    return vmax/(1+torch.exp(r*(v0-x)))


class RNNJANSEN(torch.nn.Module):
    """
    A module for forward model (JansenRit) to simulate a batch of EEG signals
    
    Attibutes
    ---------
    state_size : int
        the number of states in the JansenRit model
    input_size : int
        the number of states with noise as input
    tr : float
        tr of image
    step_size: float
        Integration step for forward model
    hidden_size: int
        the number of step_size in a tr 
    batch_size: int
        the number of EEG signals to simulate
    node_size: int
        the number of ROIs
    sc: float node_size x node_size array   
        structural connectivity
    fit_gains: bool
        flag for fitting gains 1: fit 0: not fit

    g, c1, c2, c3,c4: tensor with gradient on
        model parameters to be fit

    w_bb: tensor with node_size x node_size (grad on depends on fit_gains)
        connection gains

    std_in std_out: tensor with gradient on
        std for state noise and output noise

    hyper parameters for prior distribution of model parameters


    Methods
    -------
    forward(input, noise_out, hx)
        forward model (JansenRit) for generating a number of EEG signals with current model parameters

    """
    state_names = ['E', 'Ev', 'I', 'Iv', 'P', 'Pv']
    def __init__(self, input_size: int, node_size: int,
                 batch_size: int, step_size: float, eeg_size: int, behav_size: int, stimus_size: int, tr: float, sc: float,\
                 dist:float, lm: float, fit_gains: bool, \
                 param: ParamsJR) -> None:
        """
        Parameters
        ----------
        state_size : int
        the number of states in the JansenRit model
        input_size : int
            the number of states with noise as input
        tr : float
            tr of image
        step_size: float
            Integration step for forward model
        hidden_size: int
            the number of step_size in a tr 
        batch_size: int
            the number of EEG signals to simulate
        node_size: int
            the number of ROIs
        output_size: int
            the number of channels EEG
        sc: float node_size x node_size array   
            structural connectivity
        fit_gains: bool
            flag for fitting gains 1: fit 0: not fit
        param from ParamJR

        """
        super(RNNJANSEN, self).__init__()
        self.state_size = 6 # 6 states WWD model
        self.input_size = input_size # 1 or 2 or 3
        self.tr = tr # tr ms (integration step 0.1 ms)
        self.step_size = torch.tensor(step_size , dtype=torch.float32) # integration step 0.1 ms
        self.hidden_size = np.int(tr/step_size)
        self.batch_size = batch_size # size of the batch used at each step
        self.node_size = node_size # num of ROI   
        self.eeg_size = eeg_size # num of EEG channels 
        self.behav_size = behav_size # num of behavior data
        self.stimus_size = stimus_size # num of stimus   
        self.sc = sc # matrix node_size x node_size structure connectivity
        self.dist = torch.tensor(dist, dtype=torch.float32)
        self.fit_gains = fit_gains # flag for fitting gains
        self.param = param
        self.lm = Parameter(torch.tensor(lm, dtype=torch.float32)) # from source level to channel
        self.W_in = Parameter(torch.tensor(np.ones((self.node_size, self.stimus_size)), dtype=torch.float32)) # from stimulus to ROI
        self.W_out = Parameter(torch.tensor(np.random.randn(self.behav_size, self.node_size, self.state_size//2), dtype=torch.float32)) # from ROI to behav
        
  
        # model parameters (variables: need to calculate gradient)
        

        vars = [a for a in dir(param) if not a.startswith('__') and not callable(getattr(param, a))]
        for var in vars:
            if np.any(getattr(param, var)[1] > 0):
                if type(getattr(param,var)[0]) == 'numpy.ndarray':
                    if var == 'lm':
                        setattr(self, var, Parameter(torch.tensor(lm, dtype=torch.float32)))
                    elif var == 'w_bb':
                        setattr(self, var, Parameter(torch.tensor(-0.05*np.ones((self.node_size, self.node_size)), dtype=torch.float32)))
                    else:
                        setattr(self, var, Parameter(torch.tensor(getattr(param,var)[0]+1/getattr(param,var)[1]*np.random.randn(\
                          getattr(param,var)[0].shape[0], getattr(param,var)[0].shape[1]), dtype=torch.float32)))
                else:
                    setattr(self, var, Parameter(torch.tensor(getattr(param,var)[0]+1/getattr(param,var)[1]*np.random.randn(1, ), dtype=torch.float32)))
                if var != 'std_in':
                    dict_nv = {}
                    dict_nv['m'] = getattr(param, var)[0]
                    dict_nv['v'] = getattr(param, var)[1]

                    dict_np ={}
                    dict_np['m'] = var+'_m'
                    dict_np['v'] = var+'_v'
                    
                    for key in dict_nv:
                        setattr(self, dict_np[key], Parameter(torch.tensor(dict_nv[key], dtype=torch.float32)))
            else:
                setattr(self, var, torch.tensor(getattr(param,var)[0], dtype=torch.float32))

        if self.fit_gains == True:
            self.w_bb = Parameter(torch.tensor(np.zeros((node_size,node_size)) + 0.05, dtype=torch.float32)) # connenction gain to modify empirical sc
        else:
            self.w_bb = torch.tensor(np.ones((node_size,node_size)), dtype=torch.float32)

        
        
       

    
    
    
    
    def forward(self, input, noise_in, noise_out, hx, hE):
        """
        Forward step in simulating the EEG signal. 

        Parameters
        ----------
        input: tensor with node_size x hidden_size x batch_size x input_size
            noise for states
        noise_out: tensor with node_size x batch_size
            noise for EEG
        hx: tensor with node_size x state_size
            states of JansenRit model

        Outputs
        -------
        next_state: dictionary with keys:
        'current_state''EEG_batch''E_batch''I_batch''M_batch''Ev_batch''Iv_batch''Mv_batch'
            record new states and EEG

        """
        next_state = {}

        
            
        M = hx[:,0:1]
        E = hx[:,1:2]
        I = hx[:,2:3]
        
        Mv = hx[:,3:4]
        Ev = hx[:,4:5]
        Iv = hx[:,5:6]

        dt = self.step_size
        # Generate the ReLU module for model parameters gEE gEI and gIE
        
        m = torch.nn.ReLU()

        # define constant 1 tensor
        con_1 = torch.tensor(1.0, dtype=torch.float32)
        if  self.sc.shape[0] > 1:
            
            # Update the Laplacian based on the updated connection gains w_bb. 
            w = torch.exp(self.w_bb)*torch.tensor(self.sc, dtype=torch.float32)
            w_n = torch.log1p(0.5*(w + torch.transpose(w, 0, 1)))/torch.linalg.norm(torch.log1p(0.5*(w + torch.transpose(w, 0, 1))))
            self.sc_m = w_n
            dg = -torch.diag(torch.sum(w_n, axis =1)) 
        else:
            l_s = torch.tensor(np.zeros((1,1)), dtype=torch.float32)
        W_in_n = self.W_in/torch.norm(self.W_in)
        W_out_n = self.W_out/torch.norm(self.W_out, dim = 0)
        #lm_n = torch.exp(self.lm)/torch.reshape(torch.sum(torch.exp(self.lm),1), (self.output_size,1))
        self.delays = (self.dist/(1.5*con_1+ m(self.mu))).type(torch.int64)
        #print(torch.max(self.delays), self.delays.shape)

        # placeholder for the updated corrent state
        current_state = torch.zeros_like(hx)
        
        
        
        
        # placeholders for output BOLD, history of E I x f v and q
        eeg_batch = []
        E_batch = []
        I_batch = []
        M_batch = []
        Ev_batch = []
        Iv_batch = []
        Mv_batch = []
        B_batch = []
        #B_sum = torch.tensor(0, dtype=torch.float32)
        
        # Use the forward model to get EEGsignal at ith element in the batch. 
        for i_batch in range(self.batch_size):
            # Get the noise for EEG output.     
            noiseEEG = noise_out[:,i_batch:i_batch+1]
            
            
            
            for i_hidden in range(self.hidden_size):
                
                Ed = torch.tensor(np.zeros((self.node_size,self.node_size)), dtype=torch.float32)# delayed E
                
                
                """for ind in range(self.node_size):
                    #print(ind, hE[ind,:].shape, self.delays[ind,:].shape)

                    Ed[ind] = torch.index_select(hE[ind,:], 0, self.delays[ind,:])"""
                hE_new = hE.clone()
                Ed = hE_new.gather(1, self.delays)
                
                
                    
                LEd = torch.reshape(torch.sum(w_n*torch.transpose(Ed,0,1), 1), (self.node_size,1)) # weights on delayed E
                
                # Input noise for M. 
                noiseE = noise_in[:,i_hidden, i_batch,0:1]
                noiseI = noise_in[:,i_hidden, i_batch,1:2]
                noiseM = noise_in[:,i_hidden, i_batch,2:3]
                u = input[i_hidden:i_hidden+1, i_batch]
                
                
                #LEd+torch.matmul(dg,E): Laplacian on delayed E
                
                rM = sigmoid(E - I, self.vmax, self.v0, self.r) # firing rate for Main population
                rE =  (150*con_1+m(self.std_in))*noiseE + (0.01*con_1+ m(self.g))*(1*LEd+1*torch.matmul(dg,E))\
                      + (0.01*con_1+m(self.c2))*sigmoid((0.01*con_1+m(self.c1))*M, self.vmax, self.v0, self.r) # firing rate for Excitory population 
                rI = (0.01*con_1+m(self.c4))*sigmoid((0.01*con_1+m(self.c3))*M, self.vmax, self.v0, self.r) # firing rate for Inhibitory population
                

                # Update the states by step-size.
                ddM = M + dt* Mv
                ddE = E + dt*Ev
                ddI = I + dt*Iv
                ddMv = Mv + dt* sys2nd(0*con_1+m(self.A), 1*con_1+m(self.a), +500*torch.tanh(rM*0.002), M, Mv)\
                        +0*torch.sqrt(dt)*(1.0*con_1+m(self.std_in))*noiseM 

                ddEv = Ev + dt* sys2nd(0*con_1+m(self.A), 1*con_1+m(self.a), \
                                       torch.matmul(self.W_in, u.T)\
                                       +500*torch.tanh(rE*0.002), E, Ev) #(0.001*con_1+m_kw(self.kw))/torch.sum(0.001*con_1+m_kw(self.kw))*

                ddIv = Iv + dt* sys2nd(0*con_1+m(self.B), 1*con_1+m(self.b),  \
                                       +500*torch.tanh(rI*0.002), I, Iv) +0*torch.sqrt(dt)*(1.0*con_1+m(self.std_in))*noiseI
                
                
                # Calculate the saturation for model states (for stability and gradient calculation). 
                E = ddE#1000*torch.tanh(ddE/1000)#torch.tanh(0.00001+torch.nn.functional.relu(ddE))
                I = ddI#1000*torch.tanh(ddI/1000)#torch.tanh(0.00001+torch.nn.functional.relu(ddI))
                M = ddM#1000*torch.tanh(ddM/1000)
                Ev = ddEv#1000*torch.tanh(ddEv/1000)#(con_1 + torch.tanh(df - con_1))
                Iv = ddIv#1000*torch.tanh(ddIv/1000)#(con_1 + torch.tanh(dv - con_1))
                Mv = ddMv#1000*torch.tanh(ddMv/1000)#(con_1 + torch.tanh(dq - con_1))
                
                # update placeholders for E buffer
                hE = torch.cat([E,hE[:,:-1]], axis = 1) # update placeholders for E buffer
                
            # Put x f v q from each tr to the placeholders for checking them visually.
            M_batch.append(M)
            I_batch.append(I)
            E_batch.append(E)
            Mv_batch.append(Mv)
            Iv_batch.append(Iv)
            Ev_batch.append(Ev)
            B_sum = torch.matmul(W_out_n[:,:,0],E)+torch.matmul(W_out_n[:,:,1],I) ++torch.matmul(W_out_n[:,:,1],M)
                   

            B_batch.append(B_sum)
            
            
            # Put the EEG signal each tr to the placeholder being used in the cost calculation.
            lm_t =(self.lm -1/self.eeg_size*torch.matmul(torch.ones((1,self.eeg_size)), self.lm))
            temp = .01*self.cy0*torch.matmul(lm_t, E-I) -1*self.y0
            eeg_batch.append(temp)#torch.abs(E) - torch.abs(I) + 0.0*noiseEEG)
            current_state = torch.cat([M, E, I, Mv, Ev, Iv], axis = 1)
            
        # Update the current state. 
        
        next_state['current_state'] = current_state
        next_state['eeg_batch'] = torch.cat(eeg_batch, axis=1)
        next_state['E_batch'] = torch.cat(E_batch, axis=1)
        next_state['I_batch'] = torch.cat(I_batch, axis=1)
        next_state['M_batch'] = torch.cat(M_batch, axis=1)
        next_state['Ev_batch'] = torch.cat(Ev_batch, axis=1)
        next_state['Iv_batch'] = torch.cat(Iv_batch, axis=1)
        next_state['Mv_batch'] = torch.cat(Mv_batch, axis=1)
        next_state['B_batch'] = torch.cat(B_batch, axis=1)
        

        return next_state, hE



def cost_dist(logits_series_tf, labels_series_tf):
    """
    Calculate the Pearson Correlation between the simFC and empFC. 
    From there, the probability and negative log-likelihood.

    Parameters
    ----------
    logits_series_tf: tensor with node_size X datapoint
        simulated EEG
    labels_series_tf: tensor with node_size X datapoint
        empirical EEG
    """
    # get node_size(batch_size) and batch_size()
    node_size = logits_series_tf.shape[0]
    truncated_backprop_length = logits_series_tf.shape[1]
    
    
    losses = torch.sqrt(torch.mean((logits_series_tf - labels_series_tf)**2))#
    return losses


class Model_fitting():
    """
    Using ADAM and AutoGrad to fit JansenRit to empirical EEG

    Attributes
    ----------

    model: instance of class RNNJANSEN
        forward model JansenRit
    ts: array with num_tr x node_size
        empirical EEG time-series
    num_epoches: int
        the times for repeating trainning

    Methods:
    train()
        train model 
    test()
        using the optimal model parater to simulate the BOLD
    
    """

    def __init__(self, model, ts, behav, num_epoches):
        """
        Parameters
        ----------
        model: instance of class RNNJANSEN
            forward model JansenRit
        ts: array with num_tr x node_size
            empirical EEG time-series
        num_epoches: int
            the times for repeating trainning

        """
        self.model = model
        self.num_epoches = num_epoches
        self.y = behav
        
        self.ts = ts

    def train(self, u):
        """
        Parameters
        ----------
        None
        Outputs: OutputRJ
        
        """
        self.u = u
        
        self.output_sim = OutputJR(self.model.state_names, self.model.node_size, self.model.param, self.model.fit_gains)
        # define an optimizor(ADAM)
        optimizer = optim.Adam(self.model.parameters(), lr=0.05, eps=1e-7)

        # initial state
        X = torch.tensor(np.random.uniform(0, 5, (self.model.node_size, self.model.state_size)) , dtype=torch.float32)
        hE = torch.tensor(np.random.uniform(0, 5, (self.model.node_size,500)), dtype=torch.float32)
        # placeholders for model parameters
        

        fit_param = {}
        for key, value in self.model.state_dict().items():
            if key != 'W_out' and key != 'lm' and key != 'w_bb':
                fit_param[key] = [value.detach().numpy().ravel().copy()]
        
        
        # define masks for geting lower triangle matrix
        mask = np.tril_indices(self.model.node_size, -1)
        mask_e = np.tril_indices(self.model.eeg_size, -1)
        fit_sc = [self.model.sc[mask].copy()]
        fit_lm =  [self.model.lm.detach().numpy().ravel().copy()]
        loss_his =[]

        

        
        
        
        # define constant 1 tensor

        con_1 = torch.tensor(1.0, dtype=torch.float32)

        # define num_batches
        num_batches = np.int(self.ts.shape[1] / self.model.batch_size)
        
        for i_epoch in range(self.num_epoches):
            
            #X = torch.tensor(np.random.uniform(0, 5, (self.model.node_size, self.model.state_size)) , dtype=torch.float32)
            #hE = torch.tensor(np.random.uniform(0, 5, (self.model.node_size,83)), dtype=torch.float32)
            eeg = self.ts[:,:,i_epoch%self.ts.shape[2]]
            print(eeg.shape)
            if i_epoch%self.ts.shape[2] ==0:
                decision_train = []
                E_sim_train = []
                I_sim_train = []
            # Create placeholders for the simulated EEG E I M Ev Iv and Mv of entire time series. 
            eeg_sim_train = []
            #E_sim_train = []
            #I_sim_train = []
            M_sim_train = []
            Ev_sim_train = []
            Iv_sim_train = []
            Mv_sim_train = []
            
            
            
            # Perform the training in batches.
            print(self.u[0,0,i_epoch])
            for i_batch in range(num_batches):
                
                # Reset the gradient to zeros after update model parameters. 
                optimizer.zero_grad()

                # Initialize the placeholder for the next state. 
                X_next = torch.zeros_like(X)

                # Get the input and output noises for the module. 
                noise_in = torch.tensor(np.random.randn(self.model.node_size, self.model.hidden_size, \
                            self.model.batch_size, self.model.input_size), dtype=torch.float32)
                noise_out = torch.tensor(np.random.randn(self.model.node_size, self.model.batch_size), dtype=torch.float32)
                external = torch.tensor((self.u[:,i_batch*self.model.batch_size:(i_batch+1)*self.model.batch_size,i_epoch%self.ts.shape[2]]), dtype=torch.float32)
                yr = torch.tensor(self.y[i_epoch%self.ts.shape[2], :], dtype=torch.float32)           
                # Use the model.forward() function to update next state and get simulated EEG in this batch. 
                next_batch, hE_new = self.model(external, noise_in, noise_out, X, hE)

                E_batch=next_batch['E_batch']
                I_batch=next_batch['I_batch']
                M_batch=next_batch['M_batch']
                Ev_batch=next_batch['Ev_batch']
                Iv_batch=next_batch['Iv_batch']
                Mv_batch=next_batch['Mv_batch']
                y_es=next_batch['B_batch']

                y_e= 1.0/(1.0+torch.exp(-(torch.mean(y_es[:,30:230], 1)-self.model.Bm)/(1.+self.model.Bv)))
                
                #print(B_batch.shape, yr.shape)
                # Get the batch of emprical EEG signal. 
                ts_batch = torch.tensor(eeg[:, i_batch*self.model.batch_size:(i_batch+1)*self.model.batch_size], dtype=torch.float32)

                
                loss_prior = []
                # define the relu function
                m = torch.nn.ReLU()
                variables_p = [a for a in dir(self.model.param) if not a.startswith('__') and not callable(getattr(self.model.param, a))]

                for var in variables_p:
                    #print(var)
                    if np.any(getattr(self.model.param, var)[1]>0 )and var != 'std_in' and var != 'W_out'and var != 'lm' and var != 'w_bb':
                        #print(var)
                        dict_np ={}
                        dict_np['m'] = var+'_m'
                        dict_np['v'] = var+'_v'
                        loss_prior.append(torch.sum((0.001+m(self.model.get_parameter(dict_np['v'])))*\
                                                    (m(self.model.get_parameter(var)) -m(self.model.get_parameter(dict_np['m'])))**2)\
                                                    +torch.sum(-torch.log(0.001+m(self.model.get_parameter(dict_np['v'])))))
                
                loss = 20*cost_dist(next_batch['eeg_batch'], ts_batch) + sum(loss_prior) + 20*torch.sum((y_e-yr)**2)

                
                # Put the batch of the simulated EEG, E I M Ev Iv Mv in to placeholders for entire time-series. 
                eeg_sim_train.append(next_batch['eeg_batch'].detach().numpy())
                E_sim_train.append(next_batch['E_batch'].detach().numpy())
                I_sim_train.append(next_batch['I_batch'].detach().numpy())
                M_sim_train.append(next_batch['M_batch'].detach().numpy())
                Ev_sim_train.append(next_batch['Ev_batch'].detach().numpy())
                Iv_sim_train.append(next_batch['Iv_batch'].detach().numpy())
                Mv_sim_train.append(next_batch['Mv_batch'].detach().numpy())
                decision_train.append(y_e.detach().numpy())
                
                loss_his.append(loss.detach().numpy())
                #print('epoch: ', i_epoch, 'batch: ', i_batch, loss.detach().numpy())
                
                # Calculate gradient using backward (backpropagation) method of the loss function. 
                loss.backward(retain_graph=True)
                
                # Optimize the model based on the gradient method in updating the model parameters. 
                optimizer.step()
                
                # Put the updated model parameters into the history placeholders. 
                #sc_par.append(self.model.sc[mask].copy())
                if i_epoch%40 == 0:
                    for key, value in self.model.state_dict().items():
                        if key != 'W_out' and key != 'lm' and key != 'w_bb':
                            fit_param[key].append(value.detach().numpy().ravel().copy())
                    
                    
                    
                    fit_sc.append(self.model.sc_m.detach().numpy()[mask].copy())
                    fit_lm.append(self.model.lm.detach().numpy().ravel().copy())
                
                # last update current state using next state... (no direct use X = X_next, since gradient calculation only depends on one batch no history)
                X = torch.tensor(next_batch['current_state'].detach().numpy(), dtype=torch.float32)
                hE = torch.tensor(hE_new.detach().numpy(), dtype=torch.float32)
                #print(hE_new.detach().numpy()[20:25,0:20])
                #print(hE.shape)
            fc = np.corrcoef(self.ts[:,:,i_epoch%self.ts.shape[2]])
            ts_sim = np.concatenate(eeg_sim_train, axis=1)
            E_sim = np.concatenate(E_sim_train, axis=1)
            I_sim = np.concatenate(I_sim_train, axis=1)
            M_sim = np.concatenate(M_sim_train, axis=1)
            Ev_sim = np.concatenate(Ev_sim_train, axis=1)
            Iv_sim = np.concatenate(Iv_sim_train, axis=1)
            Mv_sim = np.concatenate(Mv_sim_train, axis=1)
            Behav_sim = np.array(decision_train)
            fc_sim = np.corrcoef(ts_sim[:, 10:])

            print('epoch: ', i_epoch, loss.detach().numpy())
            print(fc.shape, fc_sim.shape)
            
            print('epoch: ', i_epoch, np.corrcoef(fc_sim[mask_e], fc[mask_e])[0, 1], 'cos_sim: ',\
                  np.diag(cosine_similarity(ts_sim, eeg)).mean()\
                  , np.diag(cosine_similarity(ts_sim, eeg)).max())

            self.output_sim.EEG_train = ts_sim
            self.output_sim.E_train = E_sim
            self.output_sim.I_train= I_sim
            self.output_sim.P_train = M_sim
            self.output_sim.Ev_train = Ev_sim
            self.output_sim.Iv_train= Iv_sim
            self.output_sim.Pv_train = Mv_sim
            self.output_sim.Behav_train = Behav_sim
            self.output_sim.loss = np.array(loss_his)
            
            
            
            """if i_epoch > 65 and  np.corrcoef(fc_sim[mask_e], fc[mask_e])[0, 1] > 0.80:
                
                break"""
        #print('epoch: ', i_epoch, np.corrcoef(fc_sim[mask_e], fc[mask_e])[0, 1])
        self.output_sim.weights = np.array(fit_sc)
        self.output_sim.leadfield = np.array(fit_lm)
        for key, value in fit_param.items():
            setattr(self.output_sim, key, np.array(value))


    def test(self, u):
        """
        Parameters
        ----------
        num_batches: int
            length of simEEG = batch_size x num_batches

        values of model parameters from model.state_dict

        Outputs:
        output_test: OutputJR

        """

        self.u = u
        num_base = 1
        # initial state
        X = torch.tensor(np.random.uniform(0, 5, (self.model.node_size, self.model.state_size)) , dtype=torch.float32)
        hE = torch.tensor(np.random.uniform(0, 5, (self.model.node_size,500)), dtype=torch.float32)
        # placeholders for model parameters
        
        # define mask for geting lower triangle matrix
        mask = np.tril_indices(self.model.node_size, -1)
        mask_e = np.tril_indices(self.model.output_size, -1)
        
        # define num_batches
        num_batches = np.int(self.ts.shape[1] / self.model.batch_size) + num_base
        # Create placeholders for the simulated BOLD E I x f and q of entire time series. 
        eeg_sim_test = []
        E_sim_test = []
        I_sim_test = []
        M_sim_test = []
        Ev_sim_test = []
        Iv_sim_test = []
        Mv_sim_test = []
        Behav_sim_test = []

        
        
        u_hat = np.zeros((self.model.hidden_size, num_base*self.model.batch_size+self.ts.shape[1]))
        u_hat[:, num_base*self.model.batch_size:] = self.u
        
        # Perform the training in batches.
        
        for i_batch in range(num_batches):
            
            
            # Initialize the placeholder for the next state. 
            X_next = torch.zeros_like(X)

            # Get the input and output noises for the module. 
            noise_in = torch.tensor(np.random.randn(self.model.node_size, self.model.hidden_size, \
                        self.model.batch_size, self.model.input_size), dtype=torch.float32)
            noise_out = torch.tensor(np.random.randn(self.model.node_size, self.model.batch_size), dtype=torch.float32)
            external = torch.tensor((u_hat[:,i_batch*self.model.batch_size:(i_batch+1)*self.model.batch_size]), dtype=torch.float32)
                
            # Use the model.forward() function to update next state and get simulated EEG in this batch. 
            next_batch, hE_new = self.model(external, noise_in, noise_out, X, hE)

            
            if i_batch > num_base -1:
                eeg_sim_test.append(next_batch['eeg_batch'].detach().numpy())
                E_sim_test.append(next_batch['E_batch'].detach().numpy())
                I_sim_test.append(next_batch['I_batch'].detach().numpy())
                M_sim_test.append(next_batch['M_batch'].detach().numpy())
                Ev_sim_test.append(next_batch['Ev_batch'].detach().numpy())
                Iv_sim_test.append(next_batch['Iv_batch'].detach().numpy())
                Mv_sim_test.append(next_batch['Mv_batch'].detach().numpy())
                Behav_sim_test.append(next_batch['B_batch'].detach().numpy())
            
            
            
            # last update current state using next state... (no direct use X = X_next, since gradient calculation only depends on one batch no history)
            X = torch.tensor(next_batch['current_state'].detach().numpy(), dtype=torch.float32)
            hE = torch.tensor(hE_new.detach().numpy(), dtype=torch.float32)
            #print(hE_new.detach().numpy()[20:25,0:20])
            #print(hE.shape)
        fc = np.corrcoef(self.ts.mean(2)) 
        ts_sim = np.concatenate(eeg_sim_test, axis=1)
        E_sim = np.concatenate(E_sim_test, axis=1)
        I_sim = np.concatenate(I_sim_test, axis=1)
        M_sim = np.concatenate(M_sim_test, axis=1)
        Ev_sim = np.concatenate(Ev_sim_test, axis=1)
        Iv_sim = np.concatenate(Iv_sim_test, axis=1)
        Mv_sim = np.concatenate(Mv_sim_test, axis=1)
        Mv_sim = np.concatenate(Mv_sim_test, axis=1)
        Behav_sim = np.array(Behav_sim_test)
        
        fc_sim = np.corrcoef(ts_sim[:, 10:])
        print('r: ', np.corrcoef(fc_sim[mask_e], fc[mask_e])[0, 1], 'cos_sim: ', np.diag(cosine_similarity(ts_sim, self.ts.mean(2))).mean())
        
        self.output_sim.EEG_test = ts_sim
        self.output_sim.E_test = E_sim
        self.output_sim.I_test= I_sim
        self.output_sim.P_test = M_sim
        self.output_sim.Ev_test = Ev_sim
        self.output_sim.Iv_test = Iv_sim
        self.output_sim.Pv_test = Mv_sim
        self.output_sim.Behav_test = Behav_sim