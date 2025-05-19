import numpy as np
from matplotlib import pyplot as plt 
import matplotlib.cm as cm 
from tqdm import tqdm 

class HopfieldNetwork:
    def train_weights(self, train_data):
        num_data = len(train_data)
        self.num_neuron = train_data[0].shape[0] # extacting num of features from first sample
        
        # initiate weights
        W = np.zeros((self.num_neuron, self.num_neuron))
        rho = np.sum([np.sum(t) for t in train_data]) / (num_data * self.num_neuron) # mean activity per neuron
        
        # Hebb rule
        for i in tqdm(range(num_data)):
            t = train_data[i] - rho 
            W += np.outer(t, t) 

        # Making the diagonal element
        diagW = np.diag(np.diag(W))
        W = W - diagW
        W /= num_data
        
        self.W = W 
        
    def predict(self, data, num_iter=20, threshold=0, asyn=False):
        print('starting to predict....')
        self.num_iter = num_iter
        self.threshold = threshold 
        self.asyn = asyn 
        
        copied_data = np.copy(data)
        # Define the list
        predicted = []
        for i in tqdm(range(len(data))):
            predicted.append(self._run(copied_data[i]))
        return predicted 
    
    def _run(self, init_s):
        if self.asyn == False:
            # initial state energy
            s = init_s
            e = self.energy(s)
            # iteration
            for i in range(self.num_iter):
                # Update s
                s = np.sign(self.W @ s - self.threshold)
                # compute new state
                e_new = self.energy(s)
                # if e = new energy then coverged 
                if e == e_new:
                    return s
                else: 
                    e = e_new       
            return s 
        else:
            """
            Async Update
            """
            s = init_s
            e = self.energy(s)
            
            # iteratuin
            for i in range(self.num_iter):
                for j in range(100):
                    idx = np.random.randint(0, self.num_neuron) # select a random neuron
                    # update S
                    s[idx] = np.sign(self.W[idx].T @ s - self.threshold)
                e_new = self.energy(s)
                
                # coverged 
                if e == e_new:
                    return s 
                # update 
                else:
                    e = e_new 
            return s 
        
    def energy(self, s):
        return -0.5 * s @ self.W @ s + np.sum(s * self.threshold)
                    
    def plot_weights(self):
        plt.figure(figsize=(6, 5))
        w_mat = plt.imshow(self.W, cmap=cm.coolwarm)
        plt.colorbar(w_mat)
        plt.title("Network weights")
        plt.tight_layout()
        plt.savefig("weights.jpg")
        plt.show()