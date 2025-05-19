import numpy as np 
from tensorflow.keras.datasets import mnist
import network 
from utils import preprocessing, reshape, plot

def main():
    # load the data
    (x_train, y_train), (_, _) = mnist.load_data()
    data = []
    for i in range(10):
        xi = x_train[y_train==i]
        data.append(xi[0])
        
        print("starting to process data")
        data = [preprocessing(d) for d in data]
        model = network.HopfieldNetwork()
        model.train_weights(data)
        
        test = []
        for i in range(10):
            xi = x_train[y_train==i]
            test.append(xi[1])
        test = [preprocessing(d) for d in test]
        predicted = model.predict(test, threshold=50, asyn=True)
        print("show prediction results...")
        plot(data, test, predicted, figsize=(5, 5))
        print("show network weights matrix")
        model.plot_weights()

if __name__ == '__main__':
    main()
