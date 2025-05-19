import numpy as np 
def reshape(data):
    dim = int(np.sqrt(len(data)))
    data = np.reshape(data, (dim, dim))
    return data 

import matplotlib.pyplot as plt
def plot(data, test, predicted, figsize=(3, 3)):
    data = [reshape(d) for d in data]
    test = [reshape(d) for d in test]
    predicted = [reshape(d) for d in predicted]
    
    fig, axrr = plt.subplots(len(data), 3, figsize=figsize)
    if len(data) == 1:
        axrr = np.expand_dims(axrr, axis=0)

    for i in range(len(data)):
        if i == 0:
            axrr[i, 0].set_title("train data")
            axrr[i, 1].set_title("Input data")
            axrr[i, 2].set_title("Output data")
        
        axrr[i, 0].imshow(data[i])
        axrr[i, 0].axis('off')
        axrr[i, 1].imshow(test[i])
        axrr[i, 1].axis("off")
        axrr[i, 2].imshow(predicted[i])
        axrr[i, 2].axis("off")
    
    plt.tight_layout()
    plt.savefig("result_mnist.jpg")
    plt.show()
    
def preprocessing(img):
    if len(img.shape) == 1:
        dim = int(np.sqrt(len(img)))
        img = np.reshape(img, (dim, dim))

    w, h = img.shape
    from skimage.filters import threshold_mean
    thresh = threshold_mean(img)
    binary = img > thresh
    shift = 2 * (binary * 1) - 1
    flatten = np.reshape(shift, (w * h))
    return flatten

 
    