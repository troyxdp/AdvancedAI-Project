import argparse

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from classes.vae import VariationalAutoencoder
from classes.mnist_dataloader import MnistDataloader

if __name__ == '__main__':
    # Get args
    parser = argparse.ArgumentParser(description='Program to generate MNIST-like samples using a VAE')
    parser.add_argument('--vae', type=str, default='networks/best.pkl', help='Path to VAE pickle file')
    parser.add_argument('--images', type=str, default='archive/train-images.idx3-ubyte')
    parser.add_argument('--labels', type=str, default='archive/train-labels.idx1-ubyte')
    parser.add_argument('--samples', type=int, default=9, help='Number of samples to run through encoder to get mean and log variance averages')
    args = parser.parse_args()

    # Check valid number of samples given (must be square number)
    if not np.sqrt(args.samples) == float(int(np.sqrt(args.samples))):
        raise Exception("Error: non-square number provided for number of samples")

    # Load network
    vae: VariationalAutoencoder = VariationalAutoencoder.load_network(args.vae)

    # Extract decoder
    print("Getting encoder...")
    encoder = VariationalAutoencoder()
    for i in range(7):
        layer = vae.get_layer(i)
        encoder.append_layer(layer)

    # Get mean and stdev of distribution
    print("Loading data...")
    data, _ = MnistDataloader(
        args.images, 
        args.labels
    ).load_data()
    x_train = data[:1000]
    for i in range(len(x_train)):
        x_train[i] = np.array(x_train[i])
        x_train[i] = x_train[i] / 255.0

    print("Getting mean and log variance of distribution...")
    mean_sigma = np.zeros(16)
    log_var_sigma = np.zeros(16)
    for x in tqdm(x_train, desc="Log variance and mean search progress: ", ncols=150):
        _ = encoder.forward(x)
        mean_sigma += encoder.get_layer(-1).get_mean()
        log_var_sigma += encoder.get_layer(-1).get_log_var()
    mean_sigma *= (1 / len(x_train))
    log_var_sigma *= (1 / len(x_train))
    print("Mean:", mean_sigma)
    print("Log variance:", log_var_sigma)

    # Extract decoder
    print("Getting decoder...")
    decoder = VariationalAutoencoder()
    for i in range(7, 14):
        layer = vae.get_layer(i)
        decoder.append_layer(layer)
    
    # Generate samples
    print("Generating samples...")
    plt.figure(figsize=(5, 5))
    for i in range(args.samples):
        eps = np.random.normal(0, 1, 16)
        z = mean_sigma + np.exp(log_var_sigma * 0.5) * eps
        image = decoder.forward(z)
        image = np.ceil(image * 255)
        plt.subplot(3, 3, i + 1)
        plt.imshow(image, cmap=plt.cm.gray)
        plt.axis('off')

    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    plt.show()