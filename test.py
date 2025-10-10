import argparse

import numpy as np
from tqdm.auto import tqdm

from classes.vae import VariationalAutoencoder
from classes.mnist_dataloader import MnistDataloader

if __name__ == '__main__':
    # Get arguments
    parser = argparse.ArgumentParser(description='Program to generate MNIST-like samples using a VAE')
    parser.add_argument('--vae', type=str, default='networks/best.pkl', help='Path to VAE pickle file')
    parser.add_argument('--images', type=str, default='archive/t10k-images.idx3-ubyte')
    parser.add_argument('--labels', type=str, default='archive/t10k-labels.idx1-ubyte')
    args = parser.parse_args()

    # Load network
    vae: VariationalAutoencoder = VariationalAutoencoder.load_network(args.vae)

    # Load data
    data, _ = MnistDataloader(
        args.images, 
        args.labels
    ).load_data()
    for i in range(len(data)):
        data[i] = np.array(data[i])
        data[i] = data[i] / 255.0

    # Run test
    test_error = 0
    for sample in tqdm(data, desc="Testing cycle progress: ", ncols=150):
        # get song and feed it forward through network
        sample = np.array(sample)
        output = vae.forward(sample)

        # get error of output for statistics
        # Reconstruction loss + KL divergence
        error = (1 / (sample.shape[0] * sample.shape[1])) * (np.dot(np.subtract(sample, output).flatten(), np.subtract(sample, output).flatten()) + 0.5 * (vae.get_mean()**2 + np.exp(vae.get_log_var()) - 1 - vae.get_log_var()))
        test_error += error

    print('\nOVERALL ERROR:', test_error)