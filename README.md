# Advanced AI Project
Welcome to my Advanced AI Project from my Honours year! The project was a Variational Autoencoder (VAE) that generated MNIST-like data. 

To train a model, run the following command:
```
python train.py
```
I didn't use an argparser for this project, so be sure to set the hyperparameters for training on lines 352 onwards in the `train.py` file. 
Be sure to set the correct path to the training and validation data of MNIST on lines 353 and 354 and the folder to output the training results (the stats and the network) on line 356. 
Also, to change the network architecture you will have to do so inside the code of this file in lines 228 to 350. Invalid architectures will cause an error to be thrown. 

To test a network, run the following command:
```
python test.py
```
It has default values already set, but you can change the network path using the `--vae` argument. 
It also needs to get the mean and standard deviation for the distribution of the MNIST data created by the encoder by running some MNIST data through the encoder. Set the paths for the MNIST data with the `--images` and `--labels` arguments. 
