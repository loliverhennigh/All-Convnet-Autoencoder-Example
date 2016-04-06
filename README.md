# All-Convnet-Autoencoder-Example
Just a simple use example of the `conv2d_transpose` function in TensorFlow. Its run on MNIST. I was having a little trouble understanding the transpose conv stuff so I thought I would upload share my simple example

# Why All Convnet Autoecoder
All Convnet Autoencoder can be pretty powerful. They allow you to use modern network architetures on the image autoencoder problems. I havnet seen too many papers on them but I havnt been looking very hard. The best source I have seen explaining convolutional transpose is [here](http://arxiv.org/pdf/1511.06434v2.pdf).

# How well it do
I does ok. Having it reduce to a whopping 245 dimentions (mnist is 784) works well. I dont have a gpu right now so I cant test it out too well. It was geting stuck in local optima like crazy so I put dropout on the first layer at 20%. I have trained it to do this!

before
![alt text](https://github.com/loliverhennigh/All-Convnet-Autoencoder-Example/blob/master/typical_images/40_steps_in.png)
after
![alt text](https://github.com/loliverhennigh/All-Convnet-Autoencoder-Example/blob/master/typical_images/lots_o_steps.png)


