===========
Usage Guide
===========

Spyker implements spiking neural network with integrate-and-fire neurons and rank coding (read the paper for more details). The interface of the library is devided into these categories:

* Feature Enhancement

  * Difference of Gaussian (DoG) Filter
  * Laplacian of Gaussian (LoG) Filter
  * Gabor filter
  * ZCA Whitening

* Neural Coding

  * Rank Coding
  * Rate Coding

* Neural Network

  * 2D Convolution Layer
  * Fully Connected Layer
  * 2D Padding Operation
  * 2D Pooling Operation
  * Integrate-and-fire Operation

* Learning

  * Lateral Inhibition
  * Winner-take-all Mechanism
  * STDP Learning Rule
  * R-STDP Learning Rule

* Other

  * Canny Edge Detection

The library implements CPU and CUDA implementations of these operations. There is also an sparse interface that supports main operations required for a network with rank coding.
