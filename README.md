# Table of Contents
- [Table of Contents](#table-of-contents)
- [Introduction](#introduction)
- [Installation](#installation)
- [Documentation](#documentation)
- [Tutorials](#tutorials)
- [Examples](#examples)
- [Contribution](#contribution)
- [License](#license)

# Introduction
Spyker is a high-performance library written from scratch that simulates spiking neural networks. It has both C++ and Python interfaces and can be easily integrated with popular tools like Numpy and PyTorch.

# Installation
Prebuilt packages will be available soon. For now, you can follow the instructions on how to build the library form source [here](https://spyker.readthedocs.io/en/latest/files/install.html).

# Documentation
You can see the documentation for the C++ and Python interfaces [here](https://spyker.readthedocs.io/en/latest/index.html).

# Tutorials
You can take a look at the tutorials listed below to learn how to use the library.

+ [Tutorial 1: Spyker and PyTorch](./tutorials/spyker_and_pytorch.ipynb)
+ [Tutorial 2: Spyker and Numpy](./tutorials/spyker_and_numpy.ipynb)
+ [Tutorial 3: Sparse Spyker](./tutorials/sparse_spyker.ipynb)
+ [Tutorial 4: Other Functionalities](./tutorials/other_functionalities.ipynb)
+ [Tutorial 5: Rate Coding](./tutorials/rate_coding.ipynb)

# Examples
You can checkout example implementations of some networks in the [examples directory](./examples/). The example use the MNIST dataset, which is expected to be inside the `MNIST` directory beside the files, and the name of the files is expected to be `train-images-idx3-ubyte`, `train-labels-idx1-ubyte`, `t10k-images-idx3-ubyte`, `t10k-labels-idx1-ubyte`.

# Contribution
You can report bugs and request featues on the [issues page](../../issues).

# License
This library has a BSD 3-Clause permissive license. You can read it [here](LICENSE).
