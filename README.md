# pytorch-feedbacknet

This is an implementation of the work done [here](http://feedbacknet.stanford.edu/).
It's a fascinating concept and I'd like to try to incorporate external memory or something
similar.

I'm trying to be better about type safety, hence all the type checking, etc.

So far it works, just not very fast (I don't have a very big GPU for training). It definitely
learns CIFAR-10, still working on CIFAR-100. I haven't implemented the curriculum learning
yet, but that requires reworking the CIFAR-100 dataset to make use of coarse and fine
labels (meaning I can't use the torchvision dataset).