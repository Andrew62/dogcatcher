## dogcatcher

A tensorflow implementation of AlexNet and VGG19 to recognize dog breeds. For a full list of dog breed, check out the [AKC](www.akc.org) website. Alexnet is an 8 layer convolution neural network (CNN) and VGG is a 19 layer network. This implementation differs slightly from the original in that it use a smaller set of classes (252 instead of 1000). 

Alexnet:

![AlexNet Image](http://www.eecs.berkeley.edu/~shhuang/img/alexnet_small.png)

## Files:

**cnn** contains the code for both networks along with code for generating batches. It is important to note that this was originally developed to run on a laptop; however, it will take **FOREVER** to covnverge unless you have a dedicated machine. Since adding dropout, training takes about twice as long compared to without dropout. The two networks are in subfolders named for the model. Models are kept separate because they were originally developed in separate environments. 

**data_formatting_scripts** contains code to prepare the input data. The code will crop, flip, rotate, and normalize input images. mp_image_prep.py starts N instances of MPTransformer and will increase your dataset five times. You can remove the image transformations for VGG as the authors of the VGG paper say the transformations are not required for good generalization

**tests** contains scripts to ensure modules are working properly. The data test script will run, provide load times, and show an example input image. 

## Metrics

Initial AlexNet results show a loss of 15% (85% correct) after 50000 iterations without dropout. Trianing took 12 days with a quad-core i5 CPU, 32 GB RAM and SSD running Ubuntu 14.04.

VGG metrics to follow
