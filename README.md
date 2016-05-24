## dogcatcher

A tensorflow implementation of AlexNet to recognize dog breeds. For a full
list of dog breed, check out the [AKC](www.akc.org) website. Performance metrics to follow.

### Files:

*cnn* contains the code for Alexnet along with code for generating batchs. It is important to note that this was originally developed to run on a laptop; however, it will take *FOREVER* to covnverge unless you have a dedicated machine. Since adding dropout, training takes about twice as long compared to without dropout.

*data_formatting_scripts* contains code to prepare the input data. The code will crop, flip, rotate, and normalize input images. mp_image_prep.py starts N instances of MPTransformer and will increase your dataset five times. 

*tests* contains scripts to ensure modules are working properly. The data test script will run, provide load times, and show an example input image. 

## Metrics

Initial results shows a loss of 15% (85% correct) after 50000 iterations. 
