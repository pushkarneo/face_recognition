# face_recognition - keras

This is the implemention to detect similarity of faces in two images.
It is achieved using convolution neural network in keras. 

Please download the dataset required to train this network here (http://vis-www.cs.umass.edu/lfw/lfw.tgz)

## Usage Instructions

### To train the network, follow the below steps
- Download and extract the LFW dataset from above link
- run below command, it will create .npy files after serialising the training and testing data and also train the network
```
python lfw_model.py
```

### To infer from the network

```
python lfw_model.py -infer_similarity True -image1 <path/to/image1> -image2 <path/to/image2>  -weights <path/to/modelcheckpoint>
```
