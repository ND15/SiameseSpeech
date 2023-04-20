##  Siamese Capsule Network for End-to-End Speaker Recognition in the Wild

This repository is an implementation of the paper  Siamese Capsule Network 
for End-to-End Speaker Recognition in the Wild ([Paper link](https://arxiv.org/pdf/2009.13480.pdf)).
This repository contains the implementation for the front-end part of the model
and the back-end part needs to be implemented. I have changed some of the parameters
from the paper which includes window length, hop size, etc.<br>

## Comments

 One of the drawbacks of the siamese network is that for a dataset with **N** samples, 
 the dataset preprocessor will make the dataset size **N x N** and hence 
 requires more computational power and also more training time. So with a bigger window
 length, the dimensions of the spectrograms would also increase and will take a huge amount
 of space on disk.<br>
 
 In my implementation I have used a customised version of Vox Celeb Dataset. This dataset contains only 
 the recordings of the Indian celebrities, further for the ease of implementation for each 
 speaker I took only 25-30 recordings.

## Updates
- [Speech-Recognition](https://github.com/ND15/Speech-Recognition) a repo for DNN based speech recognition and verification.

## Links
- [Dataset](https://www.kaggle.com/datasets/gaurav41/voxceleb1-audio-wav-files-for-india-celebrity)
