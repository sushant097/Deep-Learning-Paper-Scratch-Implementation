# Image Captioning
Simple Implementation ofImage Captioning in PyTorch. 

### Dataset
The Flickr8k dataset is used. Download it from [here](https://www.kaggle.com/datasets/e1cd22253a9b23b073794872bf565648ddbe4f17e7fa9e74766ad3707141adeb)

### Training 
```
train.py: For training the network

model.py: creating the encoderCNN, decoderRNN and hooking them togethor

get_loader.py: Loading the data, creating vocabulary

utils.py: Load model, save model, printing few test cases downloaded online

```

## Image Captioning paper
Show and Tell: A Neural Image Caption Generator by Oriol Vinyals, Alexander Toshev, Samy Bengio, Dumitru Erhan .


### Abstract
Automatically describing the content of an image is a fundamental problem in artificial intelligence that connects computer vision and natural language processing. In this paper, we present a generative model based on a deep recurrent architecture that combines recent advances in computer vision and machine translation and that can be used to generate natural sentences describing an image. The model is trained to maximize the likelihood of the target description sentence given the training image. Experiments on several datasets show the accuracy of the model and the fluency of the language it learns solely from image descriptions. Our model is often quite accurate, which we verify both qualitatively and quantitatively. For instance, while the current state-of-the-art BLEU-1 score (the higher the better) on the Pascal dataset is 25, our approach yields 59, to be compared to human performance around 69. We also show BLEU-1 score improvements on Flickr30k, from 56 to 66, and on SBU, from 19 to 28. Lastly, on the newly released COCO dataset, we achieve a BLEU-4 of 27.7, which is the current state-of-the-art.

```bash
@misc{https://doi.org/10.48550/arxiv.1411.4555,
  doi = {10.48550/ARXIV.1411.4555},
  
  url = {https://arxiv.org/abs/1411.4555},
  
  author = {Vinyals, Oriol and Toshev, Alexander and Bengio, Samy and Erhan, Dumitru},
  
  keywords = {Computer Vision and Pattern Recognition (cs.CV), FOS: Computer and information sciences, FOS: Computer and information sciences},
  
  title = {Show and Tell: A Neural Image Caption Generator},
  
  publisher = {arXiv},
  
  year = {2014},
  
  copyright = {arXiv.org perpetual, non-exclusive license}
}

```