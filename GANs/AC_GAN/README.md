# AC GAN 
Simple Implementation of AC GAN in PyTorch. 

### Dataset
The cifar10 dataset is used and can be downloaded from PyTorch - dataset itself.

### Training 
Edit the `config.py` file to match the setup you want to use. Then run `train.py`


You can directly use this [Kaggle NoteBook](https://www.kaggle.com/sushant097/ac-gan-on-cifar10-with-pytorch) to run in kaggle.

#### Annotated ACGAN Paper & Summary[Annotated AC GAN Paper](https://github.com/sushant097/annotated_research_papers/blob/master/GANs/ACGAN-2016.pdf)

## Output
Trained for 110 epochs in Cifar10 dataset. Output:

![](generated_images/ac_gan_final_output.gif)



## AC GAN paper
Conditional Image Synthesis With Auxiliary Classifier GANs by Augustus Odena, Christopher Olah, Jonathon Shlens


### Abstract
Synthesizing high resolution photorealistic images has been a long-standing challenge in machine learning. In this paper we introduce new methods for the improved training of generative adversarial networks (GANs) for image synthesis. We construct a variant of GANs employing label conditioning that results in 128x128 resolution image samples exhibiting global coherence. We expand on previous work for image quality assessment to provide two new analyses for assessing the discriminability and diversity of samples from class-conditional image synthesis models. These analyses demonstrate that high resolution samples provide class information not present in low resolution samples. Across 1000 ImageNet classes, 128x128 samples are more than twice as discriminable as artificially resized 32x32 samples. In addition, 84.7% of the classes have samples exhibiting diversity comparable to real ImageNet data.

```bash
@misc{odena2017conditional,
      title={Conditional Image Synthesis With Auxiliary Classifier GANs}, 
      author={Augustus Odena and Christopher Olah and Jonathon Shlens},
      year={2017},
      eprint={1610.09585},
      archivePrefix={arXiv},
      primaryClass={stat.ML}
}
```