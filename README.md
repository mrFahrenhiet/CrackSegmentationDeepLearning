# Self Attention Based Efficient U-Net for Crack Segmentation

## Abstract
Crack detection, classification, and characterization are key
components of automatic structural health monitoring systems. Convolution
based encoder-decoder deep learning architecture have played a
significant role in developing crack segmentation models possessing limitations
in capturing the global context of the image. To overcome the
stated limitation, in the present study we propose a novel Self Attention
based Efficient U-Net which effectively tries to solve this limitation. The
proposed method achieved an F1 Score of 0.775, an IoU of 0.663 and
an accuracy of 97.3% on Crack500 dataset improving upon the current
state-of-the-art models.


## Train the model
```commandline
python train.py --path_imgs "path_to_image_folder" --path_masks "path_to_mask_folder"
```





## ToDo
- Complete the training script (only argparser left)
- Add evaluation script
- Add prediction script