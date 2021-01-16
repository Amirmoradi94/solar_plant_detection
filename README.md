# solar_detection
[*Automatic Boundary Extraction of Large-Scale Photovoltaic Plants Using a Fully Convolutional Network on Aerial Imagery*](https://ieeexplore.ieee.org/document/9095250)
## Overview
This study presents a novel method for boundary extraction of Photovoltaic (PV) plants using a Fully Convolutional Network (FCN). Extracting the boundaries of PV plants is essential in the process of aerial inspection and autonomous monitoring by aerial robots. The presented deep learning based method provides a clear delineation of the utility-scale PV plants’ boundaries for PV developers, Operation and Maintenance (O&M) service providers for use in aerial photogrammetry, flight mapping, and path planning during the autonomous monitoring of PV plants. For this purpose, as a prerequisite, the “Amir” dataset consisting of aerial images of PV plants from different countries, has been collected. A Mask RCNN architecture is employed as a deep network with customized VGG16 as encoder to detect the boundaries precisely. As comparison, the results of another framework based on Classical Image Processing (CIP) are compared with the proposed encoder-decoder network's performance in PV plants boundary detection. The results of the this fully convolutional network demonstrate that the trained model is able to detect the boundaries of PV plants with an accuracy of 96.99% and site-specific tuning of boundary parameters is no longer required. 
## Concept and Algorithm
The FCN network pipeline is an extension of the classical CNN by conversion of fully-connected layers to convolutional ones. The input of our FCN is an RGB image, and the output is the predicted mask of the PV plants. The Mask-RCNN aims to solve instance segmentation problems. It is a two-stage framework. The first stage scans the image and generates proposals (areas likely to contain an object). The second stage classifies the proposals and generates masks in pixel level. This structure is a kind of FCN. The FCN uses convolutional and pooling layers to down-sample the features map of an image.
### Amir Dataset
“Amir” dataset has been developed for use in autonomous monitoring of large-scale PV plants. This dataset includes different aerial imageries collected from a wide range of large-scale PV plants located in twelve countries and six continents. In Amir dataset, all necessary data has been provided to train an accurate model for an autonomous boundary detection, including aerial imagery of PV strings and their binary masks as labels.
\
\
![alt text](https://github.com/Amirmoradi94/solar_detection/blob/main/Others/masks.jpg)

**Amir consists of 3580 aerial images from mainly large-scale PV plants. Amir dataset is available on [IEEE DataPort](https://ieee-dataport.org/documents/aerial-imagery-pv-plants-boundary-detection)**

![alt text](https://github.com/Amirmoradi94/solar_detection/blob/main/Others/countries.jpg)

### Model Preparation
One popular approach for boundary extraction is to follow the Mask RCNN structure in which the spatial resolution of the input is down-sampled, and lower-resolution features map are obtained with high efficiency among classes. Up-sampling is then performed by transposing convolutions into a full- resolution segmentation map. To use the VGG16 as a backbone, the input shape of the first layer of the model should be defined. Furthermore, to achieve the highest accuracy and lowest losses in the training and evaluation process, ImageNet’s pre-trained weights are used and the model’s weights are initialized in order to prevent random initialization. This method is called the **Transfer Learning** technique, and is used to enhance the coverage pace of accuracy and loss diagrams. 

![alt text](https://github.com/Amirmoradi94/solar_detection/blob/main/Others/encoder-decoder.jpg)

## Results
In Amir dataset, 80% (2864 samples) of aerial images are selected randomly to train the model; then, the trained model is examined by the testing samples (716 samples). The training process is finalized after 16 epochs. The accuracy of training and testing is up to 97.61% and 96.99%, respectively.
- Testing Accuracy: 96.93 %
- Training Accuracy: 97.61 %
- Testing Loss: 8.81 %
- Training Loss: 5.74 %

![alt text](https://github.com/Amirmoradi94/solar_detection/blob/main/Others/FCN_results.jpg)

## Run
This encoder-decoder architecture has been optimized to be run on Google Colab. For this purpose, you need upload the whole dataset to Google Drive and provide its shareable link into the **_Boundary_Detection.ipynb_** notebook. By some simple modifications, you can run it on your personal processing unit. \
Therefore, download the dataset and the notebook from the repository, and then, upload them to your personal google drive. Finally, open the **_Boundary_Detection.ipynb_** notebook, and follow the cells' steps. \
**Note:** The dataset contains the following folders.
1. train
     - x: Image
     - y: Mask
2. test
     - x: Image
     - y: Mask

## Citation
A. M. Moradi Sizkouhi, M. Aghaei, S. M. Esmailifar, M. R. Mohammadi and F. Grimaccia, "Automatic Boundary Extraction of Large-Scale Photovoltaic Plants Using a Fully Convolutional Network on Aerial Imagery," in IEEE Journal of Photovoltaics, vol. 10, no. 4, pp. 1061-1067, July 2020, doi: 10.1109/JPHOTOV.2020.2992339.
