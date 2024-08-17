# 交通标志六分类任务（TrafficSignClassification）
这是个基于深度学习的交通标志六分类练习，以及经典主流图像分类模型的PyTorch复现项目。该项目中，我在自建的交通标志数据集上进行了主流图像分类网络的性能对比，如VGGNet、GoogLeNet、ResNet、ViT等，并不定期追加新的分类模型。该项目主要用于练习pytorch，学习、实现并总结各大主流模型的构建。它也适合深度学习入门练手，分享给大家参考交流，有问题欢迎一起探讨。

This is a deep learning-based traffic sign six-class classification practice, as well as a PyTorch implementation project for classic mainstream image classification models. In this project, I compared the performance of mainstream image classification networks such as VGGNet, GoogLeNet, ResNet and ViT on a self-built traffic sign dataset, and I occasionally add new classification models. The project mainly serves as practice with PyTorch, focusing on learning, implementing, and summarizing the construction of various mainstream models. It is also suitable for beginners in deep learning, and I am sharing it for reference and discussion. Feel free to reach out if you have any questions.

## 1. 数据集介绍（Dataset Introduction）
本项目使用的数据集是自建数据集，数据样本来自COCO数据集、ImageNet数据集以及网页图像爬取。去除损坏图像、无效图像后，数据集共包含1,662张图像。此外，为了保证数据集的复杂度、多样性，数据搜集时采取了不同视角、大小、画质的图像。

The dataset used in this project is a self-constructed dataset, with data samples collected from the COCO dataset, ImageNet dataset, and web image scraping. After removing damaged and invalid images, the dataset contains a total of 1,662 images. Additionally, to ensure the complexity and diversity of the dataset, images were collected from different perspectives, sizes, and quality levels.

数据集的6类分别为红灯、黄灯、绿灯，左转、右转、停止，对应的标注label依次为：0、1、2、3、4、5。所有图像都经过预处理，剪裁去除了多余的背景区域，并将所需识别的交通标志集中在中央区域，同时将图像尺寸调整至224×224大小，如下所示。

The dataset consists of six classes: red light, yellow light, green light, left turn, right turn, and stop, with corresponding labels assigned as 0, 1, 2, 3, 4, and 5, respectively. All images have undergone preprocessing, where unnecessary background areas were cropped out, focusing the traffic signs to be recognized in the central region, and the image size was adjusted to 224×224 pixels, as shown below.

<div align="center">
  <img src="https://github.com/user-attachments/assets/d9e991f5-36c3-4ebb-ab20-b32a60e4ba28" width="200"/>
  <img src="https://github.com/user-attachments/assets/0a013877-da0e-4621-b787-a9c67758765c" width="200"/>
  <img src="https://github.com/user-attachments/assets/cfa2723a-6b22-42a2-806f-f4f89e331f3e" width="200"/>
</div>

<div align="center">
  <img src="https://github.com/user-attachments/assets/1443fc11-783f-46e7-9ce2-05d76119e1cc" width="200"/>
  <img src="https://github.com/user-attachments/assets/da53307a-311d-4d05-89a7-a2ca5c9d350a" width="200"/>
  <img src="https://github.com/user-attachments/assets/712aa81f-0c89-4015-bf55-cebdec9f0f7d" width="200"/>
</div>

以下是数据集的分享链接：

Here is the link to the dataset:

Baidu Cloud：https://pan.baidu.com/s/1xi1l9KLTy0o6uA4xp2geVQ?pwd=z6sv

Google Drive：https://drive.google.com/file/d/1VndBrdNTMxcldmKE2hjFjJ2g-SaxRZ46/view?usp=sharing

数据集压缩包包含images-v0、images-v1、labels-v0.csv、labels-v1.csv四个文件。其中images-v0是原始的数据集图像，共1,662张，labels-v0.csv是对应的标注文件。images-v1在images-v0的基础上进行了数据增强，每张图片随机进行左右翻转、hsv抖动或放大缩小等操作，共3,324张图像，labesl-v1.csv是对应的标注文件。下图是数据增强的效果示例。

The dataset archive includes four files: images-v0, images-v1, labels-v0.csv, and labels-v1.csv. images-v0 contains the original dataset images, totaling 1,662 images, and labels-v0.csv is the corresponding annotation file. images-v1 is an augmented version of images-v0, where each image has undergone random transformations such as horizontal flipping, HSV jittering, or scaling, resulting in a total of 3,324 images. labels-v1.csv is the annotation file for these augmented images. The image below illustrates examples of the data augmentation effects.

<div align="center">
  <img src="https://github.com/user-attachments/assets/7a114c12-7317-4af9-9545-a05c1598f916" width="200"/>
  <img src="https://github.com/user-attachments/assets/0eedbcb8-db09-46e2-a6a2-08a66152fa7b" width="200"/>
</div>

<div align="center">
  <img src="https://github.com/user-attachments/assets/2a38a3bc-be4a-4ed7-861b-9acd2b857fe6" width="200"/>
  <img src="https://github.com/user-attachments/assets/099a712c-a143-4703-a5fd-afc01efccc76" width="200"/>
</div>

## 2. 代码工程介绍（Code Project Introduction）

