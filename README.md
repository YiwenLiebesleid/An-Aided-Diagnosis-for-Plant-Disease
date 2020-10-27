# An-Aided-Diagnosis-for-Plant-Disease

---

Bachelor thesis *An Aided Diagnosis for Plant Diseases Based on Image Recognition*.

Date: 2019-01 ~ 2019-05.

* Dataset: https://github.com/spMohanty/PlantVillage-Dataset. I choose 22 classes from the 38 classes in raw dataset.

* Preprocess: I tried several ways, such as splitting RGB channels, changing the pictures to gray-scale, and doing k-clustering in every image to separate affected lesion from the rest part of the leaf. But all of these seems do no better than training the dataset directly.

* Models: It's obvious to use CNN to deal with CV problem.

At first I learned from https://github.com/machrisaa/tensorflow-vgg, and used VGG to train, I used the weights for transfer training, the accuracy was 0.95+. But I think VGG is not suitable for my task, for my data wasn't adequate, and VGG is also memory-consuming and the training is really slow.

Then I use ResNet for my task. I use weights trained from ImageNet to initialize, and also data augmentation was done to reduce overfitting. For this model, the accuracy is 0.98+. I think to train with all of the classes at one time is really inconvenient and quite time-consuming, especially when there's something wrong or we need to add more classes into the trained model. So I tried to improve my work using incremental learning.

I got the idea from *Tree-CNN: a hierarchical deep convolutional neural network for incremental learning*. I divide the whole dataset by the kind of plant, this will make 4 super-classes. Every super-class is trained and then the whole dataset is trained. This makes me able to train four super-classes parallelly on four GPUs, which improves the training efficiency. Every super-class, and the sub-classes of them are trained to have an accuracy of 0.99+.
