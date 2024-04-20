# Animal Classification with CNNs
This notebook explores using convolutional neural networks (CNNs) to classify images from the "dataset_animals" dataset, featuring classes such as sheep, cow, butterfly, elephant, and squirrel. With a limited number of images per class, it demonstrates various strategies to achieve effective image classification.

# Contents
1. Basic CNN Architecture
- Architecture Details: An input layer followed by two convolutional layers with ReLU activation, a max pooling layer, a global average pooling layer, and a softmax output layer.
- Goal: Evaluate initial performance on both training and testing datasets.
2. Transfer Learning Enhancement
- Approach: Data augmentation and evaluation of three pre-trained Keras models: VGG16, ResNet50, and MobileNetV2.
- Metrics: Accuracy on training and testing datasets.
3. Optimization and Final Model Discussion
- Refinement of ResNet50: Extended training with early stopping to minimize validation error and fine-tuning of select layers to enhance performance.
- Model Architecture: Utilizes ResNet50 initialized with ImageNet weights, a GlobalAvgPool2D layer, a 512-node Dense layer (ReLU activation), and a softmax output layer.
- Hyperparameters: Adam optimizer with an adaptive learning rate, cross-entropy loss, and an early stop mechanism with a patience of 5 based on validation loss.
- Training Approach: Begins with Dense layers training while ResNet base layers are frozen. Two fine-tuning methods are explored, with one focusing on the last two layers of ResNet50.
