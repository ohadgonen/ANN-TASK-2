
1. CNN debugging

BEFORE CHANGE

Epoch 1/15
Train Loss: 2.1462
Train Accuracy: 18.57%
Test Loss: 1.9454
Test Accuracy: 27.02%
----------------------------------------
Epoch 2/15
Train Loss: 1.8762
Train Accuracy: 31.21%
Test Loss: 1.7391
Test Accuracy: 37.11%
----------------------------------------
Epoch 3/15
Train Loss: 1.6970
Train Accuracy: 38.33%
Test Loss: 1.6246
Test Accuracy: 40.92%
----------------------------------------
Epoch 4/15
Train Loss: 1.5666
Train Accuracy: 43.19%
Test Loss: 1.4983
Test Accuracy: 45.55%
----------------------------------------
Epoch 5/15
Train Loss: 1.4509
Train Accuracy: 47.25%
Test Loss: 1.4004
Test Accuracy: 48.94%
----------------------------------------
Epoch 6/15
Train Loss: 1.3752
Train Accuracy: 50.16%
Test Loss: 1.3394
Test Accuracy: 51.33%
----------------------------------------
Epoch 7/15
Train Loss: 1.3094
Train Accuracy: 52.39%
Test Loss: 1.2976
Test Accuracy: 52.57%
----------------------------------------
Epoch 8/15
Train Loss: 1.2577
Train Accuracy: 54.45%
Test Loss: 1.2516
Test Accuracy: 54.18%
----------------------------------------
Epoch 9/15
Train Loss: 1.2026
Train Accuracy: 56.62%
Test Loss: 1.2498
Test Accuracy: 54.75%
----------------------------------------
Epoch 10/15
Train Loss: 1.1515
Train Accuracy: 58.38%
Test Loss: 1.1714
Test Accuracy: 57.66%
----------------------------------------
Epoch 11/15
Train Loss: 1.0966
Train Accuracy: 60.66%
Test Loss: 1.1545
Test Accuracy: 59.08%
----------------------------------------
Epoch 12/15
Train Loss: 1.0435
Train Accuracy: 62.44%
Test Loss: 1.1038
Test Accuracy: 60.69%
----------------------------------------
Epoch 13/15
Train Loss: 0.9888
Train Accuracy: 64.65%
Test Loss: 1.0819
Test Accuracy: 61.38%
----------------------------------------
Epoch 14/15
Train Loss: 0.9451
Train Accuracy: 66.42%
Test Loss: 1.0458
Test Accuracy: 62.80%
----------------------------------------
Epoch 15/15
Train Loss: 0.9000
Train Accuracy: 68.03%
Test Loss: 1.0661
Test Accuracy: 62.67%
----------------------------------------


AFTER CHANGE

Epoch 1/15
Train Loss: 1.4337
Train Accuracy: 47.30%
Test Loss: 1.0490
Test Accuracy: 63.05%
----------------------------------------
Epoch 2/15
Train Loss: 0.9232
Train Accuracy: 67.57%
Test Loss: 0.8491
Test Accuracy: 70.54%
----------------------------------------
Epoch 3/15
Train Loss: 0.7273
Train Accuracy: 74.73%
Test Loss: 0.7832
Test Accuracy: 72.97%
----------------------------------------
Epoch 4/15
Train Loss: 0.5867
Train Accuracy: 79.69%
Test Loss: 0.7341
Test Accuracy: 74.81%
----------------------------------------
Epoch 5/15
Train Loss: 0.4860
Train Accuracy: 83.02%
Test Loss: 0.6756
Test Accuracy: 77.63%
----------------------------------------
Epoch 6/15
Train Loss: 0.3867
Train Accuracy: 86.42%
Test Loss: 0.7216
Test Accuracy: 77.46%
----------------------------------------
Epoch 7/15
Train Loss: 0.3057
Train Accuracy: 89.27%
Test Loss: 0.7812
Test Accuracy: 76.68%
----------------------------------------
Epoch 8/15
Train Loss: 0.2348
Train Accuracy: 91.79%
Test Loss: 0.8309
Test Accuracy: 76.90%
----------------------------------------
Epoch 9/15
Train Loss: 0.1935
Train Accuracy: 93.08%
Test Loss: 0.9289
Test Accuracy: 75.20%
----------------------------------------
Epoch 10/15
Train Loss: 0.1554
Train Accuracy: 94.59%
Test Loss: 1.0010
Test Accuracy: 76.38%
----------------------------------------
Epoch 11/15
Train Loss: 0.1310
Train Accuracy: 95.37%
Test Loss: 1.1542
Test Accuracy: 74.32%
----------------------------------------
Epoch 12/15
Train Loss: 0.1154
Train Accuracy: 95.98%
Test Loss: 1.2112
Test Accuracy: 75.62%
----------------------------------------
Epoch 13/15
Train Loss: 0.1170
Train Accuracy: 95.89%
Test Loss: 1.1149
Test Accuracy: 75.95%
----------------------------------------
Epoch 14/15
Train Loss: 0.0875
Train Accuracy: 96.94%
Test Loss: 1.2709
Test Accuracy: 75.22%
----------------------------------------
Epoch 15/15
Train Loss: 0.0936
Train Accuracy: 96.75%
Test Loss: 1.2542
Test Accuracy: 76.29%
----------------------------------------



The original InitialCNN used sigmoid activations after each convolutional layer. After training, I plotted the learning curves and the gradient magnitudes across epochs. The gradient plots showed that some layers changed much more than others during training, meaning that different parts of the network learned at different speeds.

The original model learned slowly over the 15 epochs. Training accuracy increased from 18.57% to 68.03%, while test accuracy increased from 27.02% to 62.67%. The best test accuracy reached by the original model was 62.80% at epoch 14.

To improve the architecture, We replaced the sigmoid activations with ReLU activations and added one BatchNorm2d layer after the first convolutional layer.  ReLU is commonly used in CNNs because it helps reduce the vanishing gradient problem and allows the network to learn faster.Batch normalization was added to keep the activations in a more stable range during training, which can improve training stability and learning speed. These changes kept the overall architecture almost the same while improving the training process.

The improved model learned substantially faster and achieved significantly better performance. Already in the first epoch, the improved model reached 63.05% test accuracy, which was approximately equal to the best performance of the original network after 15 epochs. The improved model eventually reached a best test accuracy of 77.63% at epoch 5, representing a major improvement over the original architecture.

However, after approximately epoch 5, the model began to overfit. While training accuracy continued increasing up to about 97%, the test loss steadily increased and the test accuracy stopped improving. This indicates that the network became very good at memorizing the training images, but less effective at classifying new unseen images.