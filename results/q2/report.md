2. Comparison to MLP

MLP

Number of trainable parameters: 1707274
Epoch 1/15
Train Loss: 1.6279
Train Accuracy: 42.22%
Test Loss: 1.4804
Test Accuracy: 46.83%
Epoch Time: 4.87 seconds
----------------------------------------
Epoch 2/15
Train Loss: 1.4086
Train Accuracy: 50.16%
Test Loss: 1.3967
Test Accuracy: 50.54%
Epoch Time: 4.60 seconds
----------------------------------------
Epoch 3/15
Train Loss: 1.2965
Train Accuracy: 54.22%
Test Loss: 1.3646
Test Accuracy: 52.75%
Epoch Time: 4.59 seconds
----------------------------------------
Epoch 4/15
Train Loss: 1.2011
Train Accuracy: 57.70%
Test Loss: 1.4090
Test Accuracy: 51.74%
Epoch Time: 4.65 seconds
----------------------------------------
Epoch 5/15
Train Loss: 1.1137
Train Accuracy: 60.73%
Test Loss: 1.3491
Test Accuracy: 52.54%
Epoch Time: 4.56 seconds
----------------------------------------
Epoch 6/15
Train Loss: 1.0310
Train Accuracy: 63.73%
Test Loss: 1.3537
Test Accuracy: 54.10%
Epoch Time: 4.56 seconds
----------------------------------------
Epoch 7/15
Train Loss: 0.9478
Train Accuracy: 66.51%
Test Loss: 1.3957
Test Accuracy: 53.96%
Epoch Time: 4.58 seconds
----------------------------------------
Epoch 8/15
Train Loss: 0.8720
Train Accuracy: 69.16%
Test Loss: 1.4232
Test Accuracy: 53.43%
Epoch Time: 4.51 seconds
----------------------------------------
Epoch 9/15
Train Loss: 0.7966
Train Accuracy: 71.73%
Test Loss: 1.5003
Test Accuracy: 53.61%
Epoch Time: 4.49 seconds
----------------------------------------
Epoch 10/15
Train Loss: 0.7270
Train Accuracy: 74.13%
Test Loss: 1.5429
Test Accuracy: 54.33%
Epoch Time: 4.51 seconds
----------------------------------------
Epoch 11/15
Train Loss: 0.6508
Train Accuracy: 76.92%
Test Loss: 1.6246
Test Accuracy: 53.24%
Epoch Time: 4.56 seconds
----------------------------------------
Epoch 12/15
Train Loss: 0.5960
Train Accuracy: 78.79%
Test Loss: 1.7293
Test Accuracy: 53.44%
Epoch Time: 4.65 seconds
----------------------------------------
Epoch 13/15
Train Loss: 0.5335
Train Accuracy: 81.33%
Test Loss: 1.8316
Test Accuracy: 53.45%
Epoch Time: 4.57 seconds
----------------------------------------
Epoch 14/15
Train Loss: 0.4872
Train Accuracy: 82.68%
Test Loss: 1.8890
Test Accuracy: 53.67%
Epoch Time: 4.68 seconds
----------------------------------------
Epoch 15/15
Train Loss: 0.4341
Train Accuracy: 84.72%
Test Loss: 2.0337
Test Accuracy: 53.69%
Epoch Time: 4.75 seconds
----------------------------------------

CNN

Number of trainable parameters: 653450
Epoch 1/15
Train Loss: 1.4337
Train Accuracy: 47.30%
Test Loss: 1.0490
Test Accuracy: 63.05%
Epoch Time: 44.18 seconds
----------------------------------------
Epoch 2/15
Train Loss: 0.9232
Train Accuracy: 67.57%
Test Loss: 0.8491
Test Accuracy: 70.54%
Epoch Time: 43.85 seconds
----------------------------------------
Epoch 3/15
Train Loss: 0.7273
Train Accuracy: 74.73%
Test Loss: 0.7832
Test Accuracy: 72.97%
Epoch Time: 43.21 seconds
----------------------------------------
Epoch 4/15
Train Loss: 0.5867
Train Accuracy: 79.69%
Test Loss: 0.7341
Test Accuracy: 74.81%
Epoch Time: 43.41 seconds
----------------------------------------
Epoch 5/15
Train Loss: 0.4860
Train Accuracy: 83.02%
Test Loss: 0.6756
Test Accuracy: 77.63%
Epoch Time: 43.15 seconds
----------------------------------------
Epoch 6/15
Train Loss: 0.3867
Train Accuracy: 86.42%
Test Loss: 0.7216
Test Accuracy: 77.46%
Epoch Time: 42.94 seconds
----------------------------------------
Epoch 7/15
Train Loss: 0.3057
Train Accuracy: 89.27%
Test Loss: 0.7812
Test Accuracy: 76.68%
Epoch Time: 43.45 seconds
----------------------------------------
Epoch 8/15
Train Loss: 0.2348
Train Accuracy: 91.79%
Test Loss: 0.8309
Test Accuracy: 76.90%
Epoch Time: 42.88 seconds
----------------------------------------
Epoch 9/15
Train Loss: 0.1935
Train Accuracy: 93.08%
Test Loss: 0.9289
Test Accuracy: 75.20%
Epoch Time: 42.88 seconds
----------------------------------------
Epoch 10/15
Train Loss: 0.1554
Train Accuracy: 94.59%
Test Loss: 1.0010
Test Accuracy: 76.38%
Epoch Time: 42.37 seconds
----------------------------------------
Epoch 11/15
Train Loss: 0.1310
Train Accuracy: 95.37%
Test Loss: 1.1542
Test Accuracy: 74.32%
Epoch Time: 42.37 seconds
----------------------------------------
Epoch 12/15
Train Loss: 0.1154
Train Accuracy: 95.98%
Test Loss: 1.2112
Test Accuracy: 75.62%
Epoch Time: 42.65 seconds
----------------------------------------
Epoch 13/15
Train Loss: 0.1170
Train Accuracy: 95.89%
Test Loss: 1.1149
Test Accuracy: 75.95%
Epoch Time: 42.66 seconds
----------------------------------------
Epoch 14/15
Train Loss: 0.0875
Train Accuracy: 96.94%
Test Loss: 1.2709
Test Accuracy: 75.22%
Epoch Time: 42.74 seconds
----------------------------------------
Epoch 15/15
Train Loss: 0.0936
Train Accuracy: 96.75%
Test Loss: 1.2542
Test Accuracy: 76.29%
Epoch Time: 42.56 seconds
----------------------------------------

Model | Best Test Accuracy | Best Epoch | Parameters | Approx. Memory | Avg. Epoch Time
CNN   | 77.63%             | 5          | 653,450    | 2.61 MB        | ~43.01 sec
MLP   | 54.33%             | 10         | 1,707,274  | 6.83 MB        | ~4.61 sec

(Approximate model memory was estimated by multiplying the number of trainable parameters by 4 bytes, assuming the parameters were stored as 32-bit floating point (float32) values).

The CNN first-layer filters are easier to interpret because they act as small local feature detectors, such as color or edge detectors. In contrast, the MLP first-layer weights connect to the entire flattened image, so their visualizations are more global and less spatially organized. This makes the CNN more interpretable for image data.

The visualizations of the optimized first-layer weights showed noticeable differences between the CNN and the MLP architectures. In the CNN, the first-layer filters appeared as small localized color patterns with relatively clear spatial structure. Since convolutional filters operate on small regions of the image, the learned weights remained spatially localized and easier to visualize directly.

In contrast, the MLP first-layer weights appeared more diffuse and less spatially organized. Because the MLP flattens each image into a single long vector, the learned weights were distributed across the entire image rather than being localized to small spatial regions. As a result, the visualized weights appeared more global and more difficult to interpret visually compared to the CNN filters.

These differences are consistent with the different design principles of the architectures. CNNs preserve spatial structure through convolutional operations, making them well suited for image-related tasks such as image classification and object recognition. MLPs treat the image as a general vector of values and therefore do not explicitly preserve spatial relationships between neighboring pixels. On CIFAR-10, the CNN achieved substantially higher test accuracy while using fewer parameters, demonstrating the advantage of convolutional architectures for image data.