# Halloween Candy Classification Using Neural Networks and Backpropagation

By Aissha Monceda and Sophia Logarta | F2

## Project Overview

This project implements a **feedforward neural network with backpropagation in C#** to classify Halloween candies popularity into three categories: **Low, Medium, and High**, based on their attributes and associated win percentages.

The neural network learns patterns from candy characteristics using supervised learning. It uses forward propagation to generate predictions and backpropagation to adjust weights and biases based on prediction errors.

The implementation incorporates data preprocessing, Min-Max normalization, categorical cross-entropy loss, L2 regularization (weight decay), and momentum to support the training process and improve model generalization.

## Objectives

The main objectives of this project are to:

- Develop a functional feedforward neural network using C#.
- Implement forward propagation and backpropagation from scratch.
- Apply data preprocessing and Min-Max normalization to the input features.
- Classify Halloween candies into Low, Medium, and High win percentage categories.
- Experiment with different training configurations and hyperparameters.
- Evaluate model performance using training loss, testing loss, and classification accuracy.

## Dataset Description

### Data Source

The project uses `candy-data.csv`, a dataset containing information about different Halloween candies.

[Link to the original dataset.](https://www.kaggle.com/datasets/fivethirtyeight/the-ultimate-halloween-candy-power-ranking/data)

The data was collected from a website where participants chose which of two fun-sized candies they would rather receive. More than 269,000 votes were collected from 8,371 different IP addresses. Each row of the CSV is one candy.
 
- **85 candies (rows)**, 13 columns
- **No missing values**
- The loader still skips and reports any row with the wrong number of columns or a value that cannot be parsed

### Features
 
| Feature | Type | Meaning |
|---|---|---|
| `chocolate` | Binary (0/1) | Contains chocolate |
| `fruity` | Binary | Fruit flavored |
| `caramel` | Binary | Contains caramel |
| `peanutyalmondy` | Binary | Contains peanuts, peanut butter, or almonds |
| `nougat` | Binary | Contains nougat |
| `crispedricewafer` | Binary | Contains crisped rice, wafers, or a cookie component |
| `hard` | Binary | Hard candy |
| `bar` | Binary | Candy bar |
| `pluribus` | Binary | One of many candies in a bag or box |
| `sugarpercent` | Continuous (0–1) | Sugar percentile within the dataset |
| `pricepercent` | Continuous (0–1) | Unit price percentile within the dataset |
 
`competitorname` is only an identifier. It is kept for display in the prediction tables and is **not** used as an input.
 
`winpercent` (the overall win percentage across about 269,000 matchups) is used **only to create the target class**. It is deliberately excluded from the inputs, because including it would leak the answer to the network.

### Target Classes

The continuous win percentage is converted into three categorical classes:

| Class | Description |
|---|---|
| Low | Candies with relatively low win percentages |
| Medium | Candies with moderate win percentages |
| High | Candies with relatively high win percentages |

The classification thresholds are determined by the data preparation logic in `CSVLoader.BuildTierRows()`.

The target classes are represented using one-hot encoding:

| Class | Encoded Representation |
|---|---|
| Low | `[1, 0, 0]` |
| Medium | `[0, 1, 0]` |
| High | `[0, 0, 1]` |

### Data Preprocessing

The following preprocessing steps are performed:

1. **CSV Loading:** Reads the candy dataset and extracts the input features, win percentages, and candy names.
2. **Target Preparation:** Converts continuous win percentages into three categorical classes.
3. **Train-Test Split:** Randomly divides the dataset into 80% training data and 20% testing data.
4. **Min-Max Normalization:** Scales the 11 input features to a range of `[0, 1]`.
5. **Data Leakage Prevention:** The normalization parameters are learned exclusively from the training dataset and subsequently applied to both training and testing data.

The Min-Max normalization formula is:

$$X_{normalized}=\frac{X-X_{min}}{X_{max}-X_{min}}$$

Constant-value columns are assigned a normalized value of zero to prevent division-by-zero errors.

## Neural Network Architecture

The project uses a three-layer feedforward neural network with the following architecture:

**11 – 5 – 3**

| Layer | Number of Neurons | Function |
|---|---:|---|
| Input Layer | 11 | Receives normalized candy attributes |
| Hidden Layer | 5 | Learns patterns and relationships between features |
| Output Layer | 3 | Produces probabilities for Low, Medium, and High classes |

### Activation Functions

**1. Hyperbolic Tangent (Tanh)**

The hidden layer uses the Tanh activation function to introduce nonlinearity into the network. Its output ranges from -1 to 1.

$$\tanh(x)=\frac{e^x-e^{-x}}{e^x+e^{-x}}$$

**2. Softmax**

The output layer uses Softmax to convert the output scores into probabilities.

$$P(y_i)=\frac{e^{z_i}}{\sum_j e^{z_j}}$$

The class with the highest predicted probability is the final prediction (winner-takes-all).

## Learning and Optimization Techniques

### 1. Forward Propagation

Forward propagation passes the normalized input features through the network.

The input neurons transmit information to the hidden layer, where weighted sums and biases are calculated. The Tanh activation function is then applied.

The hidden layer outputs are passed to the output layer, where Softmax produces the predicted class probabilities.

### 2. Backpropagation

Backpropagation calculates how much each weight and bias contributes to the prediction error.

The process involves:

1. Comparing predicted probabilities with actual target values.
2. Calculating output-layer error signals.
3. Computing hidden-to-output weight gradients.
4. Propagating error signals backward through the hidden layer.
5. Computing input-to-hidden weight gradients.
6. Updating weights and biases using the calculated gradients.

### 3. Categorical Cross-Entropy Loss

The implementation supports categorical cross-entropy as its primary loss function.

$$L=-\sum_{i=1}^{3}t_i\log(y_i)$$

Where:
- $t_i$ represents the actual target value.
- $y_i$ represents the predicted probability.

Cross-entropy measures the difference between the actual class distribution and the predicted probability distribution.

The implementation also includes an optional Mean Squared Error (MSE) loss configuration for experimentation.

### 4. L2 Regularization (Weight Decay)

Weight decay discourages excessively large weights by introducing a regularization penalty.

$$L_{total}=L_{prediction}+\frac{\lambda}{2}\sum w^2$$

This helps reduce overfitting by encouraging smaller weight values.

The regularization term is applied to the input-to-hidden and hidden-to-output weights.

### 5. Momentum

Momentum incorporates previous weight adjustments into the current update.

$$\Delta w_t=\eta(g_t-\lambda w_t)+\alpha\Delta w_{t-1}$$

Where:
- $\eta$ = the learning rate.
- $g_t$ = the current gradient under the implementation's update convention.
- $\lambda$ = the weight decay coefficient.
- $\alpha$ = the momentum coefficient.

## Hyperparameters

The current implementation uses the following configuration:

| Hyperparameter | Value | Description |
|---|---:|---|
| Input Neurons | 11 | Number of candy attributes |
| Hidden Neurons | 5 | Number of hidden-layer neurons |
| Output Neurons | 3 | Number of classification categories |
| Maximum Epochs | 1,000 | Maximum number of training iterations |
| Learning Rate | 0.05 | Controls the size of weight updates |
| Momentum | 0.01 | Controls the influence of previous updates |
| Weight Decay | 0.01 | L2 regularization coefficient |
| Loss Function | Cross-Entropy | Measures classification prediction error |
| Training Ratio | 80% | Proportion of data used for training |
| Testing Ratio | 20% | Proportion of data reserved for testing |
| Random Seed | 1 | Ensures reproducible data splitting |
| Logging Interval | 100 | Reports loss every 100 epochs |

These are the initial experimental settings. Additional configurations can be tested to determine their effects on model performance.

## Training Log and Output

During training, the program calculates and displays training loss and testing loss at regular intervals.

The logging interval is currently set to 100 epochs.

Example console output format:

```text
[hh:mm:ss] Loading candy-data.csv...
[hh:mm:ss] Loaded 85 candies with 11 features.
[hh:mm:ss] Tiers: Low < 42.18% | Medium 42.18%-55.38% | High >= 55.38%
[hh:mm:ss] Split: 68 train / 17 test.
[hh:mm:ss] Features normalized [0,1] on training data.
[hh:mm:ss] Network: 11-5-3 | epochs=1000 lr=0.0500 mom=0.0100 wd=0.0100 cross-entropy=True
[hh:mm:ss] Training...
[hh:mm:ss] epoch = 100   trainLoss = 0.8328   testLoss = 0.7305
[hh:mm:ss] epoch = 200   trainLoss = 0.7977   testLoss = 7699
...
[hh:mm:ss] epoch = 1000  trainLoss = 0.7932   testLoss = 0.7779
[hh:mm:ss] Done!  Train accuracy = 69.12%  |  Test accuracy = 82.35%
```

### CSV Training Log

`training_log.csv` is created in the program's output folder (next to the executable, for example `bin/Debug/net.../`). It is **overwritten on every training run**, so copy or rename it after each experiment if you want to keep it (for example `training_log_lr0.1.csv`).
 
| Column | Description |
|---|---|
| epoch | Training epoch |
| trainLoss | Loss on the training set (includes the weight-decay penalty) |
| testLoss | Loss on the test set (includes the weight-decay penalty) |

### Training Loss Visualization

The training log can be plotted to visualize how the model's loss changes over time.

The expected objective is to observe a reduction in training loss as the network learns from the dataset.

Comparing training and testing loss can also provide insight into whether the model is overfitting or generalizing effectively.

## Model Evaluation

After training, the program evaluates the model using classification accuracy on both training and testing datasets.

### Accuracy

Accuracy measures the proportion of correctly classified examples.

$$Accuracy=\frac{\text{Correct Predictions}}{\text{Total Predictions}}$$

The program uses a winner-takes-all approach, selecting the class with the highest predicted probability.

### Evaluation Output

```text
Final accuracy on training data = [48/68]
Final accuracy on test data     = [9/17]
```

### Performance Analysis

The difference between training and testing accuracy provides an initial indication of how well the network generalizes.

- **High training and testing accuracy:** Indicates that the network performs well on both learned and unseen examples.
- **High training accuracy but low testing accuracy:** May indicate overfitting.
- **Low training and testing accuracy:** May indicate underfitting or insufficient training.

Further evaluation using a confusion matrix, precision, recall, and F1-score can provide a more detailed understanding of classification performance across the three categories.

## Hyperparameter Experimentation

The project allows experimentation with different hyperparameter configurations to observe their effects on model performance.

The following parameters can be modified in `Program.cs`:

- Number of hidden neurons (`numHidden`)
- Maximum training epochs (`maxEpochs`)
- Learning rate (`learnRate`)
- Momentum (`momentum`)
- Weight decay (`weightDecay`)
- Loss function (`useCrossEntropy`)

### Experimentation Table

Use the following table to document the results of your experiments:

| # | Experiment | Hidden | Epochs | Learning Rate | Momentum | Weight Decay | Loss | Final Train Loss | Final Test Loss | Train Acc | Test Acc |
|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
| 1 | Baseline | 5 | 1,000 | 0.05 | 0.01 | 0.01 | CE | 0.7932 | 0.7779 | 69.12% | 82.35% |
| 2 | Lower learning rate | 5 | 1,000 | 0.01 | 0.01 | 0.01 | CE | 0.7741 | 0.7951 | 69.12% | 76.47% |
| 3 | Higher learning rate | 5 | 1,000 | 0.10 | 0.01 | 0.01 | CE | 0.8504 | 0.7748 | 66.18% | 82.35% |
| 4 | Many hidden nodes | 12 | 1,000 | 0.05 | 0.01 | 0.01 | CE | 0.8371 | 0.8286 | 72.06% | 76.47% |
| 5 | Fewer epochs | 5 | 500 | 0.05 | 0.01 | 0.01 | CE | 0.8118 | 0.8177 | 72.06% | 76.47% |
| 6 | More epochs | 5 | 2,000 | 0.05 | 0.01 | 0.01 | CE | 0.7803 | 0.8021 | 67.65% | 76.47% |
| 7 | No weight decay | 5 | 1,000 | 0.05 | 0.01 | 0.00 | CE | 0.2627 | 1.6904 | 88.24% | 58.82% |
| 8 | Lower weight decay | 5 | 1,000 | 0.05 | 0.01 | 0.001 | CE | 0.4344 | 1.0238 | 83.82% | 64.71% |
| 9 | Higher momentum | 5 | 1,000 | 0.05 | 0.50 | 0.01 | CE | 0.8209 | 0.7600 | 66.18% | 82.35% |
| 10 | MSE loss | 5 | 1,000 | 0.05 | 0.01 | 0.01 | MSE | 0.5263 | 0.4791 | 60.29% | 70.59% |

The purpose of these experiments is to identify a configuration that achieves good classification performance while maintaining consistent training and testing results.

The optimal configuration should be selected based on actual experimental results rather than training accuracy alone.

## Project Structure

The project is organized into separate classes to maintain modularity and make the implementation easier to understand and maintain.

```text
BackPropHalloweenCandy/
│
├── BackPropHalloweenCandy.sln
│
├── BackPropHalloweenCandy/
│   │
│   ├── Program.cs
│   ├── NeuralNetwork.cs
│   ├── Normalizer.cs
│   ├── CSVLoader.cs
│   │
│   ├── candy-data.csv
│   ├── training_log.csv
│   │
│   └── ...
│
└── README.md
```

### Main Components

| File | Responsibility |
|---|---|
| `Program.cs` | Main execution flow, hyperparameter configuration, data splitting, normalization, training, and evaluation |
| `NeuralNetwork.cs` | Neural network architecture, weight initialization, forward propagation, backpropagation, loss calculation, and accuracy evaluation |
| `Normalizer.cs` | Min-Max normalization of input features |
| `CSVLoader.cs` | Reads the candy dataset and prepares the classification targets |
| `candy-data.csv` | Input dataset |
| `training_log.csv` | Exported training and testing loss values |

## Requirements

To run the project, you will need:

- Windows operating system or a compatible .NET environment.
- Visual Studio Community or another C# development environment.
- A compatible .NET SDK/runtime.
- The provided `candy-data.csv` dataset.

## How to Run the Project

### 1. Clone the Repository

```bash
git clone https://github.com/binggbongg/BackPropHalloweenCandy.git
```

### 2. Open the Project

Open `BackPropHalloweenCandy.sln` in Visual Studio.

Alternatively, navigate to the project directory using a terminal.

### 3. Prepare the Dataset

Ensure that `candy-data.csv` is included in the project.

In Visual Studio, configure the CSV file with:

**Copy to Output Directory: Copy if newer**

This ensures that the program can locate the dataset when executed.

### 4. Configure the Hyperparameters

Open `Program.cs` and modify the desired training settings:

```csharp
int numHidden = 5;

int maxEpochs = 1000;
double learnRate = 0.05;
double momentum = 0.01;

bool useCrossEntropy = true;
double weightDecay = 0.01;
```

### 5. Execute the Program

Run the project using Visual Studio or the following command:

```bash
dotnet run
```

The program will:

1. Load the candy dataset.
2. Prepare the classification targets.
3. Split the data into training and testing datasets.
4. Normalize the input features.
5. Initialize the neural network.
6. Train the network using backpropagation.
7. Display training and testing loss.
8. Generate `training_log.csv`.
9. Display the final model weights and biases.
10. Calculate training and testing accuracy.

## Limitations and Future Improvements

Although the implementation provides a working foundation for neural network classification, several improvements can be explored:

- **Hyperparameter Optimization:** Experiment with different hidden-layer sizes, learning rates, momentum values, and epoch counts.
- **Additional Evaluation Metrics:** Implement a confusion matrix, precision, recall, and F1-score.
- **Early Stopping:** Save the best-performing model based on validation loss instead of always returning the final epoch's weights.
- **Model Checkpointing:** Save and reload trained weights for future predictions.
- **Improved Regularization:** Align the regularization scaling in the training updates with the reported average loss.
- **Improved Loss Function Alternatives:** Implement a mathematically complete Softmax-MSE gradient for a more rigorous comparison with cross-entropy.
- **Visualization:** Generate loss curves to make training behavior and model performance easier to interpret.

## Deliverables

This repository contains the following project deliverables:

| Deliverable | Description |
|---|---|
| Source Code | C# implementation of the neural network engine, CSV data loading, normalization, and training process |
| Training Log | Exported CSV file containing training and testing loss across epochs |
| Data Description | Explanation of the candy dataset, input attributes, and target classification categories |
| Performance Evaluation | Training and testing accuracy generated after model training |
| Documentation | README containing the project overview, architecture, methodology, hyperparameters, and execution instructions |

## Conclusion

This project demonstrates the implementation of a feedforward neural network using C# and backpropagation for a three-class classification problem.

Through the combination of data preprocessing, forward propagation, categorical cross-entropy, weight decay, and momentum, the network learns to identify patterns in candy attributes and predict their corresponding win percentage categories.

The project also provides an opportunity to examine the effects of different hyperparameters, understand the learning process, and evaluate how well a neural network generalizes to unseen data.

By implementing the neural network engine and its training procedures, the project demonstrates the fundamental principles of supervised learning and neural network optimization.

---

**Developed using C# | Visual Studio Community | Feedforward Neural Network | Backpropagation**
