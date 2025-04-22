
# Steel Fatigue Strength Prediction with Jaya Algorithm Tuning

This repository focuses on predicting the fatigue strength of steel alloys under rotating bending conditions, based on a variety of chemical composition and heat treatment parameters. Accurate prediction of fatigue strength is crucial in industries where material failure due to cyclic loading can lead to catastrophic consequences, such as in aerospace, automotive, and structural applications.


## Dataset

The dataset used in this project is provided by Kaggle and contains various properties of steel alloys that influence their fatigue strength. You can access the dataset [here](https://www.kaggle.com/datasets/chaozhuang/steel-fatigue-strength-prediction/data).

### Data Details

The dataset consists of the following columns representing different properties of steel alloys:

| **Abbreviation** | **Property Details**                                         |
|------------------|--------------------------------------------------------------|
| **C**            | % Carbon                                                     |
| **Si**           | % Silicon                                                    |
| **Mn**           | % Manganese                                                  |
| **P**            | % Phosphorus                                                  |
| **S**            | % Sulphur                                                    |
| **Ni**           | % Nickel                                                     |
| **Cr**           | % Chromium                                                   |
| **Cu**           | % Copper                                                     |
| **Mo**           | % Molybdenum                                                 |
| **NT**           | Normalizing Temperature                                      |
| **THT**          | Through Hardening Temperature                                |
| **THt**          | Through Hardening Time                                       |
| **THQCr**        | Cooling Rate for Through Hardening                            |
| **CT**           | Carburization Temperature                                    |
| **Ct**           | Carburization Time                                           |
| **DT**           | Diffusion Temperature                                        |
| **Dt**           | Diffusion Time                                               |
| **QmT**          | Quenching Media Temperature (for Carburization)              |
| **TT**           | Tempering Temperature                                        |
| **Tt**           | Tempering Time                                               |
| **TCr**          | Cooling Rate for Tempering                                   |
| **RedRatio**     | Reduction Ratio (Ingot to Bar)                               |
| **dA**           | Area Proportion of Inclusions Deformed by Plastic Work       |
| **dB**           | Area Proportion of Inclusions Occurring in Discontinuous Array|
| **dC**           | Area Proportion of Isolated Inclusions                       |
| **Fatigue**      | Rotating Bending Fatigue Strength (10^7 Cycles)              |

The target variable, **Fatigue**, represents the rotating bending fatigue strength at \( 10^7 \) cycles.


## Project Overview

### Objective:
The objective is to predict the fatigue strength of steel alloys based on their properties using a neural network, with the goal of providing insights into the strength and durability of different alloys in industrial applications.

### Optimization with Jaya Algorithm:
The **Jaya Algorithm** is used to tune hyperparameters of the neural network. The Jaya Algorithm helps to find the optimal values for these hyperparameters by searching the solution space effectively.


### Hyperparameters Tuned:
- **Learning Rate**: Determines the size of the steps taken during training to minimize the loss function.
- **Epochs**: The number of times the entire training dataset is passed through the neural network.
- **Activation Function**: The function used to introduce non-linearity into the model, enabling it to learn complex patterns.
- **Hidden Layers**: The layers in the neural network that process inputs through weighted connections.
- **Neurons per Layer**: The number of neurons in each hidden layer.
- **Dropout Rate**: The fraction of the network's neurons that are randomly "dropped out" during training to prevent overfitting.
- **Optimizer**: The algorithm used to update the weights of the model based on the loss function.

The **final best hyperparameters** after optimization are:

- **Learning Rate**: 0.00093
- **Epochs**: 812
- **Activation Function**: ReLU
- **Hidden Layers**: 1
- **Neurons per Layer**: 126
- **Dropout Rate**: 0.2
- **Optimizer**: RMSprop

### Neural Network Architecture:
The model consists of:
- **Input Layer**: The input features (alloy properties).
- **Hidden Layers**: 1 hidden layer with 126 neurons.
- **Output Layer**: A single output representing the predicted fatigue strength.

### Jaya Algorithm:
The **Jaya Algorithm** is a simple optimization algorithm that adjusts the model’s hyperparameters to minimize the loss function in this project. It is simple yet effective for solving both constrained and unconstrained optimization problems. You can read more about the algorithm in this paper:  
[Jaya: A Simple and New Optimization Algorithm for Solving Constrained and Unconstrained Optimization Problems](https://www.researchgate.net/publication/282532308_Jaya_A_simple_and_new_optimization_algorithm_for_solving_constrained_and_unconstrained_optimization_problems)



### Requirements:
- **Python 3.x**
- **Jupyter Notebook or JupyterLab** (for running the `.ipynb` file)
- **PyTorch** (for building and training the neural network)
- **Numpy** (for numerical operations)
- **Pandas** (for data manipulation)
- **Matplotlib** (for data visualization)
- **scikit-learn** (for machine learning utilities)


## Results

After hyperparameter optimization using the Jaya algorithm, the model achieved the following best hyperparameters:

- **Learning Rate**: 0.00093
- **Epochs**: 812
- **Activation Function**: ReLU
- **Hidden Layers**: 1
- **Neurons per Layer**: 126
- **Dropout Rate**: 0.2
- **Optimizer**: RMSprop

These optimized hyperparameters resulted in improved performance in predicting the fatigue strength of steel alloys. You can find detailed results and model evaluation in the notebook.


## References

- Dataset: [Steel Fatigue Strength Prediction Dataset](https://www.kaggle.com/datasets/chaozhuang/steel-fatigue-strength-prediction/data)
- Venkata Rao, Ravipudi. (2016). Jaya: A simple and new optimization algorithm for solving constrained and unconstrained optimization problems. International Journal of Industrial Engineering Computations. 7. 19-34. 10.5267/j.ijiec.2015.8.004. [LINK](https://www.researchgate.net/publication/282532308_Jaya_A_simple_and_new_optimization_algorithm_for_solving_constrained_and_unconstrained_optimization_problems).

