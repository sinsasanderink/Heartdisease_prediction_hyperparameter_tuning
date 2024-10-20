# Heart Disease Prediction Using Decision Tree Classifiers

![Heart Disease Prediction](heartdisease.png)

## Overview
This project aims to predict the presence of heart disease in individuals using decision tree classifiers. The model is trained on data containing features such as age, sex, blood pressure, and cholesterol levels, and it explores different hyperparameters to improve accuracy and avoid overfitting.

## Dataset
The dataset used in this project (`heart_v2.csv`) includes the following columns:
- `age`: Age of the individual
- `sex`: Gender (0 = Female, 1 = Male)
- `BP`: Blood Pressure level
- `cholestrol`: Cholesterol level
- `heart disease`: Target variable (0 = No disease, 1 = Disease)

## Steps Followed in the Project

### 1. Data Preparation
- Import necessary libraries for data manipulation, visualization, and model building.
- Read the dataset and load it into a pandas DataFrame.
- Define feature variables (`X`) and the target variable (`y`).
- Split the dataset into training (70%) and testing (30%) sets to evaluate the model.

### 2. Building the Decision Tree Model
#### Fitting Default Decision Tree
- Initially, we fit a decision tree classifier with the default parameters except for setting `max_depth=3`. This helps in visualizing the tree structure and understanding how splits are made.
- Visualize the decision tree using `export_graphviz` and `pydotplus` to plot it.

#### Model Evaluation
- Evaluate the model performance on both the training and testing datasets using accuracy scores and confusion matrices.
- Identify potential overfitting or underfitting based on performance metrics.

### 3. Hyperparameter Tuning
The project explores various hyperparameters to optimize model performance:
1. **`max_depth`**: Control the maximum depth of the tree to prevent overfitting.
2. **`min_samples_split`**: Specify the minimum number of samples required to split an internal node, avoiding splits on small data points.
3. **`min_samples_leaf`**: Specify the minimum number of samples required to be at a leaf node, which smoothens the model by avoiding over-complexity.
4. **`criterion`**: Explore both "gini" impurity and "entropy" (Information Gain) as splitting criteria.

### 4. Model Tuning with Hyperparameters
- Create helper functions to generate decision tree graphs and evaluate models.
- Fit and visualize models with specific hyperparameters:
  - **Controlling Tree Depth (`max_depth`)**: Setting different values to prevent deep, complex trees.
  - **Minimum Samples for Split (`min_samples_split`)**: Specifying the minimum data points needed to make a split.
  - **Minimum Samples in Leaf Nodes (`min_samples_leaf`)**: Ensuring each leaf contains a sufficient number of samples to generalize well.
  - **Changing Splitting Criterion (`criterion`)**: Compare performance using "gini" vs. "entropy".

### 5. Hyperparameter Tuning with `GridSearchCV`
- Perform hyperparameter tuning using `GridSearchCV` to identify the best combination of parameters for the model.
- Define a grid of hyperparameters:
  ```python
  params = {
      'max_depth': [2, 3, 5, 10, 20],
      'min_samples_leaf': [5, 10, 20, 50, 100],
      'criterion': ["gini", "entropy"]
  }
