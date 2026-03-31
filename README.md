# Medical Insurance Cost Prediction

## Project Overview
This project focuses on predicting medical insurance costs using various machine learning models. The objective is to provide insights into the factors that affect insurance costs and to implement predictive modeling techniques to achieve accurate predictions.

## Features
- **Data Preprocessing**: Clean and prepare the data for modeling.
- **Model Training**: Implementation of various models such as Linear Regression, Decision Trees, and Random Forest.
- **Performance Evaluation**: Metrics to evaluate model performance, including RMSE, Mean Absolute Error, and R-squared.
- **User Interface**: A simple user interface to input parameters and get predictions.

## File Structure
```
Medical_Insurance_Cost_Prediction/
├── data/
│   └── data.csv            # Dataset file
├── models/
│   ├── model1.py          # First model implementation
│   ├── model2.py          # Second model implementation
│   └── model3.py          # Model implementation comparisons
├── notebooks/
│   └── exploratory_analysis.ipynb  # Jupyter notebook for EDA
├── requirements.txt       # Python package dependencies
└── README.md              # Project documentation
```

## Installation Instructions
1. Clone the repository:
   ```bash
   git clone https://github.com/MasudMallik/Medical_Insurance_Cost_Prediction.git
   cd Medical_Insurance_Cost_Prediction
   ```
2. Install the required packages using pip:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Run the main application:
   ```bash
   python main.py
   ```
2. Follow the prompts to enter the required information for prediction.

## Model Comparisons
- **Linear Regression**: Expected to perform well with a linear relationship.
- **Decision Tree**: Useful for capturing non-linear patterns.
- **Random Forest**: Aggregates multiple decision trees to improve accuracy and reduce overfitting.

This project aims to provide a clear understanding of the different factors affecting medical insurance costs, leverage machine learning techniques, and present a comprehensive analysis of model performances.