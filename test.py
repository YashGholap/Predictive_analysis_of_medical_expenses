# import numpy as np
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn import metrics
# import matplotlib.pyplot as plt
# import seaborn as sns
#
#
# # Function to normalize the data
# def normalize(X, mean=None, std=None):
#     if mean is None:
#         mean = np.mean(X, axis=0)
#     if std is None:
#         std = np.std(X, axis=0)
#     return (X - mean) / std, mean, std
#
#
# # Gradient Descent function
# def gradient_descent(X, y, learning_rate, iterations):
#     m = len(y)
#     X = np.c_[np.ones(m), X]  # Add a column of ones to X for the intercept term
#     theta = np.zeros(X.shape[1])
#
#     for i in range(iterations):
#         prediction = np.dot(X, theta)
#         error = prediction - y
#         gradient = (1 / m) * np.dot(X.T, error)
#         theta = theta - learning_rate * gradient
#
#         if i % 100 == 0:  # Print the cost every 100 iterations
#             cost = (1 / (2 * m)) * np.dot(error.T, error)
#             print(f"Iteration {i}, Cost: {cost}")
#
#     return theta
#
#
# # Function to predict charges based on user input
# def predict_charges(age, sex, bmi, children, smoker, region, mean, std, theta):
#     # Create a dataframe with the input values
#     input_data = pd.DataFrame({
#         'age': [age],
#         'sex': [sex],
#         'bmi': [bmi],
#         'children': [children],
#         'smoker': [smoker],
#         'region': [region]
#     })
#
#     # Normalize the input data using the same mean and std as the training data
#     input_data = (input_data - mean) / std
#
#     # Add a column of ones to the input data for the intercept term
#     input_data = np.c_[np.ones(input_data.shape[0]), input_data]
#
#     # Predict using the trained model
#     predicted_charges = np.dot(input_data, theta)
#
#     return predicted_charges[0]
#
#
# # Load dataset
# insurance_dataset = pd.read_csv('insurance.csv')
#
# # Data preprocessing
# insurance_dataset.replace({'sex': {'male': 0, 'female': 1}}, inplace=True)
# insurance_dataset.replace({'smoker': {'yes': 0, 'no': 1}}, inplace=True)
# insurance_dataset.replace({'region': {'southeast': 0, 'southwest': 1, 'northeast': 2, 'northwest': 3}}, inplace=True)
#
# # Splitting the Features and Target
# X = insurance_dataset.drop(columns='charges', axis=1)
# Y = insurance_dataset['charges']
#
# # Normalize the features and store mean and std
# X_normalized, mean, std = normalize(X)
#
# # Splitting the data into Training data & Testing Data
# X_train, X_test, Y_train, Y_test = train_test_split(X_normalized, Y, test_size=0.2, random_state=2)
#
# # Run Gradient Descent
# learning_rate = 0.01
# iterations = 1000
# theta = gradient_descent(X_train, Y_train, learning_rate, iterations)
#
# # Predictions
# X_test = np.c_[np.ones(X_test.shape[0]), X_test]  # Add a column of ones to X_test for the intercept term
# Y_pred = np.dot(X_test, theta)
#
# # Evaluate the model
# r2_test = metrics.r2_score(Y_test, Y_pred)
# print('R squared value using Gradient Descent: ', r2_test)
#
# # Visualize the predicted vs actual values
# # plt.figure(figsize=(10, 6))
# # plt.scatter(Y_test, Y_pred, color='blue')
# # plt.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], 'k--', lw=2)
# # plt.xlabel('Actual')
# # plt.ylabel('Predicted')
# # plt.title('Actual vs Predicted Charges')
# # plt.show()
#
# # Example of predicting charges based on user input
# age = 23
# sex = 0  # 0 for male, 1 for female
# bmi = 24.8
# children = 0
# smoker = 1  # 0 for yes, 1 for no
# region = 2  # 0: southeast, 1: southwest, 2: northeast, 3: northwest
#
# predicted_expense = predict_charges(age, sex, bmi, children, smoker, region, mean, std, theta)
# print(f"Predicted Insurance Charges: {predicted_expense:.2f}")


str = 'Conclusion, Limitations & Future Scope/ Enhancement'

print(str.title())