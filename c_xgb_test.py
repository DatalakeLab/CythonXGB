import c_xgb_test

print ("XGBoost Linear Regression")

c_xgb_test.test_xgb_regression(n_samples = 10000, n_features = 20, n_estimators = 100, depth = 10)
print ("XGBoost Logistic Regression")
c_xgb_test.test_xgb_logistic_regression(n_samples = 10000, n_features = 20, n_estimators = 100, depth = 10)
