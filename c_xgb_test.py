import c_xgb_test

print "XGBoost Linear Regression")
c_xgb_test.test_xgb_regression(n_samples = 100000, n_features = 35, n_estimators = 3, depth = 15)

print "XGBoost Logistic Regression")
c_xgb_test.test_xgb_logistic_regression(n_samples = 100000, n_features = 35, n_estimators = 3, depth = 15)
