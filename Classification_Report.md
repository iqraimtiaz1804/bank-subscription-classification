# Bank Customer Subscription Prediction Using Three Classification Models

**Course:** AI Web Development | **Assignment:** TASK 06 | **Student:** Iqra Imtiaz

## 1. Objective

To build and compare three classification models that predict whether a bank customer will subscribe to a term deposit (`yes`) or not (`no`), and to identify the best-performing model using accuracy, precision, recall and confusion matrices.

## 2. Dataset Description

The project uses the **Bank Marketing** dataset from the UCI Machine Learning Repository (https://archive.ics.uci.edu/dataset/222/bank+marketing), file `bank.csv` (semicolon-separated). It contains historical marketing campaign records from a Portuguese bank: 4,521 records and 17 columns.

- **Target:** `y` (`yes` = subscribed, `no` = did not subscribe). Class split: 88.48% `no` (4,000) and 11.52% `yes` (521), so the data is imbalanced.
- **Features used (15):** age, job, marital, education, default, balance, housing, loan, contact, day, month, campaign, pdays, previous, poutcome.
- **Excluded:** `duration` (call length is only known after the call, so using it would cause information leakage) and the target `y`.

## 3. Preprocessing

- **Missing values:** No genuine missing values were found. The category `unknown` (found in job, education, contact and poutcome) was kept as a valid category instead of being treated as missing.
- **Duplicates:** No duplicate rows were found, so none were removed.
- **Encoding:** categorical columns were converted to numbers using `OneHotEncoder`.
- **Scaling:** numeric columns were scaled with `StandardScaler` (important for KNN and helpful for Logistic Regression).
- **Split:** 80% training (3,616 rows) / 20% testing (905 rows) using `train_test_split` with `random_state=42` and `stratify=y`. The split was done before preprocessing, and a scikit-learn `Pipeline` fitted the preprocessing on training data only to avoid data leakage. All three models used the same split.

## 4. Models Used

- **Logistic Regression:** calculates the probability of `yes` from a weighted combination of features (`max_iter=1000`).
- **Decision Tree:** learns a series of if/else rules from the data (`max_depth=5`).
- **K-Nearest Neighbors:** predicts by majority vote of the 5 most similar training customers (`n_neighbors=5`).

## 5. Evaluation Metrics

- **Accuracy:** share of all predictions that are correct.
- **Precision (for `yes`):** of customers predicted `yes`, the share who really subscribed.
- **Recall (for `yes`):** of customers who really subscribed, the share the model found.
- **Confusion matrix:** counts of True Negatives (TN), False Positives (FP), False Negatives (FN) and True Positives (TP).

## 6. Performance Comparison

| Model | Accuracy | Precision | Recall |
|---|---|---|---|
| Logistic Regression | 0.8873 | 0.5333 | 0.1538 |
| Decision Tree | 0.8729 | 0.3429 | 0.1154 |
| KNN | 0.8762 | 0.2778 | 0.0481 |

The test set has 905 customers, of whom 104 actually subscribed. A model that always answered `no` would be correct for 801 of 905 customers (accuracy 0.8851).

## 7. Confusion Matrix Interpretation

(Images are in the `confusion_matrices/` folder. Positive class = `yes`.)

- **Logistic Regression:** TN = 787, FP = 14, FN = 88, TP = 16. It predicted `yes` for 30 customers and 16 of them truly subscribed, but it missed 88 of the 104 real subscribers.
- **Decision Tree:** TN = 778, FP = 23, FN = 92, TP = 12. It found 12 real subscribers, made more false alarms than Logistic Regression and missed 92.
- **KNN:** TN = 788, FP = 13, FN = 99, TP = 5. It rarely predicted `yes` (18 times) and found only 5 real subscribers.

## 8. Best Model

**Selection rule:** because most customers are `no`, accuracy alone can be misleading. The models were ranked by recall for `yes` first (finding likely subscribers), then precision, then accuracy.

**Selected model:** Logistic Regression

**Reason:** Logistic Regression has the highest recall (0.1538), the highest precision (0.5333) and the highest accuracy (0.8873), so it ranks first on every metric and there is no trade-off to resolve. The Decision Tree is second (recall 0.1154) and KNN is last (recall 0.0481, precision 0.2778). However, all three models have low recall: even the best one finds only about 15% of real subscribers, and its accuracy is only slightly above the 0.8851 of an "always no" guess. The choice of Logistic Regression is therefore "best of three", not "good enough for real use".

## 9. Conclusion

This project compared Logistic Regression, a Decision Tree and KNN on the UCI Bank Marketing dataset (`bank.csv`, 4,521 records) to predict term deposit subscription. After checking for missing values and duplicates, encoding categorical features, scaling numeric features and splitting the data 80/20 with stratification, all models were evaluated on the same test set. Logistic Regression performed best (accuracy 0.8873, precision 0.5333, recall 0.1538). All models struggled to find the minority `yes` class because only 11.5% of customers subscribed and `duration` was excluded to avoid leakage. No hyperparameter tuning or class balancing was applied, as required by the assignment scope; these would be natural next steps to improve recall.
