# Presentation Script — Parkinson's Disease Severity Prediction

---

## Slide 1 — Introduction & Dataset

"Hello everyone. Our project is about predicting the severity of Parkinson's disease using machine learning.

We used the Parkinson's Telemonitoring dataset from the UCI Machine Learning Repository. It contains biomedical voice recordings collected remotely from 42 patients with early-stage Parkinson's disease over a six-month period.

The dataset has 16 voice-based features — things like jitter, shimmer, and noise-to-harmonics ratio — all captured through a microphone at home, without the patient needing to visit a clinic."

---

## Slide 2 — Problem Definition & Goal

"The problem we're solving is a regression task. Our goal is to predict the motor UPDRS score — which stands for Unified Parkinson's Disease Rating Scale.

This score measures how severe a patient's motor symptoms are. A higher score means worse symptoms.

Why does this matter? Traditionally, doctors assess this score during in-person visits. If we can predict it accurately from voice measurements recorded at home, it opens the door to continuous, low-cost remote monitoring — which is especially important for patients who live far from hospitals or have limited mobility."

---

## Slide 3 — Model Selection

"We selected three models, each representing a different level of complexity.

First, **Linear Regression** — this is our baseline. It's simple, fast, and interpretable, but it assumes a straight-line relationship between features and the target, which may not capture the full complexity.

Second, **Random Forest** — an ensemble of many decision trees. It handles non-linear relationships well and is resistant to overfitting.

Third, **Gradient Boosting** — also tree-based, but it builds trees sequentially, each one correcting the errors of the previous. It generally achieves the highest accuracy on tabular data like ours.

We chose this combination to compare a simple baseline against two powerful tree-based methods."

---

## Slide 4 — Training the Models

"We split the data 80% for training and 20% for testing, using a fixed random seed so our results are reproducible.

For Linear Regression, we applied standard scaling first, since it's sensitive to feature magnitude.

We evaluated each model using two metrics: **RMSE** — which tells us on average how far off our predictions are in the same unit as the target — and **R²** — which tells us how much of the variance in the target our model explains. An R² of 1.0 is perfect."

---

## Slide 5 — Parameter Selection

"After training the baseline models, we applied hyperparameter tuning to the best-performing one — Random Forest.

We used **GridSearchCV** with 5-fold cross-validation, which means the training data is split into 5 parts and each combination of parameters is tested 5 times. This gives a reliable estimate of how well each configuration generalizes.

The parameters we tuned were the number of trees, the maximum depth of each tree, and the minimum number of samples needed to split a node.

The reason we tune these is to prevent the model from either overfitting — memorizing the training data — or underfitting — being too simple to learn the patterns."

---

## Slide 6 — Visualizations & Results

"Now let's look at the results through five visualizations.

**Visualization 1 — Correlation Heatmap:** This shows us which features are correlated with each other. Highly correlated features carry redundant information, which helps explain why some features end up being more important than others.

**Visualization 2 — Target Distribution:** The motor UPDRS scores range roughly from 5 to 40, with most patients clustering in the middle range. This gives us a sense of the spread we need our model to handle.

**Visualization 3 — Actual vs. Predicted:** The closer the points are to the dashed diagonal line, the better the model. You can clearly see that the tuned Random Forest performs best — its points are tightest around the perfect-prediction line.

**Visualization 4 — Feature Importance:** This shows which voice features matter most. The top features include PPE, DFA, and shimmer-related measures — these are acoustic properties that change as vocal cord control degrades with Parkinson's.

**Visualization 5 — Model Comparison:** The bar chart summarizes all four model variants. The tuned Random Forest achieves the lowest RMSE and the highest R², confirming that hyperparameter tuning made a meaningful improvement."

---

## Closing

"To summarize — we showed that Parkinson's motor severity can be predicted with meaningful accuracy from simple voice recordings, using a Random Forest model optimized with grid search.

This approach could support a low-cost, scalable remote monitoring system for Parkinson's patients.

Thank you — happy to take any questions."

---
