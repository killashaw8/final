import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(page_title="Parkinson's Severity Prediction", layout="wide")
st.title("Parkinson's Disease Severity Prediction")
st.caption(
    "Dataset: UCI Parkinson's Telemonitoring — 42 patients, 5,875 voice recordings")


@st.cache_resource
def load_and_train():
    df = pd.read_json('parkinsons_updrs.json')
    target_cols = ['motor_UPDRS', 'total_UPDRS']
    feature_cols = [
        c for c in df.columns if c not in target_cols + ['subject#']]
    X = df[feature_cols]
    y = df['motor_UPDRS']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    lr = LinearRegression()
    lr.fit(scaler.fit_transform(X_train), y_train)
    y_pred_lr = lr.predict(scaler.transform(X_test))

    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)

    gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
    gb.fit(X_train, y_train)
    y_pred_gb = gb.predict(X_test)

    best_rf = RandomForestRegressor(n_estimators=200, max_depth=None,
                                    min_samples_split=2, random_state=42, n_jobs=-1)
    best_rf.fit(X_train, y_train)
    y_pred_best = best_rf.predict(X_test)

    return df, X, y_test, y_pred_lr, y_pred_rf, y_pred_gb, y_pred_best, best_rf


with st.spinner("Training models..."):
    df, X, y_test, y_pred_lr, y_pred_rf, y_pred_gb, y_pred_best, best_rf = load_and_train()

st.divider()

# ── Slide 1: Import Dataset ───────────────────────────────────────────────────
st.subheader("1: Import Dataset")
st.markdown("""
The dataset comes from the **UCI ML Repository (ID: 189)** — Parkinson's Telemonitoring.
It contains **5,875 biomedical voice recordings** from **42 patients** with early-stage Parkinson's disease,
collected remotely over several months.

Each row is one recording session. Features are acoustic measurements extracted from sustained
phonations. Targets are two clinician-rated UPDRS scores: `motor_UPDRS` and `total_UPDRS`.
We focus on predicting `motor_UPDRS`.
""")
st.code("""import pandas as pd

df = pd.read_json('parkinsons_updrs.json')

target_cols  = ['motor_UPDRS', 'total_UPDRS']
feature_cols = [c for c in df.columns if c not in target_cols + ['subject#']]

X = df[feature_cols]
y = df[target_cols]

print(f'Features : {X.shape}')
print(f'Targets  : {y.shape}')
print(f'Columns  : {feature_cols}')""", language='python')

st.divider()

# ── Slide 2: Problem Definition ───────────────────────────────────────────────
st.subheader("2: Problem Definition & Goal")
st.markdown("""
**Problem type:** Supervised Regression

**Goal:** Predict the `motor_UPDRS` score from 16 biomedical voice measurements recorded remotely.
A lower score indicates better motor function; a higher score indicates greater symptom severity.

**Why this matters:** Accurate remote prediction of UPDRS enables clinicians to monitor disease
progression without in-person visits — improving accessibility and enabling timely intervention.

**Features (16 voice metrics):** Jitter, Shimmer, NHR, HNR, RPDE, DFA, PPE, and others.
**Target:** `motor_UPDRS` (continuous, range 0–108)
""")
st.code("""import numpy as np

y_target = y['motor_UPDRS']

print("=== Dataset Overview ===")
print(f"Features shape : {X.shape}")
print(f"Target shape   : {y_target.shape}")
print(f"\\nFeatures       : {X.columns.tolist()}")
print(f"\\nMissing values : {X.isnull().sum().sum()}")
print("\\n=== Target Statistics ===")
print(y_target.describe())
print("\\n=== First 5 rows ===")
display(X.head())""", language='python')

st.divider()

# ── Slide 3: Model Selection ──────────────────────────────────────────────────
st.subheader("3: Model Selection")
st.markdown("""
Three models were selected, ranging from simple to complex:

| Model | Reason |
|---|---|
| **Linear Regression** | Baseline — fast, interpretable, assumes linear relationships |
| **Random Forest Regressor** | Ensemble of trees — handles non-linearity, robust to outliers |
| **Gradient Boosting Regressor** | Sequential boosting — typically highest accuracy on tabular data |

Starting with a linear baseline lets us quantify how much non-linearity actually matters in this dataset.
""")

st.divider()

# ── Slide 4: Training the Models ─────────────────────────────────────────────
st.subheader("4: Training the Models")
st.markdown("""
All models were trained on an **80/20 train/test split** (random state = 42 for reproducibility).

- **Linear Regression** used standardized features (StandardScaler) since it is sensitive to scale.
- **Random Forest** and **Gradient Boosting** are scale-invariant and trained on raw features.
- Evaluation metrics: **RMSE** (average error in UPDRS units) and **R²** (variance explained).
""")
st.code("""from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

X_train, X_test, y_train, y_test = train_test_split(
    X, y_target, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

lr = LinearRegression()
lr.fit(X_train_sc, y_train)
y_pred_lr = lr.predict(X_test_sc)

rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
gb.fit(X_train, y_train)
y_pred_gb = gb.predict(X_test)""", language='python')

st.divider()

# ── Slide 5: Hyperparameter Tuning ───────────────────────────────────────────
st.subheader("5: Hyperparameter Tuning")
st.markdown("""
**GridSearchCV** with 5-fold cross-validation was applied to the Random Forest to find the
best combination of:

- `n_estimators` — number of trees (50, 100, 200)
- `max_depth` — maximum tree depth (None, 5, 10)
- `min_samples_split` — minimum samples to split a node (2, 5, 10)

**Best parameters found:** `n_estimators=200`, `max_depth=None`, `min_samples_split=2`

The tuned model's RMSE improved only marginally over the default — suggesting the default
Random Forest was already well-suited to this data.
""")
st.code("""from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators':      [50, 100, 200],
    'max_depth':         [None, 5, 10],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(
    RandomForestRegressor(random_state=42),
    param_grid,
    cv=5,
    scoring='neg_root_mean_squared_error',
    n_jobs=-1,
    verbose=1
)
grid_search.fit(X_train, y_train)

best_rf     = grid_search.best_estimator_
y_pred_best = best_rf.predict(X_test)

print(f"Best parameters : {grid_search.best_params_}")
print(f"Tuned RF  RMSE  : {np.sqrt(mean_squared_error(y_test, y_pred_best)):.4f}")
print(f"Tuned RF  R²    : {r2_score(y_test, y_pred_best):.4f}")""", language='python')

st.divider()
st.subheader("Visualizations")
st.markdown(
    "The following charts explore the data, model predictions, and results in detail.")
st.divider()

# ── Viz 1: Correlation Heatmap ────────────────────────────────────────────────
st.subheader("1. Feature Correlation Heatmap")
st.markdown("""
Before modeling, we need to understand how features relate to each other.
This heatmap shows pairwise Pearson correlations across all 19 input features.
Watch for clusters of high correlation — they signal redundant information.
""")
fig1 = px.imshow(X.corr().round(2), text_auto=True, color_continuous_scale='RdBu_r',
                 zmin=-1, zmax=1, aspect='auto')
fig1.update_layout(height=600)
st.plotly_chart(fig1, use_container_width=True)
st.info("""
**Key takeaway:** The Jitter and Shimmer families are highly correlated (~0.9+), meaning they carry
overlapping information. Tree-based models handle this naturally, but future work could reduce
these groups to principal components.
""")

# ── Viz 2: Target Distribution ────────────────────────────────────────────────
st.subheader("2. Distribution of motor_UPDRS")
st.markdown("""
Before choosing a model, we check whether the target variable is well-behaved.
A heavily skewed or multimodal distribution would require transformation or special handling.
Here we look at the spread of motor UPDRS scores across all 5,875 recordings.
""")
fig2 = px.histogram(x=df['motor_UPDRS'], nbins=50, marginal='box',
                    labels={'x': 'motor_UPDRS Score'},
                    color_discrete_sequence=['steelblue'])
fig2.update_layout(xaxis_title='motor_UPDRS Score',
                   yaxis_title='Count', height=400)
st.plotly_chart(fig2, use_container_width=True)
st.info("""
**Key takeaway:** The target is roughly bell-shaped (mean ~21, max ~40) with no extreme outliers.
The data is well-behaved for regression — no log transformation needed.
""")

# ── Viz 3: Actual vs. Predicted ───────────────────────────────────────────────
st.subheader("3. Actual vs. Predicted motor_UPDRS")
st.markdown("""
The clearest way to evaluate a regression model is to plot what it predicted against what actually happened.
Points along the dashed diagonal = perfect prediction. Scatter away from it = error.
We compare all three models side by side to see which one fits tightest.
""")
perfect_line = [y_test.min(), y_test.max()]
fig3 = go.Figure()
fig3.add_trace(go.Scatter(x=y_test, y=y_pred_lr, mode='markers', name='Linear Regression',
                          marker=dict(opacity=0.5, size=4)))
fig3.add_trace(go.Scatter(x=y_test, y=y_pred_rf, mode='markers', name='Random Forest',
                          marker=dict(opacity=0.5, size=4)))
fig3.add_trace(go.Scatter(x=y_test, y=y_pred_best, mode='markers', name='RF Tuned',
                          marker=dict(opacity=0.5, size=4)))
fig3.add_trace(go.Scatter(x=perfect_line, y=perfect_line, mode='lines',
                          name='Perfect Prediction',
                          line=dict(dash='dash', color='black', width=2)))
fig3.update_layout(xaxis_title='Actual', yaxis_title='Predicted', height=500)
st.plotly_chart(fig3, use_container_width=True)
st.info("""
**Key takeaway:** Linear Regression scatters widely — voice features have non-linear relationships
with UPDRS. Random Forest clusters tightly around the perfect-prediction line, confirming
tree-based models are the right choice here.
""")

# ── Viz 4: Feature Importance ─────────────────────────────────────────────────
st.subheader("4. Feature Importance — Tuned Random Forest")
st.markdown("""
Random Forest lets us ask: which features actually drove the predictions?
Importance is measured by how much each feature reduces impurity across all trees.
This tells us what the model learned to rely on — and what turned out not to matter.
""")
feat_imp = (pd.DataFrame({'Feature': X.columns, 'Importance': best_rf.feature_importances_})
            .sort_values('Importance', ascending=True))
fig4 = px.bar(feat_imp, x='Importance', y='Feature', orientation='h',
              color='Importance', color_continuous_scale='Plasma')
fig4.update_layout(height=500, yaxis_title='', coloraxis_showscale=False)
st.plotly_chart(fig4, use_container_width=True)
st.info("""
**Key takeaway:** `age` is the dominant predictor by a wide margin, followed by `PPE` and `DFA`
(nonlinear dynamical complexity measures). Standard Jitter/Shimmer metrics contribute less than
expected — disease progression is better captured by age and signal complexity.
""")

# ── Viz 5: Model Comparison ───────────────────────────────────────────────────
st.subheader("5. Model Performance Comparison")
st.markdown("""
Finally, we summarize all four models on two metrics: RMSE (average prediction error in UPDRS units)
and R² (proportion of variance explained). This is the bottom line — which model would you
actually trust to monitor a patient remotely?
""")
results = pd.DataFrame({
    'Model': ['Linear Regression', 'Random Forest', 'Gradient Boosting', 'RF Tuned'],
    'RMSE':  [np.sqrt(mean_squared_error(y_test, p)) for p in [y_pred_lr, y_pred_rf, y_pred_gb, y_pred_best]],
    'R²':    [r2_score(y_test, p) for p in [y_pred_lr, y_pred_rf, y_pred_gb, y_pred_best]]
})
fig5 = make_subplots(rows=1, cols=2,
                     subplot_titles=['RMSE (lower is better)', 'R² (higher is better)'])
fig5.add_trace(go.Bar(x=results['Model'], y=results['RMSE'],
                      marker_color='tomato', name='RMSE'), row=1, col=1)
fig5.add_trace(go.Bar(x=results['Model'], y=results['R²'],
                      marker_color='mediumseagreen', name='R²'), row=1, col=2)
fig5.update_layout(showlegend=False, height=420)
fig5.update_xaxes(tickangle=15)
st.plotly_chart(fig5, use_container_width=True)
st.info("""
**Key takeaway:** Random Forest achieves R² ≈ 0.97 — explaining 97% of variance in motor UPDRS.
Linear Regression barely reaches R² ≈ 0.12, highlighting how non-linear this problem is.
Hyperparameter tuning gave marginal gains since the default RF was already near-optimal.
""")
