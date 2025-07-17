# AaveScore

## Overview

This project implements a robust machine learning model to assign credit scores (0–1000) to DeFi wallet addresses based on their transaction behavior on the Aave V2 protocol. The model identifies reliable users while flagging potential risks and bot-like activities.

## Problem Statement

Given 100K raw transaction-level data from the Aave V2 protocol, develop a machine learning model that:

* Assigns credit scores between 0–1000 to each wallet
* Higher scores indicate reliable and responsible usage
* Lower scores reflect risky, bot-like, or exploitative behavior
* Uses only historical transaction behavior for scoring

## Method Chosen

### Multi-Component Ensemble Approach

The project employs a weighted ensemble model that combines five distinct behavioral assessment components to derive a comprehensive credit score:

1. **Anomaly Detection (25% weight):** Uses Isolation Forest to identify outliers and bot-like behavior based on all engineered features, outputting an anomaly score (0–100).
2. **Risk Assessment (30% weight):** Applies rule-based scoring with penalties (e.g., liquidation history, high-frequency transactions) and bonuses (e.g., repayment behavior), yielding a risk score (0–100).
3. **Stability Assessment (20% weight):** Evaluates account maturity and diversification using time-based and asset-based metrics, producing a stability score (0–50).
4. **Activity Patterns (15% weight):** Analyzes transaction consistency and timing patterns, generating an activity score (0–35).
5. **Cluster Analysis (10% weight):** Implements K-means clustering (5 clusters) for peer comparison and behavioral segmentation, contributing a cluster-based score (0–100).

### Why This Approach?

* **Robustness:** Multiple components mitigate single-point failures.
* **Interpretability:** Each component has clear business logic tied to wallet behavior.
* **Scalability:** Efficiently processes large datasets (e.g., 100K transactions).
* **Flexibility:** Weights can be adjusted based on business priorities.

## Architecture

### 1. Complete Architecture

The model architecture is structured as a data processing and scoring pipeline:

1. **Data Input Layer:** Loads raw transaction data (JSON, \~87 MB) from the Aave V2 protocol.
2. **Preprocessing Layer:** Cleans data, handles missing values (median imputation), and replaces infinite values.
3. **Feature Engineering Layer:** Aggregates transactions by wallet and extracts behavioral features (e.g., transaction frequency, liquidation history) and stability indicators (e.g., account age).
4. **Modeling Layer:**

   * Anomaly Detection: Isolation Forest identifies outliers.
   * Risk Assessment: Rule-based scoring with weighted penalties and bonuses.
   * Stability Assessment: Time-based and diversification scoring.
   * Activity Patterns: Pattern analysis for consistency.
   * Cluster Analysis: K-means clustering for segmentation.
5. **Scoring Layer:** Combines component scores with weighted aggregation and scales to 0–1000.
6. **Output Layer:** Exports results to Parquet files and generates visualizations.

### 2. Feature Engineering

**Behavioral Pattern Features:**

* Transaction frequency and timing patterns
* Action distribution (deposit, borrow, repay ratios)
* Asset diversification metrics

**Risk Assessment Features:**

* Liquidation history and ratios
* Repayment behavior patterns
* Borrow-to-deposit ratios

**Stability Indicators:**

* Account age and tenure
* Asset portfolio diversification
* Transaction consistency metrics

### 3. Model Components

**Component 1: Anomaly Detection (25% Weight)**

* Algorithm: Isolation Forest
* Purpose: Detect outliers and bot-like behavior
* Input: All engineered features
* Output: Anomaly score (0–100)

**Component 2: Risk Assessment (30% Weight)**

* Algorithm: Rule-based scoring with penalties and bonuses
* Components:

  * Liquidation penalty (0–30 points)
  * Borrow-to-deposit risk (0–25 points)
  * Repayment behavior bonus (0–20 points)
  * High frequency penalty (0–15 points)
* Output: Risk score (0–100)

**Component 3: Stability Assessment (20% Weight)**

* Algorithm: Time-based and diversification scoring
* Components:

  * Account age score (0–25 points)
  * Asset diversification (0–15 points)
  * Consistency score (0–10 points)
* Output: Stability score (0–50)

**Component 4: Activity Patterns (15% Weight)**

* Algorithm: Pattern analysis and regularity scoring
* Components:

  * Activity regularity (0–20 points)
  * Time consistency (0–15 points)
* Output: Activity score (0–35)

**Component 5: Cluster Analysis (10% Weight)**

* Algorithm: K-means clustering (5 clusters)
* Purpose: Peer comparison and behavioral segmentation
* Output: Cluster-based score (0–100)

### 4. Final Score Calculation

```
Final Score = (Anomaly × 0.25 + Risk × 0.30 + Stability × 0.20 + Activity × 0.15 + Cluster × 0.10) × 10
```

## Processing Flow

### Step-by-Step Workflow

1. **Data Preprocessing:**

   ```python
   df = load_transaction_data()
   df = clean_and_validate_data(df)
   ```
2. **Feature Engineering:**

   ```python
   wallet_features = engineer_features(df)
   ```
3. **Credit Score Calculation:**

   ```python
   credit_scores = calculate_credit_score(wallet_features)
   ```
4. **Score Interpretation and Export:**

   ```bash
   model_export_parquetFile/credit__scores_1.parquet
   model_export_parquetFile/wallet_features_1.parquet
   ```
5. **Analysis and Visualization:**

   * Load Parquet files in `visualizations.ipynb`.
   * Generate statistics and plots (histograms, bar charts, heatmaps).

## File Structure

```
AaveScore/
├── .gitignore
├── model.ipynb
├── visualizations.ipynb
├── analysis.md
├── readme.md
├── parquetFile/
└── *.png
```

## Dependencies

* pandas>=1.3.0
* numpy>=1.21.0
* scikit-learn>=1.0.0
* matplotlib>=3.5.0
* seaborn>=0.11.0
* pyarrow>=14.0.0

## Usage

**Basic Usage**

```python
# Load your transaction data
df = pd.read_json('user_transactions.json')

# Engineer features
wallet_features = engineer_features(df)

# Calculate credit scores
credit_scores = calculate_credit_score(wallet_features)

# View results
print(credit_scores[['userWallet', 'credit_score', 'score_interpretation']])
```

**One-Step Script**

```bash
jupyter nbconvert --to notebook --execute model.ipynb
jupyter nbconvert --to notebook --execute visualizations.ipynb
```

## Model Performance

**Robustness Features:**

* Outlier Handling: Robust scaling and value clipping
* Missing Value Treatment: Median imputation for numerical features
* Feature Normalization: Standardized scaling across components
* Boundary Validation: All scores clipped to valid ranges (0–1000)

**Validation Approach:**

* Logical consistency checks
* Edge case testing for extreme behaviors
* Distribution analysis to confirm expected score ranges

## Score Interpretation

| Score Range | Classification | Typical Characteristics                                     |
| ----------- | -------------- | ----------------------------------------------------------- |
| 800–1000    | Excellent      | Long protocol usage, diversified portfolio, no liquidations |
| 600–799     | Good           | Stable usage patterns, minimal risk indicators              |
| 400–599     | Average        | Typical retail behavior, some variability                   |
| 200–399     | Below Average  | Some risk indicators, inconsistent patterns                 |
| 0–199       | Poor           | High risk, frequent liquidations, potential bot activity    |

## Extensibility

**Adding New Features**

```python
def calculate_new_feature(df):
    # Your feature logic here
    return feature_values
```

**Adjusting Component Weights**

```python
weights = {
    'anomaly': 0.20,
    'risk': 0.35,
    'stability': 0.20,
    'activity': 0.15,
    'cluster': 0.10
}
```

## Limitations and Assumptions

**Assumptions**

* Past behavior predicts future behavior
* Liquidations indicate higher risk
* Consistent patterns indicate human behavior
* Diversification indicates sophistication

**Limitations**

* No ground truth labels for validation
* Limited to available transaction features
* Doesn’t account for external market conditions
* Cold start problem for new users

## Future Enhancements

* Real-time scoring with streaming data
* Incorporate market context and volatility
* Cross-protocol analysis for holistic risk
* Temporal modeling of user behavior
* Integrate on-chain reputation scores

## Contact and Support

For questions about the model implementation or to request features, please refer to the **analysis.md** file for detailed metrics and insights.

## License

This project is developed for educational and research purposes. Please ensure compliance with relevant regulations when using for commercial applications.
