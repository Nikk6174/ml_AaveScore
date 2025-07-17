# DeFi Credit Scoring Model

## Overview
This project implements a robust machine learning model to assign credit scores (0-1000) to DeFi wallet addresses based on their transaction behavior on the Aave V2 protocol. The model identifies reliable users while flagging potential risks and bot-like activities.

## Problem Statement
Given 100K raw transaction-level data from Aave V2 protocol, develop a machine learning model that:
- Assigns credit scores between 0-1000 to each wallet
- Higher scores indicate reliable and responsible usage
- Lower scores reflect risky, bot-like, or exploitative behavior
- Uses only historical transaction behavior for scoring

## Method Chosen

### Multi-Component Ensemble Approach
We implemented a **weighted ensemble model** combining five distinct behavioral assessment components:

1. **Anomaly Detection (25%)** - Identifies bot-like behavior using Isolation Forest
2. **Risk Assessment (30%)** - Evaluates liquidation history and borrowing patterns
3. **Stability Assessment (20%)** - Measures account maturity and diversification
4. **Activity Patterns (15%)** - Analyzes transaction consistency and timing
5. **Cluster Analysis (10%)** - Peer comparison and behavioral segmentation

### Why This Approach?
- **Robustness**: Multiple components prevent single-point failures
- **Interpretability**: Each component has clear business logic
- **Scalability**: Efficient processing of large datasets
- **Flexibility**: Easy to adjust weights based on business requirements

## Architecture

### 1. Data Processing Pipeline
```
Raw Transaction Data → Feature Engineering → Wallet-Level Aggregation → Credit Scoring
```

### 2. Feature Engineering
**Behavioral Pattern Features:**
- Transaction frequency and timing patterns
- Action distribution (deposit, borrow, repay ratios)
- Asset diversification metrics

**Risk Assessment Features:**
- Liquidation history and ratios
- Repayment behavior patterns
- Borrow-to-deposit ratios

**Stability Indicators:**
- Account age and tenure
- Asset portfolio diversification
- Transaction consistency metrics

### 3. Model Components

#### Component 1: Anomaly Detection (25% Weight)
- **Algorithm**: Isolation Forest
- **Purpose**: Detect outliers and bot-like behavior
- **Input**: All engineered features
- **Output**: Anomaly score (0-100)

#### Component 2: Risk Assessment (30% Weight)
- **Algorithm**: Rule-based scoring with penalties and bonuses
- **Components**:
  - Liquidation penalty (0-30 points)
  - Borrow-to-deposit risk (0-25 points)
  - Repayment behavior bonus (0-20 points)
  - High frequency penalty (0-15 points)
- **Output**: Risk score (0-100)

#### Component 3: Stability Assessment (20% Weight)
- **Algorithm**: Time-based and diversification scoring
- **Components**:
  - Account age score (0-25 points)
  - Asset diversification (0-15 points)
  - Consistency score (0-10 points)
- **Output**: Stability score (0-50)

#### Component 4: Activity Patterns (15% Weight)
- **Algorithm**: Pattern analysis and regularity scoring
- **Components**:
  - Activity regularity (0-20 points)
  - Time consistency (0-15 points)
- **Output**: Activity score (0-35)

#### Component 5: Cluster Analysis (10% Weight)
- **Algorithm**: K-means clustering (5 clusters)
- **Purpose**: Peer comparison and behavioral segmentation
- **Output**: Cluster-based score (0-100)

### 4. Final Score Calculation
```
Final Score = (Anomaly × 0.25 + Risk × 0.30 + Stability × 0.20 + Activity × 0.15 + Cluster × 0.10) × 10
```

## Processing Flow

### Step 1: Data Preprocessing
```python
# Load and clean transaction data
df = load_transaction_data()
df = clean_and_validate_data(df)
```

### Step 2: Feature Engineering
```python
# Aggregate transactions by wallet
wallet_groups = df.groupby('userWallet')

# Calculate behavioral features
- Transaction patterns (frequency, timing)
- Action distributions (deposit, borrow, repay ratios)
- Risk indicators (liquidations, utilization)
- Stability metrics (account age, diversification)
```

### Step 3: Credit Score Calculation
```python
# Process features for scoring
X, feature_cols = preprocess_features(wallet_features)

# Calculate component scores
anomaly_score = detect_anomalies(X) * 100
risk_score = calculate_risk_score(df, X)
stability_score = calculate_stability_score(df)
activity_score = calculate_activity_score(df)
clusters, cluster_scores = perform_user_clustering(X)

# Combine into final score
final_score = weighted_combination(components) * 10
```

### Step 4: Score Interpretation
```python
# Categorize scores into risk levels
score_ranges = {
    800-1000: "Excellent",
    600-799: "Good", 
    400-599: "Average",
    200-399: "Below Average",
    0-199: "Poor"
}
```

## File Structure
```
project/
├── README.md                 # This file
├── analysis.md              # Detailed analysis of scoring results
├── credit_scoring_model.py  # Main scoring pipeline
├── feature_engineering.py   # Feature extraction code
├── data_preprocessing.py    # Data cleaning utilities
└── requirements.txt         # Dependencies
```

## Dependencies
```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
matplotlib>=3.5.0
seaborn>=0.11.0
```

## Usage

### Basic Usage
```python
# Load your transaction data
df = pd.read_json('user_transactions.json')

# Run feature engineering
wallet_features = engineer_features(df)

# Calculate credit scores
credit_scores = calculate_credit_score(wallet_features)

# View results
print(credit_scores[['userWallet', 'credit_score', 'score_interpretation']])
```

### One-Step Script
```bash
python credit_scoring_model.py --input user_transactions.json --output wallet_scores.csv
```

## Model Performance

### Robustness Features
- **Outlier Handling**: Robust scaling and value clipping
- **Missing Value Treatment**: Median imputation for numerical features
- **Feature Normalization**: Standardized scaling across components
- **Boundary Validation**: All scores clipped to valid ranges (0-1000)

### Validation Approach
- **Logical Consistency**: Scores align with expected risk patterns
- **Edge Case Testing**: Handles extreme user behaviors appropriately
- **Distribution Analysis**: Scores follow expected statistical distributions
- **Component Validation**: Each component tested independently

## Score Interpretation

| Score Range | Classification | Typical Characteristics |
|-------------|---------------|------------------------|
| 800-1000 | Excellent | Long protocol usage, diversified portfolio, no liquidations, consistent repayments |
| 600-799 | Good | Stable usage patterns, minimal risk indicators, regular activity |
| 400-599 | Average | Typical retail behavior, some variability in patterns |
| 200-399 | Below Average | Some risk indicators, inconsistent patterns, moderate concerns |
| 0-199 | Poor | High risk, frequent liquidations, potential bot activity |

## Extensibility

### Adding New Features
```python
# Add new behavioral indicators
def calculate_new_feature(df):
    # Your feature logic here
    return feature_values

# Update feature engineering pipeline
new_features = calculate_new_feature(df)
wallet_features = wallet_features.join(new_features)
```

### Adjusting Component Weights
```python
# Modify weights based on business requirements
weights = {
    'anomaly': 0.20,    # Reduce anomaly weight
    'risk': 0.35,       # Increase risk weight
    'stability': 0.20,
    'activity': 0.15,
    'cluster': 0.10
}
```

## Limitations and Assumptions

### Assumptions
1. Past behavior predicts future behavior
2. Liquidations indicate higher risk
3. Consistent patterns indicate human behavior
4. Diversification indicates sophistication

### Limitations
1. No ground truth labels for validation
2. Limited to available transaction features
3. Doesn't account for market conditions
4. Cold start problem for new users

## Future Enhancements

### Potential Improvements
- **Real-time Scoring**: Streaming updates for live transactions
- **Market Context**: Incorporate market volatility factors
- **Cross-Protocol Analysis**: Combine data from multiple DeFi protocols
- **Temporal Modeling**: Account for changing user behavior over time
- **External Data**: Integrate on-chain reputation scores

### Model Updates
- **Incremental Learning**: Update model with new transaction data
- **A/B Testing**: Compare model versions in production
- **Feedback Loop**: Incorporate business user feedback
- **Hyperparameter Tuning**: Optimize component weights

## Contact and Support

For questions about the model implementation or to request features, please refer to the analysis.md file for detailed performance metrics and behavioral insights.

## License

This project is developed for educational and research purposes. Please ensure compliance with relevant regulations when using for commercial applications.