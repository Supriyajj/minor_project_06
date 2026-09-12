# 🛒 QuickCart Warehouse Inventory — Stockout Risk Classification

## 📌 Project Overview

**QuickCart Warehouse Inventory — Stockout Risk Classification** is a machine learning project designed to predict the daily stockout risk for every **Store × SKU × Day** combination.

The model classifies inventory into three risk categories:

* 🟢 **Safe** — inventory is sufficiently available
* 🟡 **At-Risk** — inventory requires attention
* 🔴 **Imminent** — inventory is likely to stock out before the next supplier delivery

The project combines inventory, store, SKU, supplier, and event-level data to identify the operational factors that contribute to stockout risk.

The main objective is not only to maximize overall accuracy, but particularly to **identify Imminent stockout cases**, where missing a true stockout can have a significant business impact.

---

## 🎯 Business Problem

Inventory stockouts can lead to:

* Lost sales
* Customer dissatisfaction
* Emergency replenishment costs
* Poor warehouse planning
* Reduced service levels
* Missed demand during high-volume periods

A warehouse needs an early-warning system that can identify products approaching stockout conditions.

This project addresses that problem using a **3-class classification model**.

### Prediction Target

| Risk Level  | Meaning                                                 |
| ----------- | ------------------------------------------------------- |
| 🟢 Safe     | Inventory is sufficiently available                     |
| 🟡 At-Risk  | Inventory requires monitoring or replenishment planning |
| 🔴 Imminent | Stockout risk is high before the next supplier delivery |

---

# 📊 Dataset

The project uses multiple related tables that are joined into a single modeling dataset.

### Dataset Structure

| Table                      |   Rows | Purpose                               |
| -------------------------- | -----: | ------------------------------------- |
| `dim_stores.csv`           |     12 | Store-level information               |
| `dim_skus.csv`             |     60 | Product/SKU information               |
| `dim_suppliers.csv`        |     15 | Supplier and reliability information  |
| `dim_events.csv`           |     30 | Calendar and demand-event information |
| `fact_inventory_daily.csv` | 21,600 | Daily inventory observations          |

The fact table contains:

**12 stores × 60 SKUs × 30 days = 21,600 records**

### Date Range

**October 1, 2026 → October 30, 2026**

There are **30 unique dates** in the dataset.

---

# 🔍 Initial Data Validation

Before modeling, the dataset was validated against the expected locked numbers.

```text
Stores:      12
SKUs:        60
Suppliers:   15
Events:      30
Fact rows:   21,600
```

All row-count assertions passed successfully.

### Target Distribution

| Class    | Percentage |
| -------- | ---------: |
| Safe     |     65.42% |
| At-Risk  |     24.01% |
| Imminent |     10.57% |

The target is clearly **imbalanced**, with the Imminent class representing only about 10.6% of observations.

This makes accuracy alone an insufficient evaluation metric.

---

# 🧹 Data Quality Checks

Two intentionally planted data-quality issues were identified and handled.

## 1. Inconsistent City Casing

The `city_display` column contained inconsistent casing.

Examples:

```text
BENGALURU
hyderabad
```

while the standardized city values were:

```text
Bengaluru
Hyderabad
```

Naively grouping by `city_display` resulted in:

```text
8 unique values
```

After standardization:

```text
6 unique cities
```

The issue was handled using string standardization such as:

```python
.str.title()
```

This prevents incorrect duplicate city categories during analysis.

---

## 2. Missing Supplier Reliability Scores

Three suppliers contained the literal value:

```text
N/A
```

These were converted to proper missing values during data loading.

Missing reliability scores:

```text
3 of 15 suppliers
```

The missing values were then imputed using the median reliability score.

```python
median_reliability = dim_suppliers["reliability_score"].median()

dim_suppliers["reliability_score_clean"] = (
    dim_suppliers["reliability_score"]
    .fillna(median_reliability)
)
```

This preserves the supplier information without dropping affected records.

---

# 🔗 Data Integration

The individual datasets were merged into one modeling table using the appropriate keys.

The final modeling table combines:

* Store attributes
* SKU attributes
* Supplier attributes
* Calendar/event attributes
* Daily inventory information
* Demand indicators
* Reorder information

The final joined dataset retained:

```text
21,600 rows
```

This confirmed that the joins did not introduce unwanted fan-out or duplicate records.

---

# 📈 Exploratory Data Analysis

The exploratory analysis focused on three major business signals:

1. Festival demand
2. Supplier reliability
3. Product perishability

---

## 🎆 Festival Demand Impact

The analysis showed a strong increase in Imminent stockout risk during festival periods.

### Imminent Rate

| Period            | Imminent Rate |
| ----------------- | ------------: |
| Non-Festival Days |         9.51% |
| Festival Week     |        23.31% |

The festival-period risk was approximately:

```text
2.45× higher
```

than the non-festival period.

This demonstrates that seasonal demand spikes can significantly increase stockout pressure.

---

## 🪔 Diwali Demand Spike

The dataset contains a Diwali demand period from:

**October 22 → October 26, 2026**

The highest festive demand multiplier occurred on:

**October 25, 2026**

with a:

```text
2.6× demand multiplier
```

This makes the latter part of October a particularly challenging period for inventory planning.

---

## 🚚 Supplier Reliability

Supplier reliability also showed a strong relationship with stockout risk.

### Imminent Rate by Supplier Reliability

| Reliability Tier | Imminent Rate |
| ---------------- | ------------: |
| Low (< 0.75)     |        15.83% |
| Mid (0.75–0.85)  |         3.26% |
| High (≥ 0.85)    |         3.82% |

Low-reliability suppliers had a substantially higher Imminent stockout rate.

This suggests that supplier reliability should be considered when prioritizing replenishment and supplier management.

---

## 🥬 Product Perishability

Perishable products showed a higher Imminent stockout rate.

| Product Type   | Imminent Rate |
| -------------- | ------------: |
| Non-Perishable |         9.30% |
| Perishable     |        12.77% |

Perishable products had approximately a **3.47 percentage-point higher** Imminent risk rate.

---

# 🧠 Feature Engineering

Several operational and temporal features were created to make the dataset more useful for machine learning.

### Key Engineered Features

#### `reorder_gap`

Measures the distance between the reorder point and current closing stock.

```python
df["reorder_gap"] = (
    df["reorder_point"] - df["closing_stock"]
)
```

A larger positive value indicates greater pressure relative to the reorder threshold.

---

#### `days_of_cover_ratio`

Normalizes inventory cover against expected supplier lead time.

```python
df["days_of_cover_ratio"] = (
    df["days_of_cover"] /
    df["lead_time_days_expected"].replace(0, np.nan)
)
```

This helps compare inventory positions across suppliers with different lead times.

---

#### `is_recent_reorder`

Identifies whether a reorder was placed recently for the same Store-SKU combination.

A shifted rolling window was used to avoid directly using future information.

---

#### Festival Features

Additional time-based features were created:

* `day_of_month`
* `days_since_festival_start`
* `is_festival_week_flag`
* `applicable_demand_multiplier`

These features allow the model to capture seasonal demand pressure.

---

#### Product Flags

Categorical indicators were converted into numerical flags:

* `is_perishable_flag`
* `festive_relevant_flag`
* `is_festival_week_flag`

---

# ⏱️ Time-Based Train/Test Split

Because the dataset is a daily panel where the same Store-SKU combinations appear repeatedly, a random row-level split could cause **data leakage**.

Instead, a time-based split was used.

### Training Data

```text
October 1 → October 23
```

### Testing Data

```text
October 24 → October 30
```

This produces:

```text
Training rows: 16,560
Testing rows:   5,040
```

The test period also includes the later part of the Diwali demand spike, making it a meaningful generalization test.

---

# 🤖 Machine Learning Models

Four models were evaluated.

### 1. Majority-Class Baseline

A simple baseline predicting the most common class.

### 2. Multinomial Logistic Regression

Used with:

* StandardScaler
* Class balancing
* Multinomial classification

### 3. Random Forest

Configured with:

* 400 trees
* Maximum depth = 12
* Minimum samples per leaf = 5
* Balanced class weights

### 4. Gradient Boosting

Configured with:

* 300 estimators
* Maximum depth = 3
* Learning rate = 0.05

---

# 📊 Model Performance

The models were evaluated using:

* Precision
* Recall
* F1-score
* Accuracy
* Macro recall

However, the primary business metric was:

> **Recall for the Imminent class**

because failing to identify an actual stockout is more costly than generating an additional warning.

---

## 🏆 Model Results

### Logistic Regression

```text
Accuracy:           91%
Imminent Precision: 70%
Imminent Recall:    97%
Imminent F1-score:  81%
Macro Recall:       86%
```

Logistic Regression achieved the highest Imminent recall.

---

### Random Forest

```text
Accuracy:           95%
Imminent Precision: 85%
Imminent Recall:    82%
Imminent F1-score:  83%
Macro Recall:       91%
```

Random Forest provided a strong balance between overall performance and Imminent detection.

---

### Gradient Boosting

```text
Accuracy:           96%
Imminent Precision: 91%
Imminent Recall:    82%
Imminent F1-score:  86%
Macro Recall:       92%
```

Gradient Boosting achieved the highest overall accuracy and macro recall.

---

### Baseline

```text
Accuracy: 62%
Imminent Recall: 0%
Macro Recall: 33%
```

The baseline completely failed to identify Imminent cases, demonstrating why a majority-class prediction is unsuitable for this business problem.

---

# 🏅 Model Comparison

| Model               | Accuracy | Imminent Recall | Macro Recall |
| ------------------- | -------: | --------------: | -----------: |
| Baseline            |      62% |              0% |          33% |
| Logistic Regression |      91% |         **97%** |          86% |
| Random Forest       |      95% |             82% |          91% |
| Gradient Boosting   |  **96%** |             82% |      **92%** |

---

# 💡 Key Model Insight

There is an important trade-off between **overall performance** and **Imminent stockout detection**.

### Logistic Regression

Best when the business priority is:

> **Catch as many potential stockouts as possible.**

It achieved:

```text
97% Imminent Recall
```

This means it missed very few actual Imminent cases.

### Gradient Boosting

Best when the goal is:

> **Achieve the strongest overall classification performance.**

It achieved:

```text
96% Accuracy
92% Macro Recall
91% Imminent Precision
```

### Random Forest

Provided another strong balanced solution with:

```text
95% Accuracy
91% Macro Recall
85% Imminent Precision
```

---

# 📌 Business Interpretation

The results reveal several important inventory-management signals.

### 1. Festival periods require proactive inventory planning

The Imminent rate increased from:

```text
9.51% → 23.31%
```

during festival week.

QuickCart could therefore increase safety stock and replenishment frequency before major demand events.

---

### 2. Supplier reliability matters

Low-reliability suppliers showed:

```text
15.83% Imminent rate
```

compared with only:

```text
3.26% for mid-reliability suppliers
3.82% for high-reliability suppliers
```

Supplier reliability can therefore be incorporated into replenishment prioritization.

---

### 3. Perishable products need closer monitoring

Perishable SKUs showed:

```text
12.77% Imminent risk
```

versus:

```text
9.30% for non-perishable SKUs
```

These products may require more frequent monitoring because inventory decisions have both availability and shelf-life implications.

---

### 4. Imbalanced classification requires business-focused metrics

Because only:

```text
10.57%
```

of all records belong to the Imminent class, accuracy alone could be misleading.

The majority-class baseline achieved:

```text
62% accuracy
```

while detecting:

```text
0% of Imminent cases
```

This demonstrates why **class-specific recall and macro-level metrics** are essential for this project.

---

# 📊 Project Workflow

```text
Raw CSV Tables
      ↓
Data Loading
      ↓
Sanity Checks
      ↓
Data Quality Fixes
      ↓
Multi-table Joins
      ↓
Exploratory Data Analysis
      ↓
Feature Engineering
      ↓
Time-Based Train/Test Split
      ↓
Model Training
      ↓
Model Evaluation
      ↓
Feature Importance
      ↓
Business Insights
```

---

# 🛠️ Technologies Used

### Programming

* Python

### Libraries

* Pandas
* NumPy
* Matplotlib
* Scikit-learn

### Machine Learning

* Dummy Classifier
* Logistic Regression
* Random Forest
* Gradient Boosting

### Environment

* Google Colab
* Jupyter Notebook

---

# 📁 Project Structure

```text
QuickCart-Stockout-Risk/
│
├── Quickcart_sortRisk.ipynb
│
├── dim_stores.csv
├── dim_skus.csv
├── dim_suppliers.csv
├── dim_events.csv
├── fact_inventory_daily.csv
│
└── README.md
```

---

# 🚀 How to Run the Project

## 1. Clone the Repository

```bash
git clone <your-github-repository-url>
```

## 2. Open the Notebook

Open:

```text
Quickcart_sortRisk.ipynb
```

using:

* Google Colab
* Jupyter Notebook
* VS Code with Jupyter support

## 3. Upload the Dataset Files

Make sure the following files are available in the notebook environment:

```text
dim_stores.csv
dim_skus.csv
dim_suppliers.csv
dim_events.csv
fact_inventory_daily.csv
```

## 4. Run the Notebook

Run all cells from top to bottom.

The notebook performs:

```text
Data Loading
→ Validation
→ Cleaning
→ Joining
→ EDA
→ Feature Engineering
→ Training
→ Evaluation
→ Feature Importance
```

---

# 📈 Visualizations

The notebook includes visual analysis for:

* Target class distribution
* Supplier reliability vs. stockout risk
* Festival vs. non-festival risk
* Daily Imminent stockout rate
* Diwali demand period
* Model comparison
* Feature importance

These visualizations help convert raw inventory data into actionable business insights.

---

# 🔎 Important Takeaways

### 📌 Finding 1 — Festival demand is a major stockout driver

```text
23.31% Imminent during festival week
vs.
9.51% on non-festival days
```

### 📌 Finding 2 — Low supplier reliability increases risk

```text
15.83% Imminent for low-reliability suppliers
```

### 📌 Finding 3 — Perishable products have higher risk

```text
12.77% vs 9.30%
```

### 📌 Finding 4 — Machine learning significantly improves detection

The baseline detected:

```text
0% of Imminent cases
```

while Logistic Regression achieved:

```text
97% Imminent Recall
```

### 📌 Finding 5 — Gradient Boosting achieved the strongest overall result

```text
96% Accuracy
92% Macro Recall
91% Imminent Precision
```

---

# 🎯 Recommended Business Use

For a real inventory-warning system, the model should not be selected using accuracy alone.

A practical deployment strategy could use:

### 🚨 High-Priority Alert

Use the model to identify:

```text
Imminent
```

SKUs for immediate replenishment action.

### ⚠️ Monitoring Queue

Use:

```text
At-Risk
```

predictions for warehouse planners to monitor upcoming inventory pressure.

### ✅ Normal Inventory

Keep:

```text
Safe
```

SKUs under standard inventory monitoring.

Because stockout misses can be expensive, a model with stronger **Imminent recall** may be preferable even if it produces more false alarms.

---

# 🔮 Future Improvements

Possible extensions include:

* Hyperparameter tuning
* XGBoost / LightGBM comparison
* Probability-based risk scoring
* Threshold optimization for Imminent detection
* SHAP-based model explainability
* Longer historical datasets
* Weekly/monthly seasonality
* Supplier delay prediction
* Demand forecasting
* Automated replenishment recommendations
* Real-time inventory dashboard
* Deployment using Streamlit
* Cloud-based model serving

---

# 💻 Future Dashboard Concept

The project can be extended into an interactive inventory intelligence dashboard containing:

```text
Total SKUs
        ↓
Safe / At-Risk / Imminent counts
        ↓
Stockout Risk by Store
        ↓
Stockout Risk by Category
        ↓
Supplier Reliability
        ↓
Festival Demand Impact
        ↓
Top Imminent SKUs
        ↓
Recommended Replenishment Priority
```

This would transform the notebook from a machine-learning experiment into a practical **Inventory Decision Support System**.

---

# 👩‍💻 Author

**Supriya MS**

BE — Artificial Intelligence & Machine Learning

---

# ⭐ Project Highlights

```text
✔ 21,600 inventory records
✔ 12 stores
✔ 60 SKUs
✔ 15 suppliers
✔ 30-day inventory panel
✔ Multi-table data integration
✔ Data-quality validation
✔ Feature engineering
✔ Time-based leakage-safe split
✔ Imbalanced 3-class classification
✔ Logistic Regression
✔ Random Forest
✔ Gradient Boosting
✔ 96% best overall accuracy
✔ 97% Imminent recall
✔ Business-focused inventory insights
```

---

# 📌 Conclusion

The QuickCart Stockout Risk Classification project demonstrates how machine learning can be used to transform warehouse inventory data into an **early-warning stockout detection system**.

The analysis showed that **festival demand, supplier reliability, perishability, inventory coverage, reorder conditions, and lead times** are important signals for predicting stockout risk.

Among the evaluated models:

* **Logistic Regression** achieved the strongest **Imminent recall at 97%**
* **Random Forest** achieved **95% accuracy**
* **Gradient Boosting** achieved the strongest overall performance with **96% accuracy and 92% macro recall**

The project demonstrates the importance of combining **data engineering, exploratory analysis, feature engineering, leakage-safe validation, machine learning, and business interpretation** rather than relying on accuracy alone.

---

## ⭐ If you found this project useful

Feel free to explore the notebook, connect with me, and share your feedback.

**#Python #MachineLearning #DataScience #ArtificialIntelligence #AIML #InventoryManagement #SupplyChain #DataAnalytics #Pandas #NumPy #ScikitLearn #GoogleColab #RandomForest #GradientBoosting #LogisticRegression #EDA #FeatureEngineering #BusinessAnalytics #PredictiveAnalytics #StockoutPrediction**
