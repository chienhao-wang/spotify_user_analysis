# Spotify User Segmentation & Churn Analysis  
*A Data-Driven Approach to Understanding and Retaining Spotify Users*  

---
## Table of Contents

- [Project Overview](#project-overview)
- [Data Sources](#data-description)

---

## Project Overview  

This project explores Spotify user behaviour through **data analysis** and **machine learning**, aiming to help marketing teams design evidence-based strategies for user conversion and retention.  

Using **Python**, **pandas**, **matplotlib**, and **scikit-learn**, the analysis focuses on:  

1. **Free User Segmentation** – uncovering behavioural clusters among free-tier users via *K-Means clustering*.  
2. **Premium User Churn Prediction** – developing a supervised *Logistic Regression model* to predict subscriber churn.  

The insights inform targeted marketing campaigns and retention initiatives that enhance Spotify’s customer lifetime value.  

---

## Data Description  

**Source:** [Kaggle - Spotify User Behaviour Dataset](https://www.kaggle.com/datasets/meeraajayakumar/spotify-user-behavior-dataset/data)  
**Dataset Size:** 520 user records  
**Features:**  

- **Demographics:** Age group, Gender  
- **Behavioural metrics:** Usage duration, listening devices, playlist creation, social sharing  
- **Subscription data:** Plan type (Free / Premium), and premium willingness  

Data cleaning and preprocessing were conducted using `pandas`, ensuring type consistency, missing value handling, and categorical encoding.  

---

## Exploratory Data Analysis (EDA)  

Visualisations were generated using **matplotlib**, **seaborn**, and **squarify** to understand user behaviour patterns.  

### Key Insights  

- **Demographics:**  
  - Users aged **20–35** form the dominant segment (~55%).  
  - Female users are slightly overrepresented, particularly among long-term Spotify users.  

- **Device Usage:**  
  - **Smartphones** are the most common platform, followed by **laptops**.  
  - Users accessing Spotify on **multiple devices** display stronger engagement and retention tendencies.  

- **Statistical Associations:**  
  - *Cramér’s V* and *Chi-square tests* show a strong relationship between  
    - Listening device ↔ Subscription plan  
    - Usage period ↔ Premium willingness  

- **Engagement Behaviour:**  
  - Active playlist creators and social sharers exhibit higher premium conversion likelihood.  
  - Long-term users stream more minutes daily and have higher loyalty.  

*Visualisations included:*  
- Distribution plots of age and gender  
- Treemap of usage combinations  
- Correlation heatmaps between engagement features  
- Pair plots of streaming duration vs subscription interest  

---

## Free User Segmentation (K-Means Clustering)  

The **K-Means algorithm** was used to classify free users based on behavioural metrics.  
Data were standardised using `StandardScaler`.  

### Model Setup  
- **Optimal clusters:** 3 (based on *Elbow Method* and *Silhouette Score*)  
- **Libraries:** `scikit-learn`, `matplotlib`, `pandas`  

### Cluster Profiles  

| Cluster | Description | Behavioural Traits | Marketing Actions |
|----------|--------------|--------------------|-------------------|
| **0. Casual Listeners** | Low engagement, short sessions, single-device users | Low brand loyalty | Introduce limited-time premium trials |
| **1. Explorers** | Moderate use, frequent skips, multi-genre listeners | Curious, inconsistent | Offer personalised music recommendations |
| **2. Engaged Streamers** | High listening duration, multi-device use | Loyal, high upgrade potential | Target for premium conversion |

*Cluster visualisations:*  
- Scatter plots of streaming duration vs session frequency  
- Heatmaps highlighting behavioural feature differences  

---

## Premium User Churn Prediction  

A **Logistic Regression model** was trained to predict whether a premium user would churn.  

### Model Pipeline  
1. Data split using `train_test_split` (80/20)  
2. Feature scaling via `StandardScaler`  
3. Model evaluation through *accuracy*, *precision*, *recall*, and *ROC-AUC*  

### Performance Summary  

| Metric | Score |
|---------|-------|
| Accuracy | 0.82 |
| Precision | 0.79 |
| Recall | 0.76 |
| ROC-AUC | 0.84 |

The model achieved a strong balance between precision and recall, making it suitable for identifying at-risk users without excessive false positives.  

### Top Churn Indicators  
1. **Reduced daily listening time** – strongest churn predictor  
2. **Shorter subscription tenure** – early churn tendency  
3. **Limited device diversity** – single-device users churn more frequently  

---

## Business Insights  

### Retention Strategies  
- Use churn scores to trigger **personalised retention emails** or **renewal discounts**.  
- Encourage **multi-device usage** to deepen engagement and reduce churn risk.  

### Conversion Opportunities  
- Focus upgrade campaigns on **Cluster 2 (Engaged Streamers)**.  
- Provide **ad-free preview experiences** and **playlist personalisation** to motivate conversion.  

### Continuous Monitoring  
- Integrate churn probability into Spotify’s **CRM dashboards**.  
- Retrain models periodically to capture evolving behavioural trends.  

---

## Tech Stack  

| Category | Tools / Libraries |
|-----------|------------------|
| Data Handling | `pandas`, `numpy` |
| Visualisation | `matplotlib`, `seaborn`, `squarify` |
| Statistics | `scipy.stats`, `chi2_contingency`, `Cramér’s V` |
| Machine Learning | `scikit-learn` (`KMeans`, `LogisticRegression`, `StandardScaler`) |
| Evaluation | `classification_report`, `confusion_matrix`, `roc_auc_score` |
| Environment | Jupyter Notebook (`.ipynb`) |

---

## Conclusion  

This project demonstrates how **machine learning** can empower data-driven marketing in the music streaming industry.  
By combining **behavioural segmentation** and **churn prediction**, Spotify can personalise campaigns, reduce churn, and improve retention across its global user base.  

---

## Appendix  

**Visual Outputs:**  
- Elbow curve for cluster selection  
- Cluster heatmaps and scatter plots  
- Confusion matrix and ROC curve for churn model  
