# Spotify User Segmentation & Churn Analysis  
*A Data-Driven Approach to Understanding and Retaining Spotify Users*  

---
## Table of Contents

- [Project Overview](#project-overview)
- [Data Description](#data-description)
- [Data Cleaning and Preprocessing](#data-cleaning-and-preprocessing)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Free User Segmentation](#free-user-segmentation)
- [Premium User Churn Prediction](#premium-user-churn-prediction)
- [Conclusion & Business Insights](#conclusion--business-insights)
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

[🗂️ View the full dataset description](https://github.com/chienhao-wang/spotify_user_analysis/blob/main/dataset_description.md)

**Target variables:** `premium_sub_willingness` (Yes/No)

---

## Data Cleaning and Preprocessing

Data cleaning and preprocessing were conducted using `pandas`, ensuring type consistency, missing value handling, and categorical encoding.

**Missing values per column:**  
| Columns                        | Count |
| ------------------------------ | ----- |
| Age                            | 0     |
| Gender                         | 0     |
| spotify_usage_period           | 0     |
| spotify_listening_device       | 0     |
| spotify_subscription_plan      | 0     |
| premium_sub_willingness        | 0     |
| preffered_premium_plan         | 208   |
| preferred_listening_content    | 0     |
| fav_music_genre                | 0     |
| music_time_slot                | 0     |
| music_Influencial_mood         | 0     |
| music_lis_frequency            | 0     |
| music_expl_method              | 0     |
| music_recc_rating              | 0     |
| pod_lis_frequency              | 0     |
| fav_pod_genre                  | 148   |
| preffered_pod_format           | 140   |
| pod_host_preference            | 141   |
| preffered_pod_duration         | 129   |
| pod_variety_satisfaction       | 0     |  

Since these columns might be optional, I replaced the "No Response" entries with missing values.

**Split Multi-Response Columns by Comma:**  
Since several survey questions allowed multiple selections, the columns `spotify_listening_device`, `music_Influencial_mood`, `music_lis_frequency`, and `music_expl_method` contain multiple options separated by commas. Therefore, I used `.str.split(',')` to separate all responses and store them in individual dataframes. Additionally, to prepare for subsequent statistical modelling, I created a new dataframe to generate dummy variables for each option.

---

## Exploratory Data Analysis (EDA)  

Visualisations were generated using **matplotlib**, **seaborn**, and **squarify** to understand user behaviour patterns.  

### User Overview & Listening Behaviour
- The majority of users are **20–35 years old**, with **female** listeners forming the largest group.  
- Over **75%** stream mainly via **smartphones**, and most have used Spotify for **2+ years**.

<p align="center">
  <img src="EDA_Charts/1_eda_age_bar.png" width="48%">
  <img src="EDA_Charts/4_eda_listen_device_bar.png" width="48%">
  <br>
  <em>Figure: Age Group & Listening Devices Distribution</em>
</p>

### Music Preferences   
- Nearly **80%** primarily listen to **music**, with *Melody*, *Pop*, and *Rap* as top genres.  
- **Night-time** and **workout** are the most popular listening moments.  
- Around **70%** are moderately satisfied with Spotify’s **music recommendations**.

<p align="center">
  <img src="EDA_Charts/8_eda_music_genre_by_timeslot_bar.png" width="900">
  <br>
  <em>Figure 2: Listeners by Favourite Music Genre and Timeslot</em>
</p>

<p align="center">
  <img src="EDA_Charts/10_eda_music_lis_freq_bar.png" width="48%">
  <img src="EDA_Charts/11_eda_music_recc_rating_bar.png" width="48%">
  <br>
  <em>Figure 3: Listeners by Music Listening Frequency and Distribution of Music Recommendation Rating</em>
</p>

### Podcast Insights    
- **Podcasts** attract daily listeners, mainly for **short-form comedy or storytelling** content.

<p align="center">
  <img src="EDA_Charts/14_eda_podcast_format_bar.png" width="900">
  <br>
  <em>Figure 4: Listeners by Preferred Podcast Format</em>
</p>

### Premium Willingness  
- **73%** of free users are unwilling to upgrade, while **75%** of premium users plan to continue — early-stage premium users show the highest churn risk.

<p align="center">
  <img src="EDA_Charts/17_eda_willingness_by_plan_bar.png" width="900">
  <br>
  <em>Figure 5: Premium Subscription Willingness Distribution</em>
</p>

[📊 View all charts in EDA](https://github.com/chienhao-wang/spotify_user_analysis/tree/main/EDA_Charts)

## Free User Segmentation 

This section focuses on understanding the behavioural and attitudinal patterns of **free-tier Spotify users** to identify potential converters for Premium plans.  
The analysis combined **exploratory visualisations**, **statistical testing**, and **K-Means clustering** to uncover key drivers of upgrade willingness.  

### Behavioural Analysis  

<p align="center">
  <img src="Free_Users_Charts/1_free_device_convert_treemap.png" width="48%">
  <img src="Free_Users_Charts/2_free_convert_by_preferred_plan_bar.png" width="48%">
  <br>
  <em>Figure 1–2: Device Usage and Preferred Pricing Plan</em>
</p>

- **Device Usage:** Conversion rates are similar across devices, but **smartphone** and **wearable users** show slightly higher upgrade likelihood—suggesting that on-the-go listeners value ad-free and offline features.  
- **Pricing Preferences:** Users lean toward **multi-user plans** (Duo/Family), implying that **cost-per-user** and shared value drive premium adoption decisions.  

<p align="center">
  <img src="Free_Users_Charts/3_free_convert_by_content_line.png" width="48%">
  <img src="Free_Users_Charts/4_free_convert_by_usage_and_content_bar.png" width="48%">
  <br>
  <em>Figure 3–4: Content Type and Tenure Analysis</em>
</p>

- **Content Type:** Podcast listeners demonstrate a **stronger upgrade intention** than music listeners, indicating opportunity for **podcast-focused promotions**.  
- **Tenure Effect:** Among podcast users, **1–2 years of active usage** correlates with higher conversion probability, suggesting that long-term engagement supports premium readiness.  

### Music & Podcast Insights  

<p align="center">
  <img src="Free_Users_Charts/5_free_convert_by_genre_treemap.png" width="48%">
  <img src="Free_Users_Charts/7_free_convert_by_lis_freq_treemap.png" width="48%">
  <br>
  <em>Figure 5–6: Music Genre and Listening Context</em>
</p>

- **Music Listeners:** Potential upgraders prefer **Melody**, **Pop**, **Rap**, and **Classical** genres, typically streaming during **afternoon hours**—pointing to genre-time relationships that can inform targeted recommendations.  
- **Listening Context:** Most free users listen **while travelling, relaxing, or studying**, showing Spotify’s role as a **lifestyle companion** for leisure and productivity.  

<p align="center">
  <img src="Free_Users_Charts/8_free_convert_by_pod_genre_treemap.png" width="600">
  <br>
  <em>Figure 7: Podcast Genre Conversion</em>
</p>

- **Podcast Genres:** Health & Fitness, Sports, and Lifestyle topics exhibit the **highest upgrade rates**, highlighting active and wellness-oriented listeners as strong conversion segments.  

### Statistical Association & Segmentation  

<p align="center">
  <img src="Free_Users_Charts/10_free_cluster_elbow.png" width="48%">
  <br>
  <em>Figure 8: Cramér’s V Correlations</em>
</p>

- **Cramér’s V analysis** identified the most influential factors linked to premium willingness:  
  `preferred_premium_plan`, `device_type`, and `fav_music_genre` ranked highest (C > 0.30).  
- Variables with **p < 0.01** from the chi-square test were used in the **K-Means model**, which determined **three behavioural clusters (k = 3)**.

**Cluster Profile:**  
| Cluster | Size | Conversion Rate | Key Traits | Suggested Marketing Actions |
|----------|------|------------------|-------------|------------------------------|
| **0 – Relaxed Streamers** | 218 users | 18% | Mobile-only, listen at night for relaxation or stress relief | Offer affordable *Student/Introductory Plans* and promote mood-based playlists to increase engagement. |
| **1 – Multi-Device Enthusiasts** | 82 users | 63% | Listen across devices (PC, smart speaker, phone), high podcast activity, prefer Duo/Family plans | Target with *premium bundle offers*, emphasising ad-free listening and shared plan value. |
| **2 – Casual Listeners** | 124 users | 20% | Primarily music listeners, lower session frequency, moderate engagement | Retain via *in-app gamification* or personalised reminders to sustain activity. |

**Cluster Insights:**  
- **Cluster 1 – Multi-Device Enthusiasts:** Smallest segment (~82 users) but **most conversion-ready** (> 60% upgrade intention). Frequent podcast listeners, diverse moods, and preference for Duo/Family plans—ideal for targeted Premium offers.  
- **Cluster 0 & 2 – Casual Mobile Listeners:** Mostly **mobile-only**, using Spotify at night for relaxation or stress relief. Low upgrade interest (< 20%), representing the **retention and nurturing group**.  

**Cluster Visualisation**  


### Key Takeaways  

- **Multi-device and podcast engagement** strongly predict premium conversion potential.  
- **Shared plans** and **long-term listening tenure** drive higher upgrade interest.  
- **Cluster 1** represents the prime marketing segment—active, cross-device users with emotional diversity and higher spending power.  
- Future campaigns could focus on promoting **ad-free multi-device access**, **exclusive podcast content**, and **value-driven family plans** to maximise conversion.

---

## Premium User Churn Prediction  

This section analyses **Premium users’ churn behaviour**, exploring how plan type, usage duration, and content preferences affect retention.  
Through **visual exploration**, **statistical testing**, and a **logistic regression model**, this stage identifies the key drivers behind premium churn and potential retention levers.  

---

### Churn Overview  

**Overall Churn Rate:** 25% among Premium users  
**User Distribution:** 72 retained vs. 24 churned  

<p align="center">
  <img src="Premium_Users_Charts/1_premium_churn_by_plan_bar.png" width="48%">
  <img src="Premium_Users_Charts/2_premium_churn_by_usage_bar.png" width="48%">
  <br>
  <em>Figure 1–2: Churn Rate by Preferred Premium Plan and Usage Period</em>
</p>

**Plan Type:**  
Higher-tier plans such as **Family (£19.99)** and **Duo (£16.99)** show **30–40% churn**, implying potential **price dissatisfaction** or **shared-plan fatigue**.  
In contrast, **Individual** and **Student** plans demonstrate lower churn, indicating **clearer value perception** among solo users.  

**Usage Tenure:**  
Churn rate drops steadily with platform experience — from **over 40% (<6 months)** to **below 10% (>2 years)** — suggesting that **longer usage enhances loyalty**.  

### Content & Engagement Insights  

<p align="center">
  <img src="Premium_Users_Charts/4_premium_churn_by_genre_bar.png" width="48%">
  <img src="Premium_Users_Charts/5_premium_churn_by_pod_format_bar.png" width="48%">
  <br>
  <em>Figure 3–4: Churn Rate by Favourite Music Genre and Podcast Format</em>
</p>

**Music Preference:**  
Listeners who prefer **Classical** or **All genres** experience **higher churn**, whereas **Pop** and **Electronic/Dance** fans show **greater retention stability** — indicating stronger engagement among mainstream or upbeat genres.  

**Podcast Format:**  
**Interview-style** podcasts show the **highest churn rate (~42%)**, while **Educational** formats retain users better — highlighting how **knowledge-based content** sustains engagement among premium subscribers.  

### Statistical Modelling  

A **chi-square test** confirmed significant relationships between churn and variables including  
`preferred_premium_plan`, `spotify_usage_period`, and `fav_music_genre` *(p < 0.01)*.  

A **Logistic Regression model** was applied using `class_weight='balanced'` to correct class imbalance.  

**Model Performance:**  
| Metric               | Score |
|----------------------|-------|
| Accuracy             | 0.70  |
| Precision (weighted) | 0.70  |
| Recall (weighted)    | 0.70  |
| F1-score (weighted)  | 0.70  |

<p align="center">
  <img src="Premium_Users_Charts/6_premium_churn_confusion_matrix.png" width="450">
  <br>
  <em>Figure 5: Logistic Regression Confusion Matrix</em>
</p>

<p align="center">
  <img src="Premium_Users_Charts/7_premium_churn_model_coefficients.png" width="900">
  <br>
  <em>Figure 6: Top Factors Driving and Preventing Churn</em>
</p>

| **Risk Drivers (↑ Churn)**             | **Protective Factors (↓ Churn)** |
|----------------------------------------|----------------------------------|
| Family / Duo plans                     | Podcast-preferring users         |
| Classical music fans                   | Wearable device users            |
| Very dissatisfied with podcast variety | Age 20–35                        |
| Workout-session listeners              | High music recommendation rating |
| Unknown podcast hosts                  | Student / Individual plan users  |

### Key Findings  

- **High-tier plans** face **elevated churn** due to shared-account fatigue and cost sensitivity.  
- **Retention strengthens with tenure** — long-term users are less likely to churn.  
- **Podcast satisfaction** is a major retention factor; dissatisfaction significantly predicts churn.  
- **Younger, wearable-device users** show **low churn**, reflecting tech-adaptive engagement patterns.  

### Marketing Implications  

- **Reposition Family & Duo Plans:** Reinforce shared value with group playlists or family rewards.  
- **Enhance Early-Stage Retention:** Improve onboarding, deliver personalised recommendations, and introduce loyalty incentives within the first 6 months.  
- **Optimise Content Strategy:** Promote *educational* podcasts and *electronic/pop* genres to enhance engagement among at-risk subscribers.  

---
## Conclusion & Business Insights  

### Strategic Summary  
Through comprehensive behavioural segmentation and churn modelling, this analysis reveals clear pathways for **increasing Spotify’s conversion rate** and **reducing subscriber attrition**.  
Data-driven marketing actions can be tailored to specific audience clusters to optimise both **growth** and **retention** outcomes.  

---

### 1. **Free User Insights – Converting Intent into Action**  
- **Multi-Device Enthusiasts** (Cluster 1) represent the **highest conversion potential** (> 60%) due to their cross-device activity and preference for Duo/Family plans.  
  → *Target with ad-free multi-device bundles and exclusive podcast access to drive upgrades.*  
- **Casual Mobile Listeners** (Clusters 0 & 2) show low upgrade intention (< 20%) but high nightly usage.  
  → *Engage through mood-based playlists, gamified challenges, and affordable student or introductory offers.*  
- **Podcast engagement** strongly correlates with willingness to upgrade.  
  → *Expand podcast-focused campaigns and highlight offline listening benefits for commuters and gym users.*  

---

### 2. **Premium User Insights – Strengthening Retention**  
- **Churn Concentration:** High-tier plans (Family & Duo) face **30–40 % churn**, mainly from cost sensitivity and shared-plan fatigue.  
  → *Introduce family reward points or shared playlists to reinforce value perception.*  
- **Tenure Loyalty Effect:** Churn falls sharply after 2 years of usage, showing that early-stage retention tactics (onboarding, loyalty points, trial extensions) are vital.  
- **Content Impact:** *Educational* and *Pop/Electronic* listeners exhibit the strongest retention, while *Classical* and *Interview-style podcast* users are most at risk.  
  → *Rebalance algorithmic recommendations and create curated “knowledge-based” or upbeat playlists for at-risk cohorts.*  

---

### 3. **Commercial Implications**  
| Objective | Data-Driven Action | Expected Impact |
|------------|-------------------|-----------------|
| **Increase Free-to-Premium Conversion** | Promote Duo/Family plans to Cluster 1 with ad-free podcast bundles | +15–20 % conversion uplift |
| **Reduce Early-Stage Churn** | Personalised onboarding, push notifications, and satisfaction tracking | Lower 6-month churn by > 10 % |
| **Enhance Long-Term Retention** | Introduce loyalty incentives and multi-device continuity features | Increase lifetime value across cohorts |
| **Optimise Content Strategy** | Prioritise podcast satisfaction and genre diversity in recommendations | Strengthen platform stickiness |
