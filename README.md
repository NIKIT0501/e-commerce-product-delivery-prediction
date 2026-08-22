# 📦 E-Commerce Product Delivery Prediction

<div align="center">

**Predicting on-time delivery for an international e-commerce company using machine learning**

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat-square&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Wrangling-150458?style=flat-square&logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Modeling-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-Visualization-4C72B0?style=flat-square)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square)

</div>

<p align="center">
  <img src="https://globalskylogistics.com/wp-content/uploads/2018/04/THE-CHANGING-NATURE-OF-E-COMMERCE-DELIVERY.jpg" alt="E-commerce delivery" width="720">
</p>

---

## 📌 Overview

Late deliveries erode customer trust and drive up support costs. This project builds an **end-to-end classification pipeline** that predicts whether a shipment from an international e-commerce electronics retailer will arrive **on time or late**, and surfaces the operational and behavioral drivers behind delivery performance.

The workflow covers the full data science lifecycle: data cleaning → exploratory data analysis → feature encoding → model training with hyperparameter tuning → model comparison → business insights.

**Business questions answered:**
- Which shipments are at risk of arriving late — *before* they're dispatched?
- Does warehouse location, shipping mode, or product type actually affect delivery time?
- What customer behaviors (calls, ratings, repeat purchases) correlate with delivery outcomes?
- Where should the business focus to improve on-time delivery rates?

---

## 🗂️ Repository Contents

| File | Description |
|---|---|
| `E-Commerce_Product_Delivery_Prediction.ipynb` | Full Jupyter notebook — EDA, preprocessing, modeling, evaluation |
| `E-Commerce_Product_Delivery_Prediction.pdf` | Exported notebook report (static, shareable version) |
| `E_Commerce.csv` | Raw dataset (10,999 shipment records) |
| `README.md` | You are here |

---

## 🧾 Dataset

**10,999 shipments · 12 features · binary target**

| Feature | Description |
|---|---|
| `Warehouse_block` | Warehouse the order shipped from (A–E) |
| `Mode_of_Shipment` | Ship, Flight, or Road |
| `Customer_care_calls` | Number of inquiry calls made about the shipment |
| `Customer_rating` | 1 (worst) – 5 (best) |
| `Cost_of_the_Product` | Product cost in USD |
| `Prior_purchases` | Customer's purchase history count |
| `Product_importance` | Low / Medium / High |
| `Gender` | Customer gender |
| `Discount_offered` | % discount applied |
| `Weight_in_gms` | Product weight |
| `Reached.on.Time_Y.N` | **Target** — 1 = late, 0 = on time |

No missing values or duplicate records were present in the raw data.

---

## 🔍 Key Insights from EDA

- 📉 **Weight matters most.** Shipments between **2,500–3,500g** are far more likely to arrive on time; anything **above 4,500g** is a strong late-delivery signal.
- 💵 **Cost matters too.** Products priced **under $250** are delivered on time more consistently.
- 🎁 **Discount is the strongest single signal.** Orders with **>10% discount** are much more likely to arrive on time; the **0–10% discount band** dominates the late-delivery cases.
- 📞 **Customer care calls spike with cost and risk.** Higher-cost products trigger more inquiry calls — customers are proactively checking on shipments they're anxious about, which correlates with delay.
- 🔁 **Loyalty predicts reliability.** Customers with more prior purchases see a higher rate of on-time delivery.
- 🚢 **Logistics network is skewed but not predictive.** ~32% of all shipments originate from **Warehouse F** (suggesting proximity to a seaport) and ship predominantly via **Ship**, but warehouse and shipping mode show **no meaningful effect** on delivery outcomes.
- 🚻 **Gender is a non-factor** — delivery performance is statistically identical across both groups.

---

## 🛠️ Methodology

```
Raw Data → Cleaning → EDA → Label Encoding → Train/Test Split (80/20)
         → GridSearchCV Hyperparameter Tuning → Model Training
         → Evaluation (Accuracy, Precision, Recall, F1, Confusion Matrix)
         → Model Comparison
```

**Preprocessing:** dropped the non-predictive `ID` column, label-encoded categorical features (`Warehouse_block`, `Mode_of_Shipment`, `Product_importance`, `Gender`).

**Models trained** (each tuned via `GridSearchCV`, 5-fold CV):

| Model | Tuned Hyperparameters |
|---|---|
| 🌲 Random Forest Classifier | `max_depth`, `min_samples_leaf`, `min_samples_split`, `criterion` |
| 🌳 Decision Tree Classifier | `max_depth`, `min_samples_leaf`, `min_samples_split`, `criterion` |
| 📈 Logistic Regression | default |
| 📍 K-Nearest Neighbors | default |

---

## 📊 Results

| Model | Accuracy | Precision (Late) | Recall (Late) | F1 (Late) |
|---|:---:|:---:|:---:|:---:|
| 🏆 **Decision Tree Classifier** | **69%** | 0.95 | 0.49 | 0.65 |
| Random Forest Classifier | 68% | 0.87 | 0.54 | 0.66 |
| K-Nearest Neighbors | 65% | 0.71 | 0.68 | 0.70 |
| Logistic Regression | 63% | 0.69 | 0.67 | 0.68 |

> The **Decision Tree Classifier** delivered the best overall accuracy, with very high precision on late-delivery predictions — meaning when it flags a shipment as "at risk," it's right the vast majority of the time. This makes it well suited for a proactive alerting workflow where minimizing false alarms matters.

---

## 💡 Business Recommendations

1. **Flag high-weight, high-cost orders at checkout** for expedited handling — these are the clearest early risk indicators.
2. **Rethink the 0–10% discount tier** — this segment consistently under-performs on delivery timeliness and may warrant a logistics or fulfillment review.
3. **Use customer care call volume as a leading indicator**, not just a support metric — a spike in inquiries on a given order is a signal worth acting on before it becomes a complaint.
4. **Invest in warehouse network diversification** — over-reliance on Warehouse F creates concentration risk even though it isn't currently a predictor of delay.

---

## ⚙️ Getting Started

```bash
# Clone the repository
git clone https://github.com/<your-username>/ecommerce-delivery-prediction.git
cd ecommerce-delivery-prediction

# Install dependencies
pip install numpy pandas matplotlib seaborn scikit-learn jupyter

# Launch the notebook
jupyter notebook E-Commerce_Product_Delivery_Prediction.ipynb
```

---

## 🚀 Future Work

- [ ] Address class imbalance with SMOTE / class weighting to improve recall on late deliveries
- [ ] Engineer interaction features (e.g., weight × discount, cost × calls)
- [ ] Benchmark gradient-boosted models (XGBoost, LightGBM, CatBoost)
- [ ] Deploy the tuned model behind a lightweight API for real-time risk scoring at checkout
- [ ] Build a monitoring dashboard tracking prediction drift over time

---

## 🧰 Tech Stack

`Python` · `NumPy` · `Pandas` · `Matplotlib` · `Seaborn` · `scikit-learn` · `Jupyter Notebook`

---

## 👤 Author

**[Your Name]**
Data Scientist | Machine Learning Practitioner

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=flat-square&logo=linkedin&logoColor=white)](#)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?style=flat-square&logo=github&logoColor=white)](#)
[![Portfolio](https://img.shields.io/badge/Portfolio-Visit-000000?style=flat-square&logo=vercel&logoColor=white)](#)

*Have a project or role in mind? Let's connect — I'm always open to discussing data science opportunities.*

---

<div align="center">
<sub>⭐ If you found this project useful, consider giving it a star!</sub>
</div>
