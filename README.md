# E-Commerce Product Delivery Prediction

An end-to-end machine learning project to analyze shipment performance, predict delivery delays, and translate model outputs into actionable business insights.

## Project Overview

Late deliveries can reduce customer satisfaction, increase support requests, and create operational inefficiencies. This project uses historical shipment data to classify whether an e-commerce order is likely to arrive late or on time.

The project covers data quality checks, exploratory data analysis (EDA), feature preprocessing, model training, hyperparameter tuning, model evaluation, and business recommendations.

**Objective:** Explore the factors associated with delivery outcomes and build a classification model that can help identify shipments requiring further attention.

## Key Highlights

- Analyzed **10,999 shipment records** from an e-commerce delivery dataset.
- Performed data cleaning checks and exploratory data analysis using Python.
- Examined shipment characteristics, including product weight, product cost, discounts, customer-care calls, and prior purchases.
- Trained and compared four classification algorithms: Logistic Regression, Decision Tree, Random Forest, and K-Nearest Neighbors (KNN).
- Used GridSearchCV with 5-fold cross-validation to tune the Decision Tree and Random Forest models.
- Evaluated model performance using accuracy, precision, recall, F1-score, and a confusion matrix.
- Translated analytical findings into potential operational recommendations.

## Business Problem

E-commerce businesses need to understand which shipment characteristics are associated with late delivery so that they can investigate risks and improve fulfillment operations.

This project explores the following questions:

1. Which product and customer-related features are associated with delivery outcomes?
2. How do product weight, cost, and discount relate to on-time delivery?
3. Do shipment mode and warehouse location show differences in delivery performance?
4. How do the classification models compare when identifying late deliveries?
5. How could the findings support delivery monitoring and operational decisions?

## Dataset

The project uses `E_Commerce.csv`, containing 10,999 shipment records. The target column is `Reached.on.Time_Y.N`, where `1` represents a shipment that did not arrive on time and `0` represents an on-time shipment.

| Feature | Description |
|---|---|
| `Warehouse_block` | Warehouse category |
| `Mode_of_Shipment` | Shipping mode: Ship, Flight, or Road |
| `Customer_care_calls` | Number of customer-care calls |
| `Customer_rating` | Customer rating from 1 to 5 |
| `Cost_of_the_Product` | Product cost |
| `Prior_purchases` | Number of previous purchases |
| `Product_importance` | Low, Medium, or High |
| `Gender` | Customer gender category |
| `Discount_offered` | Discount offered on the product |
| `Weight_in_gms` | Product weight in grams |
| `Reached.on.Time_Y.N` | Binary target variable |

The notebook also contains an identifier column, which is excluded from model features.

## Tech Stack

- **Language:** Python
- **Data manipulation:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Machine learning:** Scikit-learn
- **Environment:** Jupyter Notebook

## Project Workflow

1. **Data inspection:** Reviewed the dataset structure, column types, missing values, and duplicate records.
2. **Exploratory data analysis:** Examined distributions and relationships between shipment features and delivery outcomes.
3. **Preprocessing:** Removed the identifier from the predictors and encoded categorical features.
4. **Train-test split:** Split the data into training and testing sets using an 80:20 ratio.
5. **Model development:** Trained Logistic Regression, Decision Tree, Random Forest, and KNN classifiers.
6. **Hyperparameter tuning:** Applied GridSearchCV with 5-fold cross-validation to the Decision Tree and Random Forest models.
7. **Evaluation:** Compared accuracy, late-delivery precision, late-delivery recall, F1-score, and confusion matrices.
8. **Business interpretation:** Identified patterns worth investigating and developed recommendations based on the analysis.

## Model Performance

The following results are reported in the current project notebook/README. Precision, recall, and F1-score refer to the late-delivery class.

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Decision Tree | 69% | 0.95 | 0.49 | 0.65 |
| Random Forest | 68% | 0.87 | 0.54 | 0.66 |
| K-Nearest Neighbors | 65% | 0.71 | 0.68 | 0.70 |
| Logistic Regression | 63% | 0.69 | 0.67 | 0.68 |

### How to interpret the results

- The **Decision Tree** has the highest reported accuracy and late-delivery precision.
- **KNN** has the highest reported F1-score and late-delivery recall among these models.
- The Decision Tree's 0.49 recall means it identifies approximately 49% of actual late deliveries in the evaluated test set, based on the reported metric. Its high precision does not mean it catches most late shipments.
- Model selection should depend on the business cost of missed delays versus unnecessary alerts, rather than accuracy alone.

These results are baseline findings, not a guarantee of performance on future shipments.

## Key Analytical Findings

The exploratory analysis reported the following patterns:

- **Product weight:** Delivery outcomes varied across weight ranges, making weight a useful variable to investigate.
- **Product cost:** Product cost showed differences in delivery outcomes across price ranges.
- **Discounts:** Delivery outcomes varied across discount bands.
- **Customer-care calls:** Call volume was associated with shipment characteristics and delivery outcomes, suggesting a possible monitoring signal.
- **Prior purchases:** Repeat-purchase history showed differences in delivery outcomes.
- **Shipping and warehouse features:** The initial analysis reported limited differences in delivery outcomes across these categories.

These are associations observed in the dataset; they do not establish that any feature causes delivery delays. Feature importance and statistical validation would be useful next steps.

## Business Recommendations

1. **Monitor potentially high-risk shipments:** Evaluate whether weight, product cost, and other available order features can help prioritize shipments for review.
2. **Investigate discount-related patterns:** Examine whether differences across discount bands remain after controlling for other shipment characteristics.
3. **Monitor customer-care activity:** Test whether increases in shipment-related inquiries can help identify orders needing proactive support.
4. **Improve delay detection:** Explore class weighting, threshold tuning, and other approaches to increase recall for late deliveries.
5. **Validate operational assumptions:** Investigate warehouse and shipping-mode performance using additional data before making logistics changes.

## How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/NIKIT0501/e-commerce-product-delivery-prediction.git
cd e-commerce-product-delivery-prediction
```

### 2. Install dependencies

```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
```

### 3. Launch Jupyter Notebook

```bash
jupyter notebook E-Commerce_Product_Delivery_Prediction.ipynb
```

Open the notebook and run the cells in order. Keep `E_Commerce.csv` in the expected relative location used by the notebook.

## Limitations and Future Improvements

- **Improve recall:** Investigate class weighting, resampling methods, and decision-threshold tuning.
- **Strengthen preprocessing:** Compare label encoding with one-hot encoding for nominal categorical variables.
- **Prevent data leakage:** Verify that all preprocessing and tuning steps are fitted exclusively on training data and that the final test set remains untouched during model selection.
- **Expand evaluation:** Add ROC-AUC and precision-recall curves, and report class distribution and confusion-matrix counts.
- **Improve validation:** Compare stratified cross-validation and evaluate the model on a later time period if timestamps become available.
- **Explore additional models:** Benchmark gradient-boosting algorithms against the existing baselines.
- **Build a deployment workflow:** Package the selected model in an API and monitor prediction quality and data drift.
- **Connect predictions to business value:** Estimate the costs of missed delays, unnecessary alerts, and potential interventions.

## Repository Structure

```text
e-commerce-product-delivery-prediction/
├── E-Commerce_Product_Delivery_Prediction.ipynb
├── E-Commerce_Product_Delivery_Prediction.pdf
├── E_Commerce.csv
└── README.md
```

## Skills Demonstrated

Python · Pandas · NumPy · Exploratory Data Analysis · Data Visualization · Data Preprocessing · Classification · Scikit-learn · Cross-Validation · Hyperparameter Tuning · Model Evaluation · Business Problem Solving

## Author

**Nikit**

B.Tech — Production and Industrial Engineering  
Motilal Nehru National Institute of Technology (MNNIT) Allahabad

- GitHub: [NIKIT0501](https://github.com/NIKIT0501)
- Project: [E-Commerce Product Delivery Prediction](https://github.com/NIKIT0501/e-commerce-product-delivery-prediction)

---

*This project demonstrates an end-to-end machine learning workflow, from exploratory analysis to model evaluation and business interpretation.*
