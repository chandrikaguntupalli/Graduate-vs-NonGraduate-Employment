# **An Analytical Study of Graduates vs Non-Graduates to Provide Greater Employment Prospects Using Machine Learning**

## **Project Overview**  
This project analyzes employment outcomes for graduates and non-graduates across 159 academic disciplines. Using machine learning techniques, we identify key predictors of post-graduate success and highlight fields at higher risk of unemployment. The study employs statistical analysis, correlation studies, and logistic regression modeling to provide insights that can help students, educators, and policymakers make informed decisions about education and career planning.

## **Key Findings**  
- STEM, Business, and Healthcare fields exhibit lower unemployment rates.  
- Humanities and Arts disciplines show a higher likelihood of employment instability.  
- Our **logistic regression model** achieved **73.08% accuracy** and an **AUC score of 0.80**, effectively predicting unemployment risks.  
- The analysis highlights the importance of **aligning educational programs with labor market demands**.  

## **Dataset**  
The dataset includes employment and salary metrics for both graduates and non-graduates, such as:  
- **Employment Rates**  
- **Unemployment Statistics**  
- **Full-time vs. Part-time Employment**  
- **Median Salaries**  
- **Employment Distribution by Major Category**  

## **Methodology**  

### **1. Data Collection & Preprocessing**  
- Data cleaning, handling missing values, and feature engineering.  
- Normalization and outlier detection using the **Interquartile Range (IQR) method**.  

### **2. Exploratory Data Analysis (EDA)**  
- Distribution of graduates vs. non-graduates across various disciplines.  
- Correlation between unemployment rates and median salaries.  
- Comparative analysis of employment stability.  

### **3. Machine Learning Modeling**  
- **Logistic Regression Model** to predict unemployment risks.  
- **Confusion Matrix & ROC Curve Analysis** to assess model performance.  

## **Results**  
- Graduates consistently show **lower unemployment rates** than non-graduates.  
- Certain fields, such as **Education and Healthcare**, provide stable employment opportunities regardless of degree level.  
- **Higher-paying majors** correlate with **lower unemployment risks**.  
- The model's **AUC score of 0.80** confirms its strong predictive capability.  

## **How to Use This Repository**  

### **1. Clone the Repository**  
```sh
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### **2. Install Dependencies**  
Ensure you have Python and the necessary libraries installed:  
```sh
pip install -r requirements.txt
```

### **3. Run the Analysis**  
Execute the script to analyze the dataset:  
```sh
python analysis.py
```

### **4. View Results**  
Check the generated visualizations and model evaluation metrics in the output folder.  

