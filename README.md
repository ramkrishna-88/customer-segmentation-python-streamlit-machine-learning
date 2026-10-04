# 🛍️ Customer Segmentation using Clustering

A complete Machine Learning capstone project based on **K-Means clustering** and the **Mall Customers Dataset**.

## Project Objective

Segment customers into meaningful groups using:

- Age
- Annual Income
- Spending Score

The project labels the resulting groups as:

- Low Spenders
- Medium Spenders
- High Spenders

## Assignment Alignment

The Week 4 brief specifies:

1. Apply K-Means to cluster customers.
2. Visualize customer groups using Age, Income and Spending Score.
3. Label groups as Low Spenders, Medium Spenders and High Spenders.
4. The outcome is a business-analytics-oriented project.

This project implements all four requirements.

## Project Structure

```text
customer_segmentation_clustering/
│
├── app.py
├── customer_segmentation.py
├── Mall_Customers.csv
├── requirements.txt
├── README.md
├── PROJECT_REPORT.md
└── .gitignore
```

## How to Run

### 1. Create a virtual environment

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Start the Streamlit dashboard

```bash
streamlit run app.py
```

Then open the local Streamlit URL shown in the terminal.

## Dataset

The assignment names the **Kaggle – Mall Customers Dataset**.

The repository includes a demo CSV with the expected schema so that the application runs immediately. For the assignment submission, you can replace it with the official Kaggle Mall Customers CSV.

Expected columns:

```text
CustomerID
Gender
Age
Annual Income (k$)
Spending Score (1-100)
```

The dashboard also supports uploading a CSV from the sidebar.

## Machine Learning Workflow

```text
Dataset
   ↓
Data Cleaning
   ↓
Select Age + Income + Spending Score
   ↓
StandardScaler
   ↓
Elbow Method + Silhouette Score
   ↓
K-Means Clustering
   ↓
Cluster Profiles
   ↓
Spending-Based Segment Labels
   ↓
Visualizations
   ↓
Business Interpretation
```

## Why K-Means?

K-Means is an unsupervised learning algorithm that partitions observations into K groups by minimizing the distance between observations and their assigned cluster centroid.

The project uses:

```python
KMeans(n_clusters=3, random_state=42, n_init=10)
```

The application allows K to be changed for experimentation.

## Why Scaling?

Age, income and spending score are measured on different numeric ranges. StandardScaler transforms the features so that their scales do not unfairly dominate the clustering distance.

## Evaluation for Clustering

Since this is unsupervised learning, classification metrics such as accuracy, precision and recall are not the primary evaluation measures.

This project includes:

- Elbow Method
- Silhouette Score
- Cluster profile comparison
- 2D and 3D visualizations
- PCA visualization

## Business Interpretation

### Low Spenders
Customers with relatively low average spending scores.

Possible business actions:
- introductory promotions
- product discovery campaigns
- retention offers

### Medium Spenders
Customers with moderate spending behavior.

Possible business actions:
- bundles
- cross-selling
- personalized offers

### High Spenders
Customers with relatively high spending scores.

Possible business actions:
- loyalty programs
- personalized premium offers
- high-value customer engagement

These are descriptive business interpretations, not guaranteed predictions of future customer behavior.

## Outputs

The Streamlit dashboard provides:

- dataset overview
- cleaned data table
- age distribution
- income vs spending visualization
- age vs spending visualization
- elbow curve
- silhouette score curve
- customer segment counts
- cluster profile table
- interactive 3D segmentation plot
- PCA visualization
- downloadable segmented CSV

## GitHub

Suggested repository name:

```text
customer-segmentation-clustering
```

Suggested description:

```text
Customer Segmentation using K-Means clustering with Streamlit dashboard, EDA, cluster evaluation, visualizations, and business insights.
```
