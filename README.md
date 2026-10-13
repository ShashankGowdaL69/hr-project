# Employee Attrition Prediction System

An interactive HR analytics application that analyzes workforce attrition patterns and predicts employee turnover probability using CatBoost. The system combines an employee-level prediction interface with an analytics dashboard to explore workforce trends and identify potential retention risks.

**Live Demo:** [Employee Attrition Prediction System](https://hr-analytics-byhh.onrender.com)

## Features

### 1. Employee Attrition Prediction

- Predicts employee turnover probability using a CatBoost classification model trained on HR data.
- Provides an interactive employee-profile form covering personal details, job characteristics, compensation, and work experience.
- Classifies predictions into **High Risk**, **Moderate Risk**, and **Likely to Stay** categories based on estimated attrition probability.
- Generates rule-based employee-retention recommendations using factors such as job satisfaction, work-life balance, overtime, income, and commute distance.
- Uses class-weighted learning to account for the imbalance between employees who leave and those who stay.

### 2. HR Analytics Dashboard

- Displays workforce KPIs, including total employees, attrition count, attrition rate, and average monthly income.
- Supports department-wise filtering for focused analysis.
- Provides interactive Plotly visualizations for:
  - Attrition by department
  - Monthly income distribution by attrition outcome
  - Job satisfaction and attrition
  - Work-life balance and attrition

### 3. Data Management

- Retrieves employee records from MongoDB Atlas.
- Uses Pandas for data processing and feature preparation.
- Leverages CatBoost's categorical-feature handling for HR attributes.
- Implements Streamlit caching to reduce redundant data loading and model training during application reruns.

## Tech Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Machine Learning | CatBoost |
| Data Processing | Pandas |
| Application Framework | Streamlit |
| Data Visualization | Plotly Express |
| Database | MongoDB Atlas |
| Deployment | Render |

## Machine Learning Model

The application uses a `CatBoostClassifier` with the following configuration:

| Parameter | Value |
|---|---|
| Iterations | 300 |
| Learning Rate | 0.03 |
| Tree Depth | 5 |
| L2 Regularization | 10 |
| Loss Function | Logloss |
| Class Weights | Balanced |
| Random Seed | 42 |

The model predicts the `Attrition` target, encoded as `1` for employees who left and `0` for employees who stayed. Categorical features are passed to CatBoost through its `cat_features` interface.

The dataset contains substantially more employees who stayed than employees who left. The `Balanced` class-weighting configuration helps the model account for this class imbalance during training.

### Model Evaluation

During offline model experimentation, the selected CatBoost configuration achieved the following results on a held-out test set:

| Metric | Score |
|---|---:|
| Accuracy | 82.99% |
| Precision | 47.62% |
| Recall | 63.83% |
| F1-Score | 54.55% |
| ROC-AUC | 80.60% |

These metrics represent the offline model-selection experiment. The deployed application trains the model on employee records loaded from MongoDB Atlas; it does not calculate or display these held-out test metrics as live evaluation results.

## Dataset and Database

The project uses the **IBM HR Analytics Employee Attrition & Performance** dataset.

Employee records are stored in MongoDB Atlas:

- **Database:** `hr_database`
- **Collection:** `employee_data`

The database collection must contain the employee dataset for the application to function.

## Getting Started

### Prerequisites

- Python 3.10 or a compatible Python version
- MongoDB Atlas account and connection string
- Git

### Installation

**1. Clone the repository**

```bash
git clone <YOUR_REPOSITORY_URL>
cd <YOUR_REPOSITORY_DIRECTORY>
```

**2. Create and activate a virtual environment**

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Configure MongoDB**

Set the `MONGO_URI` environment variable to your MongoDB Atlas connection string.

macOS / Linux:

```bash
export MONGO_URI="your-mongodb-connection-string"
```

Windows PowerShell:

```powershell
$env:MONGO_URI="your-mongodb-connection-string"
```

Ensure that the MongoDB user has read access to `hr_database.employee_data` and that MongoDB Atlas network-access settings permit the connection.

**5. Run the application**

```bash
streamlit run app.py
```

Open the local URL provided by Streamlit, typically `http://localhost:8501`.

## Deployment

The application is deployed on Render using Streamlit.

The Streamlit start command is:

```bash
streamlit run app.py --server.address 0.0.0.0 --server.port $PORT
```

The deployed service uses the `MONGO_URI` environment variable to connect to MongoDB Atlas. Database credentials are kept outside the source code to avoid exposing sensitive information.

## Project Structure

```text
.
├── app.py
├── requirements.txt
├── Dockerfile
└── README.md
```

The production application uses the final CatBoost implementation in `app.py`. Offline experimentation scripts, if retained in the repository, are separate from the production prediction pipeline.

## Limitations

- Predictions depend on the quality and representativeness of the training data.
- Risk categories use application-defined probability thresholds:
  - **High Risk:** 65% and above
  - **Moderate Risk:** 40% to below 65%
  - **Likely to Stay:** Below 40%
- Some prediction-form fields use fixed default values rather than user-provided inputs, which may influence predictions.
- Retention recommendations are rule-based and are not generated by the machine learning model.
- Relationships observed in the dashboard do not establish causation.
- The model is intended for analytics and demonstration purposes and should not be used as the sole basis for consequential employment decisions.

## Future Enhancements

- Integrate model explainability to identify factors influencing individual predictions.
- Implement automated data validation and schema checks.
- Monitor model performance and data drift over time.
- Validate probability calibration and risk thresholds.
- Add authentication and role-based access controls for applications handling real employee data.