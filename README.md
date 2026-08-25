# 🚢 Titanic Survival Prediction — Logistic Regression

A Machine Learning classification project that predicts whether a passenger survived the **Titanic disaster** using **Logistic Regression**.

The project covers a complete beginner-to-intermediate supervised learning workflow:

```text
Titanic Dataset
      ↓
Exploratory Data Analysis
      ↓
Data Cleaning
      ↓
Feature Selection
      ↓
Categorical Encoding
      ↓
Train/Test Split
      ↓
Feature Scaling
      ↓
Logistic Regression
      ↓
Model Evaluation
      ↓
Interactive Prediction
```

The project is designed to demonstrate how real-world tabular data can be cleaned, transformed, used to train a classification model, and then reused to make predictions for new passengers.

---

# 🎯 Project Objective

The objective of this project is to predict whether a Titanic passenger would have **survived or not survived** based on information about the passenger and their journey.

The Machine Learning problem is formulated as a **binary classification task**.

The model predicts:

```text
0 → Did Not Survive
1 → Survived
```

The target variable is:

```text
Survived
```

The prediction process can be represented as:

```text
Passenger Information
        ↓
Logistic Regression
        ↓
Survival Probability
        ↓
Survived / Did Not Survive
```

---

# 🧠 Why Logistic Regression?

**Logistic Regression** is a fundamental classification algorithm that estimates the probability of a sample belonging to a particular class.

For this project, it is used to estimate:

```text
P(Survived = 1 | Passenger Features)
```

The model produces both:

* A predicted class
* A survival probability

For example:

```text
Prediction:
Passenger likely SURVIVED

Survival Chance:
82.4%
```

This makes Logistic Regression particularly useful for demonstrating probabilistic binary classification.

---

# 📊 Dataset

The project uses:

```text
Titanic-Dataset.csv
```

which is stored directly in the repository.

The repository also contains an exploratory analysis notebook:

```text
EDA.ipynb
```

for inspecting and understanding the dataset before model development.

---

# 🧾 Dataset Features

The original Titanic dataset contains information about passengers, their class, demographics, family relationships, fare, and port of embarkation.

The model ultimately uses the following features:

| Feature      | Description                                       |
| ------------ | ------------------------------------------------- |
| `Pclass`     | Passenger class                                   |
| `Sex`        | Passenger sex                                     |
| `Age`        | Passenger age                                     |
| `SibSp`      | Number of siblings/spouses aboard                 |
| `Parch`      | Number of parents/children aboard                 |
| `Fare`       | Passenger fare                                    |
| `Embarked_C` | One-hot encoded Cherbourg embarkation indicator   |
| `Embarked_Q` | One-hot encoded Queenstown embarkation indicator  |
| `Embarked_S` | One-hot encoded Southampton embarkation indicator |

The target variable is:

```text id="y2j0t2"
Survived
```

The prediction script constructs exactly these features before passing them into the trained scaler and model.

---

# 🔎 Exploratory Data Analysis

The repository includes:

```text
EDA.ipynb
```

which is used to explore the dataset before training.

The analysis includes inspecting:

* Dataset structure
* Dataset dimensions
* Column names
* Data types
* Descriptive statistics
* Passenger information
* Potential missing values
* Features relevant to survival prediction

This exploratory stage helps determine which variables are useful for the classification problem.

---

# 🧹 Data Cleaning

Before training the model, several preprocessing steps are performed.

## Removing Unnecessary Columns

The following columns are removed:

```text
PassengerId
Name
Ticket
Cabin
```

The project treats these columns as unsuitable for the selected prediction workflow and removes them before modeling.

After removal, the model focuses on structured demographic, travel, and family-related variables.

---

# 🧽 Missing Value Handling

The implementation removes rows containing missing values:

```python
data = data.dropna()
```

This is performed before the dataset is split and used for model training.

Removing incomplete rows provides a simple approach to handling missing data for this educational project.

For a more advanced implementation, future versions could use feature-specific imputation instead of removing observations.

---

# ♻️ Duplicate Removal

The implementation also removes duplicate records:

```python
data = data.drop_duplicates()
```

This prevents identical rows from being retained multiple times in the processed dataset.

---

# 🔤 Categorical Encoding

The Titanic dataset contains categorical variables that cannot be directly processed by the Logistic Regression model.

Two encoding strategies are used.

---

## `Embarked` — One-Hot Encoding

The `Embarked` feature represents the passenger's port of embarkation.

It is transformed using:

```python
OneHotEncoder(
    sparse_output=False,
    handle_unknown='ignore'
)
```

This produces separate indicator variables:

```text
Embarked_C
Embarked_Q
Embarked_S
```

Conceptually:

```text
Embarked
   │
   ├── C → Embarked_C
   ├── Q → Embarked_Q
   └── S → Embarked_S
```

The original `Embarked` column is then removed.

---

## `Sex` — Label Encoding

The passenger sex feature is transformed numerically using `LabelEncoder`.

The resulting encoded value is then used as a model input.

Conceptually:

```text
Sex
 │
 ├── male
 └── female
      ↓
Numerical Representation
```

This allows the classifier to process the categorical variable.

---

# ✂️ Train/Test Split

The cleaned and encoded dataset is split using:

```python
train_test_split(
    data,
    test_size=0.2,
    random_state=42
)
```

This creates an **80/20 train-test split**.

```text
Processed Dataset
       │
       ├── 80% → Training Data
       │
       └── 20% → Testing Data
```

The model learns from the training partition and is evaluated on the held-out testing partition.

---

# 📏 Feature Scaling

The input features are standardized with:

```python
StandardScaler()
```

The scaler is fitted using the training data:

```python
scaled_X_train = scaler.fit_transform(X_train)
```

and then applied to the test data:

```python
scaled_X_test = scaler.transform(X_test)
```

This ensures that the model sees training and testing features on the same scale.

It is especially useful for Logistic Regression when input variables have very different numerical ranges.

---

# 🤖 Logistic Regression

The project uses:

```python
LogisticRegression()
```

as its classification model.

The model is trained using the scaled training features:

```python
model.fit(
    scaled_X_train,
    y_train
)
```

It then generates predictions for the test dataset:

```python
predictions = model.predict(scaled_X_test)
```

The model can also estimate class probabilities for new passengers using:

```python
model.predict_proba(...)
```

which is used by the interactive predictor.

---

# 📈 Model Evaluation

The project evaluates the classifier using several standard classification metrics.

These include:

* Confusion Matrix
* Accuracy
* Precision
* Recall
* F1 Score

---

# 🔲 Confusion Matrix

A binary confusion matrix separates predictions into four categories:

```text
                    Predicted
                  No Survival  Survival
                ┌────────────┬─────────┐
Actual           │            │         │
No Survival     │     TN     │   FP    │
                ├────────────┼─────────┤
Survival        │     FN     │   TP    │
                └────────────┴─────────┘
```

Where:

* **True Negative (TN):** Correctly predicted non-survivor
* **True Positive (TP):** Correctly predicted survivor
* **False Positive (FP):** Predicted survival when the passenger did not survive
* **False Negative (FN):** Predicted non-survival when the passenger survived

The project calculates the confusion matrix using:

```python
skm.confusion_matrix(
    y_test,
    predictions
)
```

---

# 🎯 Accuracy

Accuracy measures the proportion of predictions that were correct overall.

```text
Accuracy =
Correct Predictions
-------------------
Total Predictions
```

It provides a useful general performance measure for the classifier.

---

# 🎯 Precision

Precision answers:

> Of the passengers predicted to have survived, how many actually survived?

```text
Precision =
True Positives
-------------------------
True Positives + False Positives
```

---

# 🔍 Recall

Recall answers:

> Of the passengers who actually survived, how many did the model correctly identify?

```text
Recall =
True Positives
-------------------------
True Positives + False Negatives
```

Recall is especially useful when evaluating how effectively the classifier identifies the positive survival class.

---

# ⚖️ F1 Score

The F1 score combines precision and recall:

```text
F1 =
2 × Precision × Recall
----------------------
Precision + Recall
```

This provides a balanced metric when both false positives and false negatives matter.

---

# 💻 Interactive Survival Predictor

One of the main features of the repository is the interactive prediction interface inside:

```text
titanic_survival_predictor.py
```

After training and evaluating the model, the script starts a command-line prediction loop.

The user is asked to enter:

```text
Passenger class
Passenger sex
Passenger age
Number of siblings/spouses aboard
Number of parents/children aboard
Passenger fare
Port of embarkation
```

The entered information is then transformed using the exact preprocessing structure expected by the model.

---

# 🚢 Prediction Workflow

The interactive predictor follows this pipeline:

```text
User Input
    ↓
Validate Input
    ↓
Encode Sex
    ↓
Encode Embarkation
    ↓
Build Input DataFrame
    ↓
Apply StandardScaler
    ↓
Logistic Regression
    ↓
Prediction + Probability
```

The prediction script deliberately constructs a DataFrame with the exact feature names expected by the scaler:

```text
Pclass
Sex
Age
SibSp
Parch
Fare
Embarked_C
Embarked_Q
Embarked_S
```

This reduces the chance of feature-order mismatches during inference.

---

# 🎲 Example Prediction

A user might enter:

```text
Passenger class (1, 2, or 3): 1
Passenger sex (male/female): female
Passenger age: 25
Number of siblings/spouses aboard: 0
Number of parents/children aboard: 0
Passenger fare: 80
Port of embarkation (C, Q, or S): S
```

The model then produces something like:

```text
Prediction:
Passenger likely SURVIVED

Survival Chance:
XX.XX%
```

The exact probability depends on the trained model and input values.

---

# 🔄 Complete Machine Learning Pipeline

```text
                     Titanic Dataset
                           │
                           ▼
                  Exploratory Analysis
                           │
                           ▼
                  Remove Unused Columns
                           │
                           ▼
                Handle Missing Values
                           │
                           ▼
                  Remove Duplicates
                           │
                           ▼
                  Encode Categorical Data
                           │
                ┌──────────┴──────────┐
                │                     │
                ▼                     ▼
        Label Encode Sex       One-Hot Embarkation
                │                     │
                └──────────┬──────────┘
                           ▼
                     Train/Test Split
                           │
                           ▼
                     Feature Scaling
                           │
                           ▼
                  Logistic Regression
                           │
                           ▼
                       Predictions
                           │
                           ▼
            ┌──────────────┼──────────────┐
            │              │              │
            ▼              ▼              ▼
        Accuracy       Precision        Recall
            │              │              │
            └──────────────┼──────────────┘
                           ▼
                         F1 Score
                           │
                           ▼
                 Interactive Prediction
```

---

# 📁 Project Structure

```text
Titanic-Survival-Classification-Logistic-Regression/
│
├── Titanic-Dataset.csv
│
├── EDA.ipynb
│
├── titanic_survival_predictor.py
│
├── LICENSE
│
└── README.md
```

| File                            | Description                                                      |
| ------------------------------- | ---------------------------------------------------------------- |
| `Titanic-Dataset.csv`           | Titanic passenger dataset                                        |
| `EDA.ipynb`                     | Exploratory Data Analysis notebook                               |
| `titanic_survival_predictor.py` | Complete training, evaluation, and interactive prediction script |
| `LICENSE`                       | Project license                                                  |
| `README.md`                     | Project documentation                                            |

These are the files currently present in the repository's `main` branch.

---

# 🛠️ Technologies Used

* **Python**
* **Pandas**
* **NumPy**
* **Scikit-Learn**
* **Jupyter Notebook**

Scikit-Learn components used include:

```text
LogisticRegression
OneHotEncoder
LabelEncoder
StandardScaler
train_test_split
confusion_matrix
accuracy_score
precision_score
recall_score
f1_score
```

The main prediction script uses these components directly.

---

# ⚙️ Installation

## 1. Clone the Repository

```bash
git clone https://github.com/syedsyedhady13-alt/Titanic-Survival-Classification-Logistic-Regression.git
```

Move into the project directory:

```bash
cd Titanic-Survival-Classification-Logistic-Regression
```

---

# 🐍 Create a Virtual Environment

### Windows

```bash
python -m venv venv
```

Activate:

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv
```

Activate:

```bash
source venv/bin/activate
```

---

# 📦 Install Dependencies

```bash
pip install numpy pandas scikit-learn jupyter
```

---

# 🚀 Running the Project

## Explore the Dataset

Launch Jupyter:

```bash
jupyter notebook
```

Then open:

```text
EDA.ipynb
```

This notebook can be used to explore the Titanic dataset before running the classifier.

---

## Run the Classifier

Run:

```bash
python titanic_survival_predictor.py
```

The script will:

1. Load the dataset.
2. Remove unused columns.
3. Handle missing values.
4. Remove duplicate records.
5. Encode categorical variables.
6. Split the dataset.
7. Scale the features.
8. Train Logistic Regression.
9. Evaluate the model.
10. Start the interactive prediction interface.

---

# ⚠️ Portability Note

The current Python script loads the dataset using a **machine-specific Windows path**:

```python
C:\Users\Syed\Documents\...
```

That path will not exist on another computer after cloning the repository.

For a more portable project, the dataset should be loaded relative to the script:

```python
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
data = pd.read_csv(BASE_DIR / "Titanic-Dataset.csv")
```

This would allow the repository to run without modifying local file paths.

---

# 📚 Machine Learning Concepts Demonstrated

This project demonstrates:

### Data Analysis

* Exploratory Data Analysis
* Dataset inspection
* Descriptive statistics
* Feature selection

### Data Cleaning

* Missing-value handling
* Duplicate removal
* Removing irrelevant columns

### Feature Engineering

* Label encoding
* One-hot encoding
* Feature scaling

### Classification

* Binary classification
* Logistic Regression
* Probability prediction

### Model Evaluation

* Confusion Matrix
* Accuracy
* Precision
* Recall
* F1 Score

### Model Inference

* Processing new input
* Reusing preprocessing
* Predicting class probability
* Interactive command-line prediction

---

# 💡 Why the Titanic Dataset Is Useful

The Titanic dataset is a useful introduction to Machine Learning because it contains a mixture of:

```text
Numerical Features
       +
Categorical Features
       +
Missing Values
       +
Target Variable
       ↓
Classification Problem
```

This makes it more representative of practical tabular Machine Learning than a perfectly clean dataset.

It requires the model pipeline to deal with preprocessing before the algorithm can be trained.

---

# 🔬 Feature Engineering Lessons

The project demonstrates that Machine Learning models generally cannot consume raw categorical text directly.

For example:

```text
Sex = female
```

must be represented numerically.

Similarly:

```text
Embarked = C / Q / S
```

must be transformed into numerical features.

This creates an important general Machine Learning workflow:

```text
Raw Data
   ↓
Clean Data
   ↓
Encode Data
   ↓
Scale Data
   ↓
Model
```

## Better Missing-Value Handling

Instead of dropping incomplete rows, use techniques such as:

* Mean/median imputation
* Most-frequent categorical imputation
* Scikit-Learn `SimpleImputer`

This can preserve more of the available data.

---

## Feature Engineering

Potential additional features include:

### Family Size

```text
FamilySize = SibSp + Parch + 1
```

### Is Alone

```text
IsAlone = FamilySize == 1
```

### Age Groups

Create meaningful age categories for analysis.

### Fare Per Person

Adjust fare based on family size or ticket grouping.

These features could potentially improve predictive performance and provide richer insights.

---

# 📊 Model Comparison Dashboard

A future version could compare models using:

```text
Accuracy
Precision
Recall
F1
ROC-AUC
```

and display the results in a simple comparison table or visualization.

---

# 🌐 Interactive Web Application

The existing command-line predictor could be transformed into a **Streamlit application**.

Possible interface:

```text
┌─────────────────────────────────┐
│ Titanic Survival Predictor      │
├─────────────────────────────────┤
│ Passenger Class     [ 1 ▼ ]     │
│ Sex                 [ Female ]  │
│ Age                 [ 25.0 ]    │
│ Siblings/Spouses    [ 0 ]       │
│ Parents/Children    [ 0 ]       │
│ Fare                [ 80.0 ]    │
│ Embarkation         [ S ▼ ]     │
│                                 │
│        [ Predict ]              │
└─────────────────────────────────┘
```

with an output such as:

```text
🚢 Likely Survived

Survival Probability: 82.4%
```

---

# 🧠 Important Interpretation Note

A Machine Learning prediction is not a historical fact about an individual passenger.

The model learns statistical patterns from the dataset.

Therefore:

```text
Model Prediction
      ≠
Historical Certainty
```

The model should be understood as a classification exercise rather than a mechanism for determining what actually would have happened to a specific passenger.

---

# 📌 Key Learning Outcome

This project demonstrates a very practical supervised-learning workflow:

```text
Raw Dataset
     ↓
Understand the Data
     ↓
Clean the Data
     ↓
Select Features
     ↓
Encode Categories
     ↓
Scale Features
     ↓
Split Dataset
     ↓
Train Classifier
     ↓
Evaluate Performance
     ↓
Deploy Predictions
```

This workflow is reusable far beyond the Titanic dataset and forms a foundation for more advanced classification projects.

---

# 👨‍💻 Author

**Syed Abdul Hadi**

Aspiring Machine Learning Engineer

Building practical projects to develop skills in:

* Python
* Data Science
* Machine Learning
* Artificial Intelligence

---

# ⭐ Support

If you found this project useful, consider giving the repository a ⭐ on GitHub.

---

## 📄 License

This project includes a `LICENSE` file in the repository.
