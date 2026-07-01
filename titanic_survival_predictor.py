from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sklearn.metrics as skm
import numpy as np
import pandas as pd

print("step 1:understand the dataset")
data=pd.read_csv(r"C:\Users\Syed\Documents\abdul hadi\python\ai\projects\titanic survival predictor\Titanic-Dataset.csv")

print("--Dataset--")
print(data.head(50))
print("")
print("--Dataset Info--")
print(data.info())
print("")
print("No of columns: ", len(data.columns))
print("No of rows:    ", len(data))
print("")
print("--Dataset Description--")
print(data.describe())
print("")

print("""step 1 : exploring the dataset
dataset is clean with no missing values — ready to use""")

print("")
print("step 2:formating dataset")
print("beginning the step for formation of dataset")
print("")
print("now filtering unnessesary columns")
print("all of columns:",end="")
no_of_cols=0
for i in data.columns:
    no_of_cols+=1
    if i != data.columns[11]:
        print(" ",i,",",end="")
    else:
        print(i)
print("number of columns:",no_of_cols)

data=data.drop(columns=['PassengerId','Name','Ticket','Cabin'])

print("number of dropped cols: 4")
print("number of columns after dropping:",len(data.columns))
print("")
print("starting to remove duplicte records and NaN value records")
print("number of NaN Value Records:",data.dropna().shape[0])
print("number of duplicate records",data.shape[0]-data.drop_duplicates().shape[0])
data=data.dropna()
data=data.drop_duplicates()
print("Successfully removed")
print("")
print("step 3:encoding the dataset")
print("encoding columns...")

ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
ohe.set_output(transform='pandas')
data = pd.concat([data, ohe.fit_transform(data[['Embarked']])], axis=1)
data.drop('Embarked', axis=1, inplace=True)
print("encoding completed")
print("here is what the dataset looks like after encoding")
print(data.head(5))
print("")

le = LabelEncoder()
data['Sex'] = le.fit_transform(data['Sex'])

print("all is ready to go for spliting the model")
data_train, data_test = train_test_split(data, test_size=0.2, random_state=42)
print("training rows : ", len(data_train))
print("testing rows  : ", len(data_test))
print("")

print("step 4:scaling the dataset")
print("")
X_train = data_train.drop('Survived', axis=1)
X_test  = data_test.drop('Survived', axis=1)
y_train = data_train['Survived']
y_test  = data_test['Survived']

scaler=StandardScaler()
scaled_X_train=scaler.fit_transform(X_train)
scaled_X_test=scaler.transform(X_test)
print("")
print("step 5 : training the logistic regression model")

model = LogisticRegression()
model.fit(scaled_X_train, y_train)
print("model has been trained")
print("")
predictions = model.predict(scaled_X_test)
print("step 6 : evaluating the model")
print("")
print("the confusion matrix is:")
print(skm.confusion_matrix(y_test, predictions))
print("")
print("the accuracy score is:",skm.accuracy_score(y_test, predictions))
print("the precision score is:",skm.precision_score(y_test, predictions))
print("the recall score is:",skm.recall_score(y_test, predictions))
print("the f1 score is:",skm.f1_score(y_test, predictions))
print("")
print("\nModel is ready for deployment loop!")

while True:
    try:
        pclass = int(input("\nPassenger class (1, 2, or 3): "))
        sex = input("Passenger sex (male/female): ").strip().lower()
        age = float(input("Passenger age: "))
        sibsp = int(input("Number of siblings/spouses aboard: "))
        parch = int(input("Number of parents/children aboard: "))
        fare = float(input("Passenger fare: "))
        embarked = input("Port of embarkation (C, Q, or S): ").strip().upper()
    except ValueError:
        print("Invalid input type. Please restart loop.")
        continue
    
    # Process inputs exactly how the model expects them
    sex_encoded = 1 if sex == 'male' else 0
    emb_c = 1.0 if embarked == 'C' else 0.0
    emb_q = 1.0 if embarked == 'Q' else 0.0
    emb_s = 1.0 if embarked == 'S' else 0.0

    # Build DataFrame to guarantee perfect alignment with scaler/model feature names
    input_df = pd.DataFrame([{
        'Pclass': pclass, 'Sex': sex_encoded, 'Age': age, 
        'SibSp': sibsp, 'Parch': parch, 'Fare': fare,
        'Embarked_C': emb_c, 'Embarked_Q': emb_q, 'Embarked_S': emb_s
    }])
    
    # Scale and predict
    scaled_input = scaler.transform(input_df)
    predicted_survival = model.predict(scaled_input)[0]
    probabilities = model.predict_proba(scaled_input)[0]

    status = "SURVIVED" if predicted_survival == 1 else "DIED"
    print(f"\nPrediction: Passenger likely {status} (Survival Chance: {probabilities[1]*100:.2f}%)")

    again = input("\nPredict another passenger? (yes/no): ").strip().lower()
    if again != 'yes':
        print("Goodbye!")
        break
