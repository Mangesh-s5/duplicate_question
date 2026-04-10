import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


df = pd.read_csv("placement.csv")

print("Dataset:")
print(df)


X = df[['cgpa', 'aptitude', 'communication']]
y = df['placed']



X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


model = LogisticRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
print("\nModel Accuracy:", accuracy_score(y_test, y_pred))


print("\nEnter Student Details:")

cgpa = float(input("Enter CGPA: "))
aptitude = int(input("Enter Aptitude Score: "))
communication = int(input("Enter Communication Score: "))

new_student = [[cgpa, aptitude, communication]]

prediction = model.predict(new_student)


if prediction[0] == 1:
    print("✅ Student will be PLACED")
else:
    print("❌ Student will NOT be placed")