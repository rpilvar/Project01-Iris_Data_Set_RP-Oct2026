# %%
# Hello! This is my Python script to run a classifier model on the Iris dataset. 


# This block is to check the versions of Python, NumPy, Pandas, and Scikit-learn installed in the environment.
import numpy as np

numbers = np.array([10, 20, 30, 40, 50])

print("Mean:", np.mean(numbers))
print("Maximum:", np.max(numbers))
print("Minimum:", np.min(numbers))

# %%
# This block is to check the versions of Python, NumPy, Pandas, and Scikit-learn installed in the environment.
import sys

print(sys.executable)

# %%
# This block is to check the versions of Python, NumPy, Pandas, and Scikit-learn installed in the environment.
import sys

print("Python:", sys.version)
print("Python location:", sys.executable)

try:
    import numpy
    print("NumPy:", numpy.__version__)
except Exception as e:
    print("NumPy ERROR:", e)

try:
    import pandas
    print("Pandas:", pandas.__version__)
except Exception as e:
    print("Pandas ERROR:", e)

try:
    import sklearn
    print("Scikit-learn:", sklearn.__version__)
except Exception as e:
    print("Scikit-learn ERROR:", e)

# %%
# This block is to implrt Pandas and create a simple DataFrame to test if Pandas is working correctly.
import pandas as pd

data = {
    "Name": ["John", "Sarah", "David"],
    "Age": [25, 31, 28],
    "Salary": [35000, 45000, 40000]
}

df = pd.DataFrame(data)

df

# %%
#This block is to test Scikit-learn by creating a simple linear regression model and making a prediction.
from sklearn.linear_model import LinearRegression #test Scikit-learn
import numpy as np

X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 6, 8, 10])

model = LinearRegression()

model.fit(X, y)

prediction = model.predict([[6]])

print("Prediction:", prediction)

# %%
#this block is to test Scikit-learn by loading the Iris dataset and printing its features and target names.
from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data # shape (150,4)
y = iris.target # shape (150,)
print(iris.feature_names,iris.target_names) 


# %%
# This is just an additional check

import pandas as pd

df = pd.DataFrame(iris.data, columns=iris.feature_names)
print(df.head(4))

# %%
# This is just an additional check
df.head(4)
df.head()

# %%
df.head()

# %%
# coursework model development
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)



# %%
from sklearn.tree import DecisionTreeClassifier
model=DecisionTreeClassifier(random_state=42)
model.fit(X_train,y_train)


# %%
y_pred=model.predict(X_test)

# %%
print("Predictions:",y_pred[:5])
print("True labels:",y_test[:5])

# %%
# evaluate model accuracy
from sklearn.metrics import accuracy_score
accuracy = accuracy_score (y_test, y_pred)
print ("Accuracy:",accuracy)

# %%
#creating a confusion matrix for our trained model
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

# %%
# End of my small project!
# Thank you for you for your attention to my basic code!


