import pandas as pd
from sklearn.tree import DecisionTreeClassifier

data = pd.DataFrame({
    'screen_time': [2,4,6,8,5,7],
    'sleep_hours': [8,7,6,5,6,5],
    'study_hours': [5,4,3,2,3,2],
    'distraction': ['Low','Low','Medium','High','Medium','High']
})

X = data[['screen_time','sleep_hours','study_hours']]
y = data['distraction']

model = DecisionTreeClassifier()
model.fit(X, y)

screen = int(input("Enter screen time (hours): "))
sleep = int(input("Enter sleep hours: "))
study = int(input("Enter study hours: "))


prediction = model.predict([[screen, sleep, study]])

print("Predicted Distraction Level:", prediction[0])