gitfrom sklearn import tree

#features; Market trend (1 for positive, 0 for negative)
#Historical Performance, Market Cap, Recent New (1 for positive, 0 for negative)

features = [
    [1, 1, 1], # high amount of users, good spending, high gains
    [1, 0, 1], # high amount of users, bad spending, high gains
    [0, 1, 0], # low amount of users, good spending, low gains
    [0, 0, 0], # low amount of users, bad spending, low gains
    [0, 1, 1], # low amount of users, good spending, high gains
    [1, 1, 0], # high amount of users, good spending, low gains
    [1, 0, 0], # high amount of users, bad spending, low gains
]

#labels for #1 for invest, 0 for do not invest
labels = [1, 1, 0, 0, 1, 0, 0]

#creating the decision tree classifier
clf = tree.DecisionTreeClassifier()

#Train the classifier with the example data
clf = clf.fit(features, labels)

#example prediction
eg = [[1, 1, 1]]

#Make a prediction!
prediction = clf.predict(eg)

#results
advice = "Invest" if prediction[0] == 1 else "Do not Invest"
print(advice)
