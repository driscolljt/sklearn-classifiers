from sklearn import tree
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.neighbors import KNeighborsClassifier

# [height, weight, shoe size]
x = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37],
    [166, 65, 40], [190, 90, 47], [175, 64, 39], [177, 70, 40], [159, 55, 37],
    [171, 75, 42], [181, 85, 43]]

y = ['male', 'female', 'female', 'female', 'male', 'male',
     'male', 'female', 'male', 'female', 'male']

treeCLF = tree.DecisionTreeClassifier()

treeCLF = treeCLF.fit(x,y)

treePredict = treeCLF.predict([[190,70,43]])

print('Decision Tree Classifier')
print(treePredict)
print('-------------------------')

pac = PassiveAggressiveClassifier(random_state=0)
pac.fit(x,y)

pacPredict = pac.predict([[190,70,43]])

print('Passive Aggressive Classifier')
print(pacPredict)
print('-------------------------')

nc = KNeighborsClassifier()
nc.fit(x,y)

ncPredict = nc.predict([[190,70,43]])

print('Neighbor Classifier')
print(ncPredict)
print('-------------------------')