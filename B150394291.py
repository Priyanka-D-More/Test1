
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import r2_score 

years = [2005, 2006, 2007, 2008, 2009] 
sales = [12, 19, 29, 37, 45] 

years = np.array(years).reshape(np.array(years).shape[0], 1)
model = LinearRegression()
model.fit(years, sales)
prediction = model.predict(np.array(years).reshape(np.array(years).shape[0], 1))
print('We have generated the least square regression line y=ax+b', prediction)
print('The value of coefficient a is', model.coef_)
print('The value of coefficient b is',model.intercept_)
print(model.predict(np.array(14).reshape(1,1)))
print('The R-squared statistic value using inbuilt classifier is', model.score(years, sales))

#plt.scatter(years, sales, color='b', marker='o', s=30)
#plt.plot(years, sales, color='r')
#plt.plot(years, prediction, color='g')
#plt.show()


