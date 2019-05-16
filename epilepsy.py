#In this code we have tested various machine learning models to predict number of emergency visits as a regression
#problem. We have used 5 fold cross validation for each model. Best parameters have been subsequently chosen
#from the final fit, which is also used to find mean-squared-error and root-mean-squared-error


import sklearn
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import Normalizer
from sklearn.linear_model import Ridge, RidgeCV, Lasso, LassoCV, ElasticNetCV, ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
#from sklearn.grid_search import RandomizedSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from sklearn.cross_validation import  cross_val_score
from sklearn.metrics import make_scorer
from sklearn.ensemble import AdaBoostRegressor
from sklearn import metrics


def model_fit(model, xtrain=xtri, ytrain=ytrain, xtest=xtest_nm, ytest=ytest_nm):
    m = model.fit(xtrain, ytrain)
    mse = (ytest - m.predict(xtest))  # mean-squared error
    RMSE = np.sqrt(np.sum((ytest - m.predict(xtest)) ** 2) / len(xtest))  # root mean squared error

    return mse, RMSE

def randomsearch(x, y, model, dist,cv , n_iter_search):
    random_search = RandomizedSearchCV(model,dist, cv = cv, n_iter = n_iter_search)
    random_search.fit(x, y)
    top_params = report(random_search.grid_scores_, 3)
    return top_params


#alphas
alphas = np.linspace(0.1,1,10)

#Ridge Regression

ridgecv = RidgeCV(alphas = alphas, scoring = 'neg_mean_squared_error', normalize = True)
ridgecv.fit(xtri, ytrain)
ridge_final = Ridge(alpha = ridgecv.alpha_, normalize = True)

model_fit(ridge_final) #30.06

#Lasso Regression

lassocv = LassoCV(alphas = None, cv = 10, max_iter = 100000, normalize = True)
lassocv.fit(xtri, ytrain)
lasso_final = Lasso(alpha =lassocv.alpha_, normalize= True)
lasso_final.fit(xtri, ytrain)

model_fit(lasso_final)  #28.21

#Elastic Net

elntcv = ElasticNetCV(eps = None, n_alphas = None, alphas= alphas, l1_ratio = [0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0],cv=5, normalize= True)
elntcv.fit(xtri, ytrain)

elnt_final = ElasticNet(alpha = elntcv.alpha_, l1_ratio= elntcv.l1_ratio_, normalize= True)
elnt_final.fit(xtri, ytrain)

model_fit(elnt_final) #26.62

#CART

param_dist = {"min_samples_split": range(2, 20,2),
              "max_depth": range(2, 20,2)
              }

rt = DecisionTreeRegressor()
rt_rs = RandomizedSearchCV(rt, param_distributions = param_dist,cv = 5, n_iter=20)
rt_rs.fit(xtrin, ytrain)

print("best parameters: ", rt_rs.best_params_)

rt_rs_final = DecisionTreeRegressor(max_depth=6, min_samples_split=4)
model_fit(rt_rs_final)


#RandomForestRegression

random_grid = {'n_estimators': range(200, 1200, 200)}

rfr = RandomForestRegressor()
rfr_rs = GridSearchCV(rfr, random_grid,cv = 5)
rfr_rs.fit(xtrin, ytrain)

print("best parameters: ", rfr_rs.best_params_)

rfr_rs_final = RandomForestRegressor(n_estimators = 800)
model_fit(rfr_rs_final)

#Support Vector Regression

param_grid = [
  {'C': [1,10, 100, 1000], 'kernel': ['linear']},
  {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
 ]

scorer = make_scorer(mean_squared_error, greater_is_better=False)
svr_gs = GridSearchCV(SVR(epsilon = 0.01), param_grid, cv = 5, scoring=scorer)
svr_gs.fit(xtrin, ytrain)

print("best parameters: ", svr_gs.best_params_) #c =1000, kernal = linear

svr_gs_final = SVR(epsilon= 0.1, kernel= 'linear', C = 1000)
model_fit(svr_gs_final)


#Adaboost

search_grid={'n_estimators':[500,1000],'learning_rate':[0.01,.1]}

ada_rs = GridSearchCV(AdaBoostRegressor(base_estimator= SVR(kernel = 'linear',C = 1000)), search_grid, cv = 5)
ada_rs.fit(xtrin, ytrain)

print("best parameters: ", ada_rs.best_params_)

ada_rs = AdaBoostRegressor(base_estimator= SVR(kernel = 'linear',C = 1000), n_estimators= 1000, learning_rate= 0.01)
ada_rs.fit(xtrin, ytrain)









