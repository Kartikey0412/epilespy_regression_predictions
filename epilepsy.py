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


#alphas
alphas = np.linspace(0.1,1,10)

#Ridge Regression

ridgecv = RidgeCV(alphas = alphas, scoring = 'neg_mean_squared_error', normalize = True)
ridgecv2 = RidgeCV(alphas = alphas, scoring = 'neg_mean_squared_error')

ridgecv.fit(xtri, ytrain)
ridgecv2.fit(xtrin, ytrain)
ridgecv.alpha_


ridge_final = Ridge(alpha = ridgecv.alpha_, normalize = True)
ridge_final.fit(xtri, ytrain)

ridge_final2 = Ridge(alpha = ridgecv2.alpha_)
ridge_final2.fit(xtrin, ytrain)
#mean_squared_error(ytest_nm, ridge_final.predict(xtri))

score_ridge = ridge_final.score(xtest_nm, ytest_nm) #-0.144
mean_squared_error(ytest_nm, ridge_final.predict(xtest_nm)) #30.46
mean_squared_error(ytest_nm, ridge_final2.predict(xtest_nmn)) #39.48 (on seperately normalizing conitnous data)

#Lasso Regression

lassocv = LassoCV(alphas = None, cv = 10, max_iter = 100000, normalize = True)
lassocv.fit(xtri, ytrain)

lasso_final = Lasso(alpha =lassocv.alpha_, normalize= True)
lasso_final.fit(xtri, ytrain)

score_lasso = lasso_final.score(xtest_nm, ytest_nm) #-0.06

mean_squared_error(ytest_nm, lasso_final.predict(xtest_nm)) #28.21

#Elastic Net

elntcv = ElasticNetCV(eps = None, n_alphas = None, alphas= alphas, l1_ratio = [0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0],cv=5, normalize= True)
elntcv.fit(xtri, ytrain)
elntcv2 = ElasticNetCV(eps = None, n_alphas = None, alphas= alphas, l1_ratio = [0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0],cv=5)
elntcv2.fit(xtrin, ytrain)

elnt_final = ElasticNet(alpha = elntcv.alpha_, l1_ratio= elntcv.l1_ratio_, normalize= True)
elnt_final.fit(xtri, ytrain)
elnt_final2 = ElasticNet(alpha = elntcv2.alpha_, l1_ratio= elntcv2.l1_ratio_)
elnt_final2.fit(xtrin, ytrain)

score_elnt = elnt_final.score(xtest_nm, ytest_nm) #-0.0004
mean_squared_error(ytest_nm, elnt_final.predict(xtest_nm)) #26.62
mean_squared_error(ytest_nm, elnt_final2.predict(xtest_nmn)) #27.37


#CART

def randomsearch(x, y, model, dist,cv , n_iter_search):
    random_search = RandomizedSearchCV(model,dist, cv = cv, n_iter = n_iter_search)
    random_search.fit(x, y)
    #top_params = report(random_search.grid_scores_, 3)
    return top_params

param_dist = {"min_samples_split": range(2, 20,2),
              "max_depth": range(2, 20,2)
              }
#"criterion": ["gini", "entropy"]

rt = DecisionTreeRegressor()
rt_rs = RandomizedSearchCV(rt, param_distributions = param_dist,cv = 5, n_iter=20)
rt_rs.fit(xtrin, ytrain)

print("best parameters: ", rt_rs.best_params_)

rt_rs_final = DecisionTreeRegressor(max_depth=6, min_samples_split=4)
rt_rs_final.fit(xtrin, ytrain)

mean_squared_error(ytest_nm, rt_rs_final.predict(xtest_nm)) #13623.02
RMSE = np.sqrt(np.sum((ytest_nm - rt_rs_final.predict(xtest_nm))**2)/len(xtest_nm))#116.71


#RandomForestRegression

random_grid = {'n_estimators': range(200, 1200, 200)}

rfr = RandomForestRegressor()
rfr_rs = GridSearchCV(rfr, random_grid,cv = 5)
rfr_rs.fit(xtrin, ytrain)

print("best parameters: ", rfr_rs.best_params_)

rfr_rs_final = RandomForestRegressor(n_estimators = 800)
rfr_rs_final.fit(xtrin, ytrain)

mean_squared_error(ytest_nm, rfr_rs_final.predict(xtest_nm)) #4992.66
RMSE_rfr_rs_final = np.sqrt(np.sum((ytest_nm - rfr_rs_final.predict(xtest_nm))**2)/len(xtest_nm)) #70.66

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
svr_gs_final.fit(xtrin, ytrain)

mean_squared_error(ytest_nm, svr_gs_final.predict(xtest_nm)) #2853.93
RMSE_svr_gs_final = np.sqrt(np.sum((ytest_nm - svr_gs_final.predict(xtest_nm))**2)/len(xtest_nm)) #50.83


#Adaboost

search_grid={'n_estimators':[500,1000],'learning_rate':[0.01,.1]}

ada_rs = GridSearchCV(AdaBoostRegressor(base_estimator= SVR(kernel = 'linear',C = 1000)), search_grid, cv = 5)
ada_rs.fit(xtrin, ytrain)

print("best parameters: ", ada_rs.best_params_)

ada_rs = AdaBoostRegressor(base_estimator= SVR(kernel = 'linear',C = 1000), n_estimators= 1000, learning_rate= 0.01)
ada_rs.fit(xtrin, ytrain)





