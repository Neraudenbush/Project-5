# Project-5
The goal of project 5 is to demonstrate using various methods of variable selection techniques such as Lasso, Ridge, Elastic Net, Square Root Lasso, and SCAD


Set up
```Python
%matplotlib inline
%config InlineBackend.figure_format = 'retina'
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 150

# general imports
import numpy as np
import pandas as pd
from math import ceil
from scipy import linalg
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.datasets import make_spd_matrix
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.linalg import toeplitz
from matplotlib import pyplot
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold

from scipy.optimize import minimize
from sklearn.base import BaseEstimator, RegressorMixin
from numba import njit
```
### Step 1
Now We must define the functions that we will use for square root lasso and SCAD

```Python
#Function for Square Root Lasso
class SQRTLasso:
    def __init__(self, alpha=0.01):
        self.alpha = alpha
    
    def fit(self, x, y):
        alpha=self.alpha
        def f_obj(x,y,beta,alpha):
          n =len(x)
          beta = beta.flatten()
          beta = beta.reshape(-1,1)
          output = np.sqrt(1/n*np.sum((y-x.dot(beta))**2)) + alpha*np.sum(np.abs(beta))
          return output
        
        def f_grad(x,y,beta,alpha):
          n=x.shape[0]
          p=x.shape[1]
          beta = beta.flatten()
          beta = beta.reshape(-1,1)
          output = np.array((-1/np.sqrt(n))*np.transpose(x).dot(y-x.dot(beta))/np.sqrt(np.sum((y-x.dot(beta))**2))+alpha*np.sign(beta)).flatten()
          return output
        
        def objective(beta):
          return(f_obj(x,y,beta,alpha))

        def gradient(beta):
          return(f_grad(x,y,beta,alpha))
        
        beta0 = np.ones((x.shape[1],1))
        output = minimize(objective, beta0, method='L-BFGS-B', jac=gradient,options={'gtol': 1e-8, 'maxiter': 50000,'maxls': 25,'disp': True})
        beta = output.x
        self.coef_ = beta
        
    def predict(self, x):
        return x.dot(self.coef_)
```
```Python
#Functions for SCAD
@njit
def scad_penalty(beta_hat, lambda_val, a_val):
    is_linear = (np.abs(beta_hat) <= lambda_val)
    is_quadratic = np.logical_and(lambda_val < np.abs(beta_hat), np.abs(beta_hat) <= a_val * lambda_val)
    is_constant = (a_val * lambda_val) < np.abs(beta_hat)
    
    linear_part = lambda_val * np.abs(beta_hat) * is_linear
    quadratic_part = (2 * a_val * lambda_val * np.abs(beta_hat) - beta_hat**2 - lambda_val**2) / (2 * (a_val - 1)) * is_quadratic
    constant_part = (lambda_val**2 * (a_val + 1)) / 2 * is_constant
    return linear_part + quadratic_part + constant_part

@njit    
def scad_derivative(beta_hat, lambda_val, a_val):
    return lambda_val * ((beta_hat <= lambda_val) + (a_val * lambda_val - beta_hat)*((a_val * lambda_val - beta_hat) > 0) / ((a_val - 1) * lambda_val) * (beta_hat > lambda_val))
    


class SCAD(BaseEstimator, RegressorMixin):
    def __init__(self, a=2,lam=1):
        self.a, self.lam = a, lam
  
    def fit(self, x, y):
        a = self.a
        lam   = self.lam

        @njit
        def scad(beta):
          beta = beta.flatten()
          beta = beta.reshape(-1,1)
          n = len(y)
          return 1/n*np.sum((y-x.dot(beta))**2) + np.sum(scad_penalty(beta,lam,a))

        @njit  
        def dscad(beta):
          beta = beta.flatten()
          beta = beta.reshape(-1,1)
          n = len(y)
          output = -2/n*np.transpose(x).dot(y-x.dot(beta))+scad_derivative(beta,lam,a)
          return output.flatten()
        
        
        beta0 = np.zeros(p)
        output = minimize(scad, beta0, method='L-BFGS-B', jac=dscad,options={'gtol': 1e-8, 'maxiter': 50000,'maxls': 50,'disp': False})
        beta = output.x
        self.coef_ = beta
        
    def predict(self, x):
        return x.dot(self.coef_)

```

### Finding Optimal Hyperparameters
I could not figure out how to use GridSearchCV after going back and looking at the posted code and lectures so instead I simulated 1 data set and attempted
to find the optimal hyperparameters on that and apply it to each of the 100 data sets

I know this is probably not the most accruate way but its all I could figure out. 
```Python
n = 200
p = 1200

#Real values that we are searching for
beta_star = np.concatenate(([1]*7,[0]*25,[0.25]*5,[0]*50,[0.7]*15,[0]*1098))

#non zero coefficients
pos = np.where(beta_star!=0)

#Generating the ficticous data
v = []
for i in range(p):
  v.append(0.8**i)

mu = [0]*p
sigma = 3.5
# Generate the random samples.
np.random.seed(410)
x = np.random.multivariate_normal(mu, toeplitz(v), size=n) # this where we generate some fake data
#y = X.dot(beta) + sigma*np.random.normal(0,1,[num_samples,1])
y = np.matmul(x,beta_star).reshape(-1,1) + sigma*np.random.normal(0,1,size=(n,1))
```
Now that we have the data sets we apply each of the models to the data to find the optimal hyper parameters
This was determined by which hyper parameters gave the lowest MSE

```Python
#Starting with Lasso and testing using the Norm it appears that an Alpha of .84 is optimal
model_Las = Lasso(alpha=.84,fit_intercept=False,max_iter=10000)
model_Las.fit(x,y)
```
```Python
#After some testing alpha for Ridge is optimal at about 380
#I did some research online and it seemed ok to have an alpha this large but I am not sure
model_Rid = Ridge(alpha=380,fit_intercept=False,max_iter=10000)
model_Rid.fit(x,y)
```
```Python
#after sone testing it appears that a good alpha value is .9 and the optimal l1 ratio is .59
model_ElN = ElasticNet(alpha=.9,l1_ratio=.59,fit_intercept=False,max_iter=10000)
model_ElN.fit(x,y)
```
```Python
#After some testing it appears that .16 is the optimal alpha value
model_SRL = SQRTLasso(alpha=0.16)
model_SRL.fit(x,y)
```
```Python
#After testing SCAD the optimal hyperparameters seem to be .91 for Alpha and .2 for Lambda
model_SCAD = SCAD(a=.91,lam=.2)
model_SCAD.fit(x,y)
```

### Apply to 100 sets of fictucious data

Now we generate 100 sets of fictucous data and loop each of the models with our optimal hyperparameters on all of them
I recorded the RMSE l2 norms and the number of correct non zero results and average them to determine which model is optimal

```Python
NumCorrect_lasso = []
NumCorrect_ridge = []
NumCorrect_elastic = []
NumCorrect_sqrtlas = []
NumCorrect_scad = []


for s in range(100):
  v = []
  for i in range(p):
    v.append(0.8**i)

  mu = [0]*p
  sigma = 3.5
  # Generate the random samples.
  np.random.seed(s)
  x = np.random.multivariate_normal(mu, toeplitz(v), size=n) # this where we generate some fake data
  #y = X.dot(beta) + sigma*np.random.normal(0,1,[num_samples,1])
  y = np.matmul(x,beta_star).reshape(-1,1) + sigma*np.random.normal(0,1,size=(n,1))


  model_Las.fit(x,y)
  rmse_lasso.append(np.sqrt(mean_squared_error(y,model_Las.predict(x))))
  l2_lasso.append(np.linalg.norm(model_Las.coef_-beta_star,ord=2))
  pos_lasso = np.where(model_Las.coef_!=0)
  NumCorrect_lasso.append(np.intersect1d(pos,pos_lasso).shape[0])

  model_Rid.fit(x,y)
  rmse_ridge.append(np.sqrt(mean_squared_error(y,model_Rid.predict(x))))
  l2_ridge.append(np.linalg.norm(model_Rid.coef_-beta_star,ord=2))
  pos_ridge = np.where(model_Rid.coef_!=0)
  NumCorrect_ridge.append(np.intersect1d(pos,pos_ridge).shape[0])

  model_ElN.fit(x,y)
  rmse_elastic.append(np.sqrt(mean_squared_error(y,model_ElN.predict(x))))
  l2_elastic.append(np.linalg.norm(model_ElN.coef_-beta_star,ord=2))
  pos_elastic = np.where(model_ElN.coef_!=0)
  NumCorrect_elastic.append(np.intersect1d(pos,pos_elastic).shape[0])

  model_SRL.fit(x,y)
  rmse_sqrtlas.append(np.sqrt(mean_squared_error(y,model_SRL.predict(x))))
  l2_sqrtlas.append(np.linalg.norm(model_SRL.coef_-beta_star,ord=2))
  pos_sqrtlas = np.where(model_SRL.coef_!=0)
  NumCorrect_sqrtlas.append(np.intersect1d(pos,pos_sqrtlas).shape[0])

  model_SCAD.fit(x,y)
  rmse_scad.append(np.sqrt(mean_squared_error(y,model_SCAD.predict(x))))
  l2_scad.append(np.linalg.norm(model_SCAD.coef_-beta_star,ord=2))
  pos_scad = np.where(model_SCAD.coef_!=0)
  NumCorrect_scad.append(np.intersect1d(pos,pos_scad).shape[0])
```
Now Report the results
```Python
print('Average Root MSE for Lasso is', np.mean(rmse_lasso))
print('Average l2 distance for Lasso is', np.mean(l2_lasso))
print('Average number of correct non-zero coefficients for Lasso is', np.mean(NumCorrect_lasso))
print()
print('Average Root MSE for Ridge is', np.mean(rmse_ridge))
print('Average l2 distance for Ridge is', np.mean(l2_ridge))
print('Average number of correct non-zero coefficients for Ridge is', np.mean(NumCorrect_ridge))
print()
print('Average Root MSE for Elastic Net is', np.mean(rmse_elastic))
print('Average l2 distance for Elastic Net is', np.mean(l2_elastic))
print('Average number of correct non-zero coefficients for elastic net is', np.mean(NumCorrect_elastic))
print()
print('Average Root MSE for Square Root Lasso is', np.mean(rmse_sqrtlas))
print('Average l2 distance for Square Root Lasso is', np.mean(l2_sqrtlas))
print('Average number of correct non-zero coefficients for Square Root Lasso is', np.mean(NumCorrect_sqrtlas))
print()
print('Average Root MSE for SCAD is', np.mean(rmse_scad))
print('Average l2 distance for SCAD is', np.mean(l2_scad))
print('Average number of correct non-zero coefficients for SCAD is', np.mean(NumCorrect_scad))
print()
```
Output
```Markdown
Average Root MSE for Lasso is 3.807166487525358
Average l2 distance for Lasso is 2.5917062359299017
Average number of correct non-zero coefficients for Lasso is 18.99

Average Root MSE for Ridge is 2.1384954485213052
Average l2 distance for Ridge is 2.7713568017843415
Average number of correct non-zero coefficients for Ridge is 27.0

Average Root MSE for Elastic Net is 3.511669965132439
Average l2 distance for Elastic Net is 1.47758221146977
Average number of correct non-zero coefficients for elastic net is 24.61

Average Root MSE for Square Root Lasso is 3.4561694207211104
Average l2 distance for Square Root Lasso is 1.3200285669763323
Average number of correct non-zero coefficients for Square Root Lasso is 27.0

Average Root MSE for SCAD is 9.643333594454337
Average l2 distance for SCAD is 3.8291643997091573
Average number of correct non-zero coefficients for SCAD is 27.0
```

Conclusion
Ridge offered the lowest RMSE of all of the models while Elastic Net and Square Root Lasso both had the smallest L2 norm distances
Ridge, SR Lasso, and SCAD all correctly included all 27 non zero coefficients but this was because they all predicted a very large amount
