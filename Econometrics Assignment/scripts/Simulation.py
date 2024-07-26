# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 16:31:54 2023

@author: Aleksandr Medvedev
"""

import os
# for linear algebra
import numpy as np
import numpy.linalg as la
# for plotting
import matplotlib.pyplot as plt
# for econometric models (OLS)
import statsmodels.api as sm
# for quantiles
import scipy.stats as stats

colors=['red', 'blue', 'green']


figure_dir = '../figures/'

# ALL FUNCTIONS ARE UP TOP, THE MAIN CODE IS BELOW

# -----------------------------------------------------------------------------

def hist_plot(x, label,  fignum = 1, num_bins=100, t = True, num_obs = None, normal=False):

    # create bins from the minimum to the maximum divided by num_bins
    bins = np.linspace(x.min(), x.max(), num_bins)
    # create a histogram
    plt.figure(fignum, figsize= (10,5))
    #The "issue" here with the density function is that it doesn't provide probabilities
    plt.hist(x, bins, density=True, label=label)
    # if you need a t-distribution
    if t == True:
        plt.plot(bins, stats.t.pdf(bins, num_obs-1, loc=x.mean(),
                           scale=1), color=colors[0], label = "t-distribution")
        
    # use gaussian distribution
    elif normal == True:
        plt.plot(bins, stats.norm.pdf(bins, loc=x.mean(),
                           scale=x.std()), color=colors[0], label = "Normal distribution")
    
    # set the legend
    plt.legend(loc='best')
    # set the title
    plt.title('Distribution ' + label)
    # save the figure
    plt.savefig(figure_dir + "Figure_{}".format(fignum))
    plt.show()
    
def model_test(y, X, beta_hat, num_obs, num_params=1):
    # This function assumes a constant coefficient is used
    # predicted values
    y_hat = X @ beta_hat
    
    # residuals
    e = y - y_hat
    
    # sigma estimates
    sigma2_hat = np.sum(e * e, 0)/ (num_obs - num_params - 1)
    sigma_hat = np.sqrt(sigma2_hat)
    
    # standard errors of beta
    # num_sims columns of estimators
    # each column represents the standard errors
    beta_se = sigma_hat * np.sqrt(np.diag(la.inv(X_100.T@X_100)))[:, None]

    t_values = (beta_hat - 0) / beta_se
    
    return t_values

def model_test_GLS(y, X, beta_hat, num_obs, num_params=1):
    
    # This function assumes a constant coefficient is used
    # predicted values
    y_hat = X @ beta_hat
    
    # residuals
    e = (y - y_hat)
    
    # sigma estimates
    sigma2_hat = np.sum(e * e, 0) / (num_obs - num_params - 1)
    
    XO_1 = np.fliplr(X**(-1)).T
    XOX = XO_1 @ X
    XOX_1 = la.inv(XOX)

    beta_se = np.outer(np.sqrt(np.diag(XOX_1)),sigma2_hat)

    t_values = (beta_hat - 0) / beta_se
    
    return t_values

def rejection(t_values, num_obs, num_vars, case=None):
    # confidence interval
    alpha = 0.05
    z_alpha = stats.t.ppf(1 - alpha/2, num_obs-num_vars-1)
    
    # calculates a rejection indicator (0, 1) for each scenario and coefficient
    # then calculates the percentage of scenarios for which there is a rejection.
    # expect this to be close to alpha
    rejection_rate = np.sum(np.abs(t_values) > z_alpha, 1) / num_sims
    print (rejection_rate, case)
    
def scatter_plot(x, y, fignum = 1, title = None, xlabel = None, ylabel = None):
    plt.figure(fignum, figsize= (10,5))
    plt.scatter(x,y)
    plt.title(title)
    plt.xlabel(xlabel)  
    plt.ylabel(ylabel)
    plt.savefig(figure_dir + "Figure_{}".format(fignum))
    plt.show
    
def t_statistics(y, X, beta_hat, num_obs,num_params=1):
    # This function assumes a constant coefficient is used
    # predicted values
    y_hat = X @ beta_hat
    
    # residuals
    e = y - y_hat
    
    # sigma estimates
    sigma2_hat = np.sum(e * e, 0)/ (num_obs - num_params - 1)
    sigma_hat = np.sqrt(sigma2_hat)
    
    # standard errors of beta
    # num_sims columns of estimators
    # each column represents the standard errors
    beta_se = sigma_hat * np.sqrt(np.diag(la.inv(X.T @ X)))[:, None]
    
    # Again num_sims columns of t-values
    # each column represents the t value for the given simulation
    # Note we are testing if the estimated value equals the true value
    # Hence, we would expected a alpha rejection rate
    beta = np.array([1.0, 2.0])
    
    t_values = (beta_hat - beta[:, None]) / beta_se
    
    return t_values

def t_statistics_GLS(y, X, beta_hat, num_obs,num_params=1):
    
    # This function assumes a constant coefficient is used
    # predicted values
    y_hat = X @ beta_hat
    
    # residuals
    e = (y - y_hat)
    
    # sigma estimates
    sigma2_hat = np.sum(e * e, 0) / (num_obs - num_params - 1)
    
    XO_1 = np.fliplr(X**(-1)).T
    XOX = XO_1 @ X
    XOX_1 = la.inv(XOX)

    beta_se = np.outer(np.sqrt(np.diag(XOX_1)),sigma2_hat)
    
    # Again num_sims columns of t-values
    # each column represents the t value for the given simulation
    # Note we are testing if the estimated value equals the true value
    # Hence, we would expected a alpha rejection rate
    beta = np.array([1.0, 2.0])
    
    t_values = (beta_hat - beta[:, None]) / beta_se
    
    return t_values

def t_statistics_white(y, X, beta_hat, white_error):
    
    # standard errors of beta
    # num_sims columns of estimators
    # each column represents the standard errors
    beta_se = white_error 
    
    # Again num_sims columns of t-values
    # each column represents the t value for the given simulation
    # Note we are testing if the estimated value equals the true value
    # Hence, we would expected a alpha rejection rate
    beta = np.array([1.0, 2.0])
    
    t_values = (beta_hat - beta[:, None]) / beta_se
    
    return t_values

def white_standard_error(y, X, beta_hat):

    # Loop over each simulation
    # Calculate residuals for the current simulation
    y_hat = X @ beta_hat
    residuals = y - y_hat
    
    # Form the heteroskedasticity-consistent covariance matrix
    residuals_squared = np.square(residuals)

    A = la.inv(X.T @ X) @ X.T
    
    A_squared = A * A 
    
    white_standard_errors = A_squared @ residuals_squared
    white_se = np.sqrt(white_standard_errors) 

    return white_se

#---------------------------------------------------------------------------------

# for the figures
figure_folder = "../figures/"
if not os.path.exists(figure_folder):
    os.makedirs(figure_folder)
# for the latex document
report_folder = "../report/"
if not os.path.exists(report_folder):
    os.makedirs(report_folder)

# first birthday
bd_1 = 703

rng = np.random.default_rng(bd_1)

""" QUESTION 1 - Generating x1 """ 
num_vars = 1
num_obs = 100

# This line is actually important because of how the rng function works. If you remove this line
# You will get different graphs for the error distributions 
x = rng.normal(20.0, 2.0, (100, num_vars))

x = {}
x['1000'] = rng.normal(20.0, 2.0, (1000, num_vars))
x['100'] = rng.normal(20.0, 2.0, (100, num_vars))

x_bar = {}
x_bar['1000'] = np.average(x['1000'])
x_bar['100'] = np.average(x['100'])

# add a constant
X_1000 = sm.add_constant(x['1000'])
X_100 = sm.add_constant(x['100'])

""" QUESTION 2 - Generating error terms for each observation """ 


# beta = (1, 2)
beta = np.array([1.0, 2.0])

# sigma
sigma = 0.3/np.sqrt(20)

# set the number of simulations (higher -: more accurate results)
num_sims = 2**16

E = {}

# In case if it doesn't use the square of the second parameter
# E['Het_1000'] = rng.normal(0.0, (sigma**2) * x['1000'], (1000, num_sims))
#Here the square root is needed because rng.normal squares the second parameter

E['Het_1000'] = rng.normal(0.0, sigma * np.sqrt(x['1000']), (1000, num_sims))
# print(np.var(E['Het_1000'][:,0]))

E['Het_100'] = rng.normal(0.0, sigma * np.sqrt(x['100']), (100, num_sims))
E['Hom_1000'] = rng.normal(0.0, sigma * np.sqrt(x_bar['1000']), (1000, num_sims))

# E['Hom_1000'] = rng.normal(0.0, (sigma**2) * x_bar['1000'], (1000, num_sims))

E['Hom_100'] = rng.normal(0.0, sigma * np.sqrt(x_bar['100']), (100, num_sims))

""" QUESTION 3 - Generating the dependent variable """

Y_1000_Het = X_1000 @ beta[:, None] + E['Het_1000']
Y_1000_Hom = X_1000 @ beta[:, None] + E['Hom_1000']
Y_100_Het = X_100 @ beta[:, None] + E['Het_100']
Y_100_Hom = X_100 @ beta[:, None] + E['Hom_100']

""" QUESTION 4 - Plotting the error terms """

avg_error_1000_het = np.sum(E['Het_1000'],1)/1000
avg_error_1000_hom = np.sum(E['Hom_1000'],1)/1000
avg_error_100_het = np.sum(E['Het_100'],1)/100
avg_error_100_hom = np.sum(E['Hom_100'],1)/100

fignum = 1

# Plotting the histograms 
# Bins were adjusted through trial and error to get more or less good looks
hist_plot(avg_error_1000_het, 'Avg. Error 1000 Het',fignum, 24,False, normal=True)
fignum += 1

# print statistics

# print("Mean:", np.mean(avg_error_1000_het))
# print("Variance:", np.var(avg_error_1000_het))
# print("Skewness:", stats.skew(avg_error_1000_het))
# print("Kurtosis: ", stats.kurtosis(avg_error_1000_het))

hist_plot(avg_error_1000_hom, 'Avg. Error 1000 Hom',fignum, 24, False, normal=True)
fignum += 1
hist_plot(avg_error_100_het, 'Avg. Error 100 Het',fignum, 17, num_obs = 100)
fignum += 1
hist_plot(avg_error_100_hom, 'Avg. Error 100 Hom',fignum, 17, num_obs = 100)
fignum += 1

model = sm.OLS(Y_1000_Het, X_1000)
results = model.fit()

#Plotting all of the results really takes a while, but it should work (Takes 10 mins to run the whole program)
scatter_plot(results.fittedvalues, np.square(results.resid),
              fignum,'Errors VS Fitted 1000 Het', 'Fitted Values', 'Error Term')
fignum += 1

model = sm.OLS(Y_100_Het, X_100)
results = model.fit()


scatter_plot(results.fittedvalues, np.square(results.resid),
              fignum,'Errors VS Fitted 100 Het', 'Fitted Values', 'Error Term')
fignum += 1

model = sm.OLS(Y_1000_Hom, X_1000)
results = model.fit()

scatter_plot(results.fittedvalues, np.square(results.resid),
              fignum,'Errors VS Fitted 1000 Hom', 'Fitted Values', 'Error Term')
fignum += 1

model = sm.OLS(Y_100_Hom, X_100)
results = model.fit()

scatter_plot(results.fittedvalues, np.square(results.resid),
              fignum,'Errors VS Fitted 100 Hom', 'Fitted Values', 'Error Term')
fignum += 1
#Scatter plot part end

""" QUESTION 5 - Omega Matrix """

# GLS Heteroskedastic
# Omega would be the variance times the weighted matrix 

model = sm.OLS(Y_100_Het, X_100)
results = model.fit()

omega_het_100 = np.cov(results.resid) 

model = sm.OLS(Y_1000_Het, X_1000)
results = model.fit()

omega_het_1000 = np.cov(results.resid) 

# GLS Homoskedastic
# Omega in homoskedastic case will be the variance of the error terms times the identity matrix

model = sm.OLS(Y_100_Hom, X_100)
results = model.fit()

omega_hom_100 = np.cov(results.resid)

model = sm.OLS(Y_1000_Hom, X_1000)
results = model.fit()
 
omega_hom_1000 = np.cov(results.resid)

""" QUESTION 6 - Beta_hat estimates """

#Homoskedastic case, OLS Estimation
# Both yield identical results therefore only 1 is sufficient
Hom_beta_hat_OLS_100 = la.inv(X_100.T @ X_100)@X_100.T@Y_100_Hom
Hom_beta_hat_GLS_100 = la.inv(X_100.T @ la.inv(omega_hom_100) @ X_100) @ X_100.T @ la.inv(omega_hom_100) @ Y_100_Hom


#Heteroskedastic case
Het_beta_hat_OLS_100 = la.inv(X_100.T @ X_100) @ X_100.T@Y_100_Het
Het_beta_hat_GLS_100 = la.inv(X_100.T @ la.inv(omega_het_100) @ X_100) @ X_100.T @ la.inv(omega_het_100) @ Y_100_Het

#Calculating t statistics
t_hom_100_GLS = t_statistics(Y_100_Hom, X_100, Hom_beta_hat_GLS_100, 100)
t_het_100_GLS = t_statistics_GLS(Y_100_Het, X_100, Het_beta_hat_GLS_100, 100)
t_het_100_OLS = t_statistics(Y_100_Het, X_100, Het_beta_hat_OLS_100, 100)
t_hom_100_OLS = t_statistics(Y_100_Hom, X_100, Hom_beta_hat_OLS_100, 100)

# is model test just checking whether the b = 0?
model_hom_100_GLS = model_test_GLS(Y_100_Hom, X_100, Hom_beta_hat_GLS_100, 100)
model_het_100_GLS = model_test_GLS(Y_100_Het, X_100, Het_beta_hat_GLS_100, 100)
model_het_100_OLS = model_test(Y_100_Het, X_100, Het_beta_hat_OLS_100, 100)

Hom_beta_hat_GLS_1000 = la.inv(X_1000.T @ la.inv(omega_hom_1000) @ X_1000) @ X_1000.T @ la.inv(omega_hom_1000) @ Y_1000_Hom
Hom_beta_hat_OLS_1000 = la.inv(X_1000.T @ X_1000)@X_1000.T@Y_1000_Hom

#Heteroskedastic case
Het_beta_hat_OLS_1000 = la.inv(X_1000.T @ X_1000) @ X_1000.T@Y_1000_Het
Het_beta_hat_GLS_1000 = la.inv(X_1000.T @ la.inv(omega_het_1000) @ X_1000) @ X_1000.T @ la.inv(omega_het_1000) @ Y_1000_Het

#Calculating t statistics
t_hom_1000_GLS = t_statistics(Y_1000_Hom, X_1000, Hom_beta_hat_GLS_1000, 1000)
t_het_1000_GLS = t_statistics_GLS(Y_1000_Het, X_1000, Het_beta_hat_GLS_1000, 1000)
t_het_1000_OLS = t_statistics(Y_1000_Het, X_1000, Het_beta_hat_OLS_1000, 1000)
t_hom_1000_OLS = t_statistics(Y_1000_Hom, X_1000, Hom_beta_hat_OLS_1000, 1000)

# model test
model_hom_1000_GLS = model_test_GLS(Y_1000_Hom, X_1000, Hom_beta_hat_GLS_1000, 1000)
model_het_1000_GLS = model_test_GLS(Y_1000_Het, X_1000, Het_beta_hat_GLS_1000, 1000)
model_het_1000_OLS = model_test(Y_1000_Het, X_1000, Het_beta_hat_OLS_1000, 1000)


""" QUESTION 7 - Plotting histograms for t-values """

# T-statics plotting for the true values
hist_plot(t_hom_100_GLS[0], label= r"$\beta_{0 GLS}$ Homoskedastic, n = 100, t-test",fignum = fignum,num_obs=num_sims, t=False)
fignum += 1
hist_plot(t_hom_100_GLS[1], label=r"$\beta_{1 GLS}$ Homoskedastic, n = 100, t-test",fignum = fignum,num_obs=num_sims, t=False)
fignum += 1

hist_plot(t_het_100_GLS[0], label=r"$\beta_{0 GLS}$ Heteroskedastic, n = 100, t-test",fignum = fignum,num_obs=num_sims, t=False)
fignum += 1
hist_plot(t_het_100_GLS[1], label=r"$\beta_{1 GLS}$ Heteroskedastic, n = 100, t-test",fignum = fignum,num_obs=num_sims, t=False)
fignum += 1

hist_plot(t_het_100_OLS[0], label=r"$\beta_{0 OLS}$ Heteroskedastic, n = 100, t-test",fignum = fignum,num_obs=num_sims, t=False)
fignum += 1
hist_plot(t_het_100_OLS[1], label=r"$\beta_{1 OLS}$ Heteroskedastic, n = 100, t-test",fignum = fignum,num_obs=num_sims, t=False)
fignum += 1

hist_plot(t_hom_1000_GLS[0], label=r"$\beta_{0 GLS}$ Homoskedastic,, n = 1000, t-test",fignum = fignum,num_obs=num_sims, t=False)
fignum += 1
hist_plot(t_hom_1000_GLS[1], label=r"$\beta_{1 GLS}$ Homoskedastic,, n = 1000, t-test",fignum = fignum,num_obs=num_sims, t=False)
fignum += 1

hist_plot(t_het_1000_GLS[0], label=r"$\beta_{0 OLS}$ Heteroskedastic, n = 1000, t-test",fignum = fignum,num_obs=num_sims, t=False)
fignum += 1
hist_plot(t_het_1000_GLS[1], label=r"$\beta_{1 GLS}$ Heteroskedastic, n = 1000, t-test",fignum = fignum,num_obs=num_sims, t=False)
fignum += 1

hist_plot(t_het_1000_OLS[0], label=r"$\beta_{0 OLS}$ Heteroskedastic, n = 1000, t-test",fignum = fignum,num_obs=num_sims, t=False)
fignum += 1
hist_plot(t_het_1000_OLS[1], label=r"$\beta_{1 OLS}$ Heteroskedastic, n = 1000, t-test",fignum = fignum,num_obs=num_sims, t=False)
fignum += 1



#T-statistics for the model test
hist_plot(model_hom_100_GLS[0], label=r"$\beta_{0 GLS}$ Homoskedastic, n = 100, model test",fignum = fignum,num_obs=num_sims, t=False)
fignum += 1
hist_plot(model_hom_100_GLS[1], label=r"$\beta_{1 GLS}$ Homoskedastic, n = 100, model test",fignum = fignum,num_obs=num_sims, t=False)
fignum += 1

hist_plot(model_het_100_GLS[0], label=r"$\beta_{0 GLS}$ Heteroskedastic, n = 100, model test",fignum = fignum,num_obs=num_sims, t=False)
fignum += 1
hist_plot(model_het_100_GLS[1], label=r"$\beta_{1 GLS}$ Heteroskedastic, n = 100, model test",fignum = fignum,num_obs=num_sims, t=False)
fignum += 1

hist_plot(model_het_100_OLS[0], label=r"$\beta_{0 OLS}$ Heteroskedastic, n = 100, model test",fignum = fignum,num_obs=num_sims, t=False)
fignum += 1
hist_plot(model_het_100_OLS[1], label=r"$\beta_{1 OLS}$ Heteroskedastic, n = 100, model test",fignum = fignum,num_obs=num_sims, t=False)
fignum += 1

hist_plot(model_hom_1000_GLS[0], label=r"$\beta_{0 GLS}$ Homoskedastic, n = 1000, model test",fignum = fignum,num_obs=num_sims, t=False)
fignum += 1
hist_plot(model_hom_1000_GLS[1], label=r"$\beta_{1 GLS}$ Homoskedastic, n = 1000, model test",fignum = fignum,num_obs=num_sims, t=False)
fignum += 1

hist_plot(model_het_1000_GLS[0], label=r"$\beta_{0 GLS}$ Heteroskedastic, n = 1000, model test",fignum = fignum,num_obs=num_sims, t=False)
fignum += 1
hist_plot(model_het_1000_GLS[1], label=r"$\beta_{1 GLS}$ Heteroskedastic, n = 1000, model test",fignum = fignum,num_obs=num_sims, t=False)
fignum += 1

hist_plot(model_het_1000_OLS[0], label=r"$\beta_{0 OLS}$ Heteroskedastic, n = 1000, model test",fignum = fignum,num_obs=num_sims, t=False)
fignum += 1
hist_plot(model_het_1000_OLS[1], label=r"$\beta_{1 OLS}$ Heteroskedastic, n = 1000, model test",fignum = fignum,num_obs=num_sims, t=False)
fignum += 1

""" QUESTION 8 - Rejection level checking """

rejection(t_hom_100_GLS, num_sims, 1,'t_hom_100_GLS') 
rejection(t_het_100_GLS, num_sims, 1, 't_het_100_GLS')
rejection(t_het_100_OLS, num_sims, 1, 't_het_100_OLS')
rejection(t_hom_100_OLS, num_sims, 1,'t_hom_100_OLS') 


rejection(t_hom_1000_GLS, num_sims, 1,'t_hom_1000_GLS') 
rejection(t_het_1000_GLS, num_sims, 1, 't_het_1000_GLS')
rejection(t_het_1000_OLS, num_sims, 1, 't_het_1000_OLS')
rejection(t_hom_1000_OLS, num_sims, 1,'t_hom_1000_OLS') 

""" QUESTION 9 - Adjust the standard errors to White standard errors + rejection test"""
# # Recalc the errors
white_errors_100_OLS  = white_standard_error(Y_100_Het, X_100, Het_beta_hat_OLS_100)
white_errors_1000_OLS  = white_standard_error(Y_1000_Het, X_1000, Het_beta_hat_OLS_1000)

white_errors_100_GLS  = white_standard_error(Y_100_Het, X_100, Het_beta_hat_GLS_100)
white_errors_1000_GLS  = white_standard_error(Y_1000_Het, X_1000, Het_beta_hat_GLS_1000)

# t-stats
t_het_100_GLS = t_statistics_white(Y_100_Het, X_100, Het_beta_hat_GLS_100,white_errors_100_GLS)
t_het_100_OLS = t_statistics_white(Y_100_Het, X_100, Het_beta_hat_OLS_100,white_errors_100_OLS)

t_het_1000_GLS = t_statistics_white(Y_1000_Het, X_1000, Het_beta_hat_GLS_1000,white_errors_1000_GLS)
t_het_1000_OLS = t_statistics_white(Y_1000_Het, X_1000, Het_beta_hat_OLS_1000,white_errors_1000_OLS)

# REJECTION RATE PRINT 
rejection(t_het_100_GLS, num_sims, 1, 't_het_100_GLS_white')
rejection(t_het_100_OLS, num_sims, 1, 't_het_100_OLS_white')

rejection(t_het_1000_GLS, num_sims, 1, 't_het_1000_GLS_white')
rejection(t_het_1000_OLS, num_sims, 1, 't_het_1000_OLS_white')
