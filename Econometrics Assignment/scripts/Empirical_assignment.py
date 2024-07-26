# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 16:34:54 2023

@author: Aleksandr Medvedev
"""
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm



def data_frame_to_latex_table_file(file_name, df):
    """
    Take a pandas DataFrame and creates a file_name.tex with LaTeX table data.

    Parameters
    ----------
    file_name : string
                name of the file
    df : Pandas DataFrame
        Correlation matrix.

    Returns
    -------
    saves DataFrame to disk.
    """
    # create and open file
    text_file = open(file_name, "w")
    # data frame to LaTeX
    df_latex = df.to_latex(escape=True)
    # Consider extensions (see later in class)
    # write latex string to file
    text_file.write(df_latex)
    # close file
    text_file.close()

# -----------------------------------------------------------------------------


def summary_to_latex_table_file(file_name, summary):
    """
    Take a pandas DataFrame and creates a file_name.tex with LaTeX table data.

    Parameters
    ----------
    file_name : string
                name of the file
    df : statsmodels.iolib.summary.Summary
        Summary

    Returns
    -------
    saves Summary to disk.
    """
    # create and open file
    text_file = open(file_name, "w")
    # data frame to LaTeX
    df_latex = summary.as_latex()
    # Consider extensions (see later in class)
    # write latex string to file
    text_file.write(df_latex)
    # close file
    text_file.close()

# -----------------------------------------------------------------------------


def results_summary_to_dataframe(results, rounding=2):
    """
    Transform the result of an statsmodel results table into a dataframe.

    Parameters
    ----------
    results : string
                name of the file
    rounding : int
                rounding

    Returns
    -------
    returns a pandas DataFrame with regression results.
    """
    # get the values from results
    # if you want, you can of course generalize this.
    # e.g. if you don't have normal error terms
    # you could change the pvalues and confidence bounds
    # see exercise session 9?!
    pvals = results.pvalues
    tvals = results.tvalues
    coeff = results.params
    conf_lower = results.conf_int()[0]
    conf_higher = results.conf_int()[1]

    # create a pandas DataFrame from a dictionary
    results_df = pd.DataFrame({"pvals": np.round(pvals, rounding),
                               "tvals": np.round(tvals, rounding),
                               "coeff": np.round(coeff, rounding),
                               "conf_lower": np.round(conf_lower, rounding),
                               "conf_higher": np.round(conf_higher, rounding)})
    # This is just to show you how to re-order if needed
    # Typically you should put them in the order you like straigh away
    # Reordering...
    results_df = results_df[["coeff", "tvals", "pvals", "conf_lower",
                             "conf_higher"]]

    return results_df

# -----------------------------------------------------------------------------

def print_statement(print_statement, print_line_length = 90, print_line_start = 5):
    print(print_line_start * '#' + ' ' + print_statement + ' ' +
          (print_line_length - len(print_statement) - print_line_start - 2) * '#')
    
# for the figures
figure_folder = "../figures/"
if not os.path.exists(figure_folder):
    os.makedirs(figure_folder)
# for the latex document
report_folder = "../report/"
if not os.path.exists(report_folder):
    os.makedirs(report_folder)

# set the folder for reporting
report_folder = '../report/'
# folder for figures
figure_folder = '../figures/'

# first birthday
bd_1 = 703

rng = np.random.default_rng(bd_1)

# setting for output printing
print_statement('Q1: Descriptive Statistics')

try:
    data_full = pd.read_csv('sleep_data.csv')
    print('Data loaded')
except:
    print('Data could not be loaded')
    
num_obs = 600

observations = rng.choice(len(data_full), num_obs ,
replace=False)

data = data_full.iloc[observations , :].copy()

d_stats = data.describe()

print(d_stats)

data_frame_to_latex_table_file(report_folder + 'describe.tex',
                               np.round(d_stats.T, 2))

print_statement('Q3: Regression model on total work')

# set the dependent variable to equal sleep
y = data['sleep']


x_q3 = data['totwrk']
X_q3 = sm.add_constant(x_q3)

# set-up the model
model_q3 = sm.OLS(y, X_q3)
# estimate the model
results_q3 = model_q3.fit()

# output to screen
print(results_q3.summary())

# after this you will have a 'dummy_model.tex' file in your report folder
summary_to_latex_table_file(report_folder + 'sleep_model.tex',
                            results_q3.summary())

print_statement('Q5 & 6: Add age and educ to the model')

# Add 'educ' and 'age' to the independent variables
x_q5 = data[['totwrk', 'educ', 'age']]
X_q5 = sm.add_constant(x_q5)  # Adds a constant term to the predictor

# Set-up the extended model
model_q5 = sm.OLS(y, X_q5)

# Estimate the model
results_q5 = model_q5.fit()

# Output to screen
print(results_q5.summary())

# Save the model summary to a LaTeX file if needed
summary_to_latex_table_file(report_folder + 'extended_sleep_model_Q5.tex', results_q5.summary())

print_statement('Q10: Add agesq and yngkid + re estimation of the model')

x_q10 = data[['totwrk', 'educ', 'age', 'agesq', 'yngkid']]
X_q10 = sm.add_constant(x_q10)  # Adds a constant term to the predictor

# Set-up the extended model
model_q10 = sm.OLS(y, X_q10)

# Estimate the model
results_q10 = model_q10.fit()

# Output to screen
print(results_q10.summary())

# Save the model summary to a LaTeX file if needed
summary_to_latex_table_file(report_folder + 'extended_sleep_model_Q10.tex', results_q10.summary())

print_statement('Q11: Model for men and women')

# Separate the data for males
data_males = data[data['male'] == 1]

# Separate the data for females
data_females = data[data['male'] == 0]

# Set up and estimate the model for males
x_males = data_males[['totwrk', 'educ', 'age', 'agesq', 'yngkid']]
X_males = sm.add_constant(x_males)

model_males = sm.OLS(data_males['sleep'], X_males)
results_males = model_males.fit()

print("Regression Results for Males:")
print(results_males.summary())

# Save the model summary to a LaTeX file if needed
summary_to_latex_table_file(report_folder + 'male_sleep_model_Q11.tex', results_males.summary())

# Set up and estimate the model for females
x_females = data_females[['totwrk', 'educ', 'age', 'agesq', 'yngkid']]
X_females = sm.add_constant(x_females)

model_females = sm.OLS(data_females['sleep'], X_females)
results_females = model_females.fit()

print("Regression Results for Females:")
print(results_females.summary())

# Save the model summary to a LaTeX file if needed
summary_to_latex_table_file(report_folder + 'female_sleep_model_Q11.tex', results_females.summary())

print_statement('Q12: Model using male dummy variable ')

x_q12 = data[['totwrk', 'educ', 'age', 'agesq', 'yngkid', 'male']]
X_q12 = sm.add_constant(x_q12)  # Adds a constant term to the predictor

# Set-up the extended model
model_q12 = sm.OLS(y, X_q12)

# Estimate the model
results_q12 = model_q12.fit()

# Output to screen
print(results_q12.summary())

# Save the model summary to a LaTeX file if needed
summary_to_latex_table_file(report_folder + 'sleep_model_Q12.tex', results_q12.summary())

print_statement('Q15: Adding new variables to the model')

#south has variable has extreme effect on sleep
x_q15 = data[['totwrk', 'educ', 'age', 'agesq', 'yngkid', 'male','south']]
X_q15 = sm.add_constant(x_q15)  # Adds a constant term to the predictor

# Set-up the extended model
model_q15 = sm.OLS(y, X_q15)

# Estimate the model
results_q15 = model_q15.fit()

# Output to screen
print(results_q15.summary())

# Save the model summary to a LaTeX file if needed
summary_to_latex_table_file(report_folder + 'sleep_model_Q15.tex', results_q15.summary())