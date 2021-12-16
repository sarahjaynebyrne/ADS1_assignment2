## assignment 2.1

# libraries 
import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd
import seaborn as sns

# data pre-processing function 
def pre_processing(x):
    '''
    Parameters
    ----------
    x : dataset, performing functions described below on the dataset
    Returns
    -------
    the column values with ',' replaced and 
    stored as an int value
    '''
    head = x.head()             # displaying the first 5 lines in the data
    tail = x.tail()             # displaying the last 5 lines in the data
    columns = x.columns         # name of columns in dataset
    describe = x.describe       # general statistics on the dataset 
    info = x.info               # general statistics on the dataset 
    null = x.isna().sum()       # any nan values in columns of dataset
    dtype = x.dtypes            # data types of the columns in the dataset
    index = x.index             # the row identifirs
    
    return (f'The top 5 columns in the dataset = \n {head} \n \
            The bottom 5 columns in the dataset = \n {tail} \n \
            The name of the columns in the dataset = \n {columns} \n \
            The statistic description of the dataset = \n {describe} \n \
            The information on the dataset = \n {info} \n \
            The presence of any NA values = \n {null} \n \
            The datatype of the columns in the dataset = \n {dtype} \n \
            The index of the dataset = \n {index}') 

def data_load(x):
    x = input()     # user would input the location of the file e.g. 'C:/Users/sjjby/Documents/Applied Data Science 1/Assignment 2/total co2.csv'
    df = pd.read(x,
                 header = 2,
                 engine = "python",
                 dtype = "str")
    return df

def clean_convert(df, column):
    df[column] = pd.to_numeric(df[column], 
                               downcast = "float")
    return df
                        
# loading in the datasets 
df = pd.read_csv('C:/Users/sjjby/Documents/Applied Data Science 1/Assignment 2/total co2.csv',
                 header = 2,    # it removes the top line of jargon as its not data
                 engine = "python",
                 dtype = "str")

# performing pre-processing
print(pre_processing(df))

'''
all data are saved as objects so need to change them to being floats
'''
for col in df.columns[4:]:
    df = clean_convert(df, col)
print(df.dtypes)

# adding total rows and columns for plots
df.loc['Column_Total']= df.sum(numeric_only=True, axis=0)
df.loc[:,'Row_Total'] = df.sum(numeric_only=True, axis=1)

'''
columns 2019, 2020 and 'unnamed: 65' have no values so dropping those
also dropping country code, Indicator Name, and indicator code
because i am not going to use them
'''
co2_df = df.copy()
co2_df.drop(labels = ['Country Code', 'Indicator Name', 'Indicator Code', '2019', '2020', 'Unnamed: 65'],
            axis = 1,   # column axis
            inplace = True)

# adding a totals row for each column
#print(co2_df.sum(axis = 0, 
#                 skipna = True))    # skips NA values so still calculates the sum


df2 = pd.read_csv('C:/Users/sjjby/Documents/Applied Data Science 1/Assignment 2/total population.csv',
                  header = 2,
                  engine = "python",
                  dtype = "str")
print(df2)

# performing pre-processing
print(pre_processing(df2))

for col in df2.columns[4:]:
    df2 = clean_convert(df2, col)
print(df2.dtypes)

pop_df = df2.copy()
pop_df.drop(labels = ['Country Code', 'Indicator Name', 'Indicator Code', '2019', '2020', 'Unnamed: 65'],
            axis = 1,   # column axis
            inplace = True)
print(pop_df)

## CORRELATION 

print(pop_df.corrwith(co2_df,
                      axis = 1,     # correlated with the countries
                      method = 'pearson'))

## PLOTS


# time-series 
'''
def plot_df(df, x, y, title="", xlabel='Date', ylabel='Value', dpi=100):
    plt.figure(figsize=(16,5), dpi=dpi)
    plt.plot(x, y, color='tab:red')
    plt.gca().set(title=title, xlabel=xlabel, ylabel=ylabel)
    plt.show()

plot_df(co2_df, 
        x = co2_df., 
        y = co2_df, 
        title='Hey bitch do you wanna go HARD') 
'''


# world map (co2 and population)



## PREDICTIVE MODELLING for CO2 production ???






















           