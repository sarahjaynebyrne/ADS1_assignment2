## ASSIGNMENT 2 ##

# importing libraries 
import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd
import seaborn as sns

# data pre-processing 
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

# loading in dataset from url
wb_df = pd.read_csv('C:/Users/sjjby/Documents/Applied Data Science 1/Assignment 2/worldbank_climatechange.csv')

# performing pre-processing on dataset
print(pre_processing(wb_df))

# displaying the variables in the column 'Indicator Name'
for col in wb_df:
    print(wb_df['Indicator Name'].unique())
    
# sorting the values by the indicator name
print(wb_df.sort_values('Indicator Name'))

# resetting the index of the table
print(wb_df.reset_index)

# saving dataset tp CSV file
#wb_df.to_csv('C:/Users/sjjby/Documents/Applied Data Science 1/Assignment 2/world_bank.csv', 
#             index = False)

###############################################################################

'''
Exploring the statistical properties of a few indicators and 
cross-compare between individual countries and the whole world
'''






###############################################################################

'''
Exploring any correlations between indicators:
    - does this vary between country 
    - have any correlations/ trends changed with time
'''
'''
# total correlations between each variable using pearson

x = wb_df.corr(method = 'pearson')
print(x)
'''

# showing only the GBR variables
mask = wb_df['Country Code'] == 'GBR'

# dataframe WITHOUT GBR variables
df2 = wb_df[~mask]
print('Dataframe 2 =', '\n', df2)

# dataframe WITH GBR variables 
GBR_df = wb_df[mask]
print('Dataframe 1 =', '\n', GBR_df)

GBR_df1 = GBR_df.copy()
print(GBR_df1)

# removing the useless columns
GBR_df1.drop(columns = ['Country Name', 'Indicator Code', 'Country Code'], 
            inplace = True)

# correlations 
print(GBR_df1)

'''
sns.lineplot(x = 'Year',
             y = 'Population, total',
             data = GBR_df1)
plt.xticks(rotation=15)
plt.title('Hello baby')


GBR_corr = GBR_df1.corr(method = 'pearson')
print(GBR_corr)


# plotting graph
fig, axs = plt.subplots(nrows = 1, ncols = 2, figsize = (20, 10))

axs[0].scatter(df_world[''], 
               df_world['Total Cases'],
               marker = '*',
               color = 'green')
axs[0].set(xlabel = 'Total Recovered',
        ylabel = 'Total Cases',
        title = 'Pearsons Correlation')
'''

# time series
dates = [1960,
         1961,
         1962,
         1963,
         1964,
         1965,
         1966,
         1967,
         1968,
         1969,
         1970,
         1971,
         1972,
         1973,
         1974,
         1975,
         1976,
         1977,
         1978,
         1979,
         1980,
         1981,
         1982,
         1983,
         1984,
         1985,
         1986,
         1987,
         1988,
         1989,
         1990,
         1991,
         1992,
         1993,
         1994,
         1995,
         1996,
         1997,
         1998,
         1999,
         2000,
         2001,
         2002,
         2003,
         2004,
         2005,
         2006,
         2007,
         2008,
         2009,
         2010,
         2011,
         2012,
         2013,
         2014,
         2015,
         2016,
         2017,
         2018,
         2019,
         2020]



'''
sns.lineplot(x = dates,
             y = 'Population, total',
             data = GBR_df1)
plt.xticks(rotation=15)
plt.title('Hello baby')
'''
# CO2 production and total population






# Time Series 














