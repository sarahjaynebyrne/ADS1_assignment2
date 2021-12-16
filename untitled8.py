from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd
import seaborn as sns
from scipy import interpolate
from scipy import stats

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


""" Data Pre-Processing """

# performing pre-processing
print(pre_processing(df))

#all data are saved as objects so need to change them to being floats
for col in df.columns[4:]:
    df = clean_convert(df, col)
print(df.dtypes)

# removing unuseful columns
df.drop(['Country Code', 'Indicator Name', 'Indicator Code', '2019', '2020', 'Unnamed: 65'],
        axis = 1,
        inplace = True)

# renaming a column
df.loc[254, 'Country Name'] = "Venezuela"


""" Total CO2 line plot """

# copy original data
co2_df = df.copy()

# adding total rows and columns for plots
co2_df.loc['Column_Total']= co2_df.sum(numeric_only=True, axis=0)
#co2_df.loc[:,'Row_Total'] = co2_df.sum(numeric_only=True, axis=1)

# transposing data to make it easy
co2_df = co2_df.set_index('Country Name').transpose()
co2_df.reset_index(level = 0,
              inplace = True)
co2_df.rename(columns = {"index": "Year"},
         inplace = True)
print(co2_df.head())

# rename the NaN column
co2_df.rename(columns = {np.nan: "Column Total" }, 
         inplace = True)
print(co2_df.head())

# subset new dataframe 
co2_df = co2_df[['Year', 'Column Total']]
print(co2_df.head())

# plotting the TOTAL co2 emissions through the years
fig, axs = plt.subplots(figsize = (20, 10))
axs.plot(co2_df['Year'], co2_df['Column Total'],
         color = 'magenta', 
         marker = 'o', 
         linestyle = 'dashed',
         linewidth = 2, 
         markersize = 12)
plt.xticks(rotation = 90)
plt.xlabel('Years',
           fontsize = 15)
plt.ylabel('CO2 emissions (kt)',
           fontsize = 15)

""" Population dataframe """

df2 = pd.read_csv('C:/Users/sjjby/Documents/Applied Data Science 1/Assignment 2/total population.csv',
                  header = 2,
                  engine = "python",
                  dtype = "str")

# performing pre-processing
print(pre_processing(df2))

# ensuring the columns have the correct datatype now
for col in df2.columns[4:]:
    df2 = clean_convert(df2, col)

# dropping unuseful columns
df2.drop(labels = ['Country Code', 'Indicator Name', 'Indicator Code', '2019', '2020', 'Unnamed: 65'],
         axis = 1,   # column axis
         inplace = True)

# renaming a column
df.loc[254, 'Country Name'] = "Venezuela"







""" Correlation Analysis between Indicators """

""" Manipulation of CO2 emission """
#Identifying rows to use 
arg_row = df.loc[df['Country Name'] == 'Argentina']
bra_row = df.loc[df['Country Name'] == 'Brazil']
per_row = df.loc[df['Country Name'] == 'Peru']
uru_row = df.loc[df['Country Name'] == 'Uruguay']

arg_values = arg_row.iloc[0:, 1:]
arg_list = np.array(arg_values.values.tolist())
arg_list = np.ndarray.flatten(arg_list)

bra_values = bra_row.iloc[0:, 1:]
bra_list = np.array(bra_values.values.tolist())
bra_list = np.ndarray.flatten(bra_list)

per_values = per_row.iloc[0:, 1:]
per_list = np.array(per_values.values.tolist())
per_list = np.ndarray.flatten(per_list) 

uru_values = uru_row.iloc[0:, 1:]
uru_list = np.array(uru_values.values.tolist())
uru_list = np.ndarray.flatten(uru_list) 

""" Manipulation of Population dataframe """

arg_row2 = df2.loc[df['Country Name'] == 'Argentina']
bra_row2 = df2.loc[df['Country Name'] == 'Brazil']
per_row2 = df2.loc[df['Country Name'] == 'Peru']
uru_row2 = df2.loc[df['Country Name'] == 'Uruguay']

arg_values2 = arg_row2.iloc[0:, 1:]
arg_list2 = np.array(arg_values2.values.tolist())
arg_list2 = np.ndarray.flatten(arg_list2)

bra_values2 = bra_row2.iloc[0:, 1:]
bra_list2 = np.array(bra_values2.values.tolist())
bra_list2 = np.ndarray.flatten(bra_list2)

per_values2 = per_row2.iloc[0:, 1:]
per_list2 = np.array(per_values2.values.tolist())
per_list2 = np.ndarray.flatten(per_list2) 

uru_values2 = uru_row2.iloc[0:, 1:]
uru_list2 = np.array(uru_values2.values.tolist())
uru_list2 = np.ndarray.flatten(uru_list2)

""" Correlation for top 4 Developed countries in South America """

print('ARGENTINA: correlation between population and CO2 emissions')
a, b = stats.pearsonr(arg_list, arg_list2)
print(f'pearson rank test: r={a}, p={b}')
c, d = stats.spearmanr(arg_list, arg_list2)
print(f'spearman rank test: r={c}, p={d}')

print('BRAZIL: correlation between population and CO2 emissions')
a, b = stats.pearsonr(bra_list, bra_list2)
print(f'pearson rank test: r={a}, p={b}')
c, d = stats.spearmanr(bra_list, bra_list2)
print(f'spearman rank test: r={c}, p={d}')

print('PERU: correlation between population and CO2 emissions')
a, b = stats.pearsonr(per_list, per_list2)
print(f'pearson rank test: r={a}, p={b}')
c, d = stats.spearmanr(per_list, per_list2)
print(f'spearman rank test: r={c}, p={d}')

print('ARGENTINA: correlation between population and CO2 emissions')
a, b = stats.pearsonr(uru_list, uru_list2)
print(f'pearson rank test: r={a}, p={b}')
c, d = stats.spearmanr(uru_list, uru_list2)
print(f'spearman rank test: r={c}, p={c}')



'''

""" Experimenting with Interpolation """

from scipy.interpolate import interp1d
from sklearn.metrics import mean_squared_error

df3 = pd.read_csv('C:/Users/sjjby/Documents/Applied Data Science 1/Assignment 2/forest area.csv',
                  header = 2,    # it removes the top line of jargon as its not data
                  engine = "python",
                  dtype = "str")

print(pre_processing(df3))

# renaming a column
df.loc[254, 'Country Name'] = "Venezuela"

# subsetting the data
sa_df3 = df3[(df3['Country Name'] == 'Argentina') |
                (df3['Country Name'] == 'Bolivia') |
                (df3['Country Name'] == 'Brazil') |
                (df3['Country Name'] == 'Chile') |
                (df3['Country Name'] == 'Colombia') |
                (df3['Country Name'] == 'Ecuador') |
                (df3['Country Name'] == 'Guyana') |
                (df3['Country Name'] == 'Paraguay') |
                (df3['Country Name'] == 'Peru') |
                (df3['Country Name'] == 'Suriname') |
                (df3['Country Name'] == 'Uruguay') |
                (df3['Country Name'] == 'Venezuela')]

# resetting the index 
sa_df3.reset_index(inplace = True,
                      drop = True)

# transposing data for ease of manipulation
sa_for_df = sa_df3.set_index('Country Name').transpose()
sa_for_df.reset_index(level = 0,
                      inplace = True)
sa_for_df.rename(columns = {"index": "Year"},
                 inplace = True)


'''

