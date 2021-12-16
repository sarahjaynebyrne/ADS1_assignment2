""" Assignment 2 """

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


""" Identifying rows to use for south america analysis"""

sa_df = df[(df['Country Name'] == 'Argentina') |
         (df['Country Name'] == 'Bolivia') |
         (df['Country Name'] == 'Brazil') |
         (df['Country Name'] == 'Chile') |
         (df['Country Name'] == 'Colombia') |
         (df['Country Name'] == 'Ecuador') |
         (df['Country Name'] == 'Guyana') |
         (df['Country Name'] == 'Paraguay') |
         (df['Country Name'] == 'Peru') |
         (df['Country Name'] == 'Suriname') |
         (df['Country Name'] == 'Uruguay') |
         (df['Country Name'] == 'Venezuela')]

# resetting the index 
sa_df.reset_index(inplace = True,
                  drop = True)

# performng data pre-procssing
print(pre_processing(sa_df))

'''
there are no NA values in any of the columns/ rows so no need
for interpolation
'''

""" Map of South America """

'''
# loading in the map
with plt.cbook.get_sample_data('C:/Users/sjjby/Documents/Applied Data Science 1/Assignment 2/south america map.jpg') as image_file:
    image = plt.imread(image_file)

# define figure 
fig, axs = plt.subplots()

# load image onto the axes
axs.imshow(image)
'''

""" Trying something new for a map of south america (1980 and 2018)"""

# copying south american dataset
sa_df2 = sa_df.copy()

sa_df2.rename(columns = {'Country Name': 'country'},
          inplace = True)

import pycountry_convert as pc

#function to convert to alpah2 country codes
from pycountry_convert import country_alpha2_to_continent_code, country_name_to_country_alpha2

def cont_convert(row):
    # convert country name to country code 
    for country in row.country:
        try:
            cn_code = pc.country_name_to_country_alpha2(row.country, 
                                                        cn_name_format = "default")
        except KeyError:
            pass
    # convert country code to continent code
    return cn_code

# applying function to dataframe
sa_df2['code'] = sa_df2.apply(cont_convert, 
                                      axis = 1)
print(sa_df2.tail())

import plotly
import plotly.express as px

# create figure for the newest measurements
fig = px.choropleth(sa_df2,
                    locations = 'country',
                    locationmode = "country names",
                    color = '2018', 
                    scope = "south america")
# plotly is browser based so had to write as a html to view image
fig.write_html("C:/Users/sjjby/Documents/Applied Data Science 1/Assignment 2/SouthAmericaMap_2018.html")

# create figure for the oldest measurements
fig = px.choropleth(sa_df2,
                    locations = 'country',
                    locationmode = "country names",
                    color = '1960', 
                    scope = "south america")
# plotly is browser based so had to write as a html to view image
fig.write_html("C:/Users/sjjby/Documents/Applied Data Science 1/Assignment 2/SouthAmericaMap_1960.html")


""" Stacked area plot of CO2 production """

# transposing data for ease of manipulation
sa_co2_df = sa_df.set_index('Country Name').transpose()
sa_co2_df.reset_index(level = 0,
                      inplace = True)
sa_co2_df.rename(columns = {"index": "Year"},
                 inplace = True)

# plot
fig = sa_co2_df.plot(kind = 'area',
                 legend = True,
                 figsize = (18, 15),
                 colormap = 'plasma')
axs = plt.subplot()

plt.xticks(range(0,len(sa_co2_df.Year.values)), sa_co2_df.Year.values)
plt.xticks(rotation = 90)
plt.xlabel('Year',
           fontsize = 20)
plt.ylabel('CO2 Emissions (kt per million)',
           fontsize = 20)
# show the graph
plt.show()

# adding row total to the end
#sa_df.loc[:,'Row_Total'] = sa_df.sum(numeric_only=True, axis=1)


""" Explore and understand any correlations between indicators """

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

# subsetting the data
sa_df2 = df2[(df2['Country Name'] == 'Argentina') |
                (df2['Country Name'] == 'Bolivia') |
                (df2['Country Name'] == 'Brazil') |
                (df2['Country Name'] == 'Chile') |
                (df2['Country Name'] == 'Colombia') |
                (df2['Country Name'] == 'Ecuador') |
                (df2['Country Name'] == 'Guyana') |
                (df2['Country Name'] == 'Paraguay') |
                (df2['Country Name'] == 'Peru') |
                (df2['Country Name'] == 'Suriname') |
                (df2['Country Name'] == 'Uruguay') |
                (df2['Country Name'] == 'Venezuela')]

# resetting the index 
sa_df2.reset_index(inplace = True,
                      drop = True)


""" Stacked map of Population """

# transposing data for ease of manipulation
sa_pop_df = sa_df2.set_index('Country Name').transpose()
sa_pop_df.reset_index(level = 0,
                      inplace = True)
sa_pop_df.rename(columns = {"index": "Year"},
                 inplace = True)

labels = sa_co2_df['Year']

# plot
fig = sa_pop_df.plot(kind = 'area',
                 legend = True,
                 figsize = (18, 15),
                 colormap = 'plasma')
axs = plt.subplot()

plt.xticks(range(0,len(sa_pop_df.Year.values)), sa_pop_df.Year.values)
plt.xticks(rotation = 90)
plt.xlabel('Year',
           fontsize = 20)
plt.ylabel('Population',
           fontsize = 20)
# show the graph
plt.show()


""" Line Plot of CO2 Production and Population """

# copying CO2 dataset so not override previous versions
sa_df3 = sa_df.copy()

sa_df3 = sa_df3.set_index('Country Name').transpose()
sa_df3.reset_index(level = 0,
                   inplace = True)
sa_df3.rename(columns = {"index": "Year"},
              inplace = True)

fig1 = sa_df3.plot(kind = 'line',
                   legend = True,
                   figsize = (18, 15),
                   colormap = 'plasma')
plt.xticks(range(0,len(sa_df3.Year.values)), sa_df3.Year.values)
plt.xticks(rotation = 90)
plt.xlabel('Year',
           fontsize = 20)
plt.ylabel('CO2 Emissions (kt)',
           fontsize = 20)

fig2 = sa_pop_df.plot(kind = 'line',  
                      linestyle = 'dashed',
                      legend = True,
                      figsize = (15, 15),
                      colormap = 'plasma',
                      layout = 'tight')
plt.xticks(range(0,len(sa_pop_df.Year.values)), sa_pop_df.Year.values)
plt.xticks(rotation = 90)
plt.xlabel('Year',
           fontsize = 20)
plt.ylabel('Population',
           fontsize = 20)
# show the graph
plt.show()


# save the graph
#fig1.savefig('C:/Users/sjjby/Documents/Applied Data Science 1/Assignment 2/lineplotCO2.pdf')


""" Correlation between Population and CO2 emissions in SA """












