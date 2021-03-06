""" Assignment 2 """

# libraries  - general
import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd

# libraries - statistics
from scipy import interpolate
from scipy import stats
from scipy.stats import kurtosis, skew
from scipy.stats import shapiro

# libraries - trying new things
import pycountry_convert as pc
from pycountry_convert import country_alpha2_to_continent_code, country_name_to_country_alpha2


""" Functions for Data Pre-Processing """

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
    '''
    Parameters
    ----------
    df : dataframe to be manipulated
    column : column of the dataframe to be manipulated

    Returns
    -------
    df : changes the column datatype to float

    '''
    df[column] = pd.to_numeric(df[column], 
                               downcast = "float")
    return df
                        
# loading in the datasets 
df = pd.read_csv('C:/Users/sjjby/Documents/Applied Data Science 1/Assignment 2/total co2.csv',
                 header = 2,    # it removes the top line of jargon as its not data
                 engine = "python",
                 dtype = "str")


""" CO2 Emissions Dataset and Pre-Processing """

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


""" Summary Statistics on SA Dataset """

# performng data pre-procssing
print(pre_processing(sa_df))

'''
there are no NA values in any of the columns/ rows so no need
for interpolation but will experiment using a new dataframe and
set some of those values to be NA.
this function contains the sub-functions; describe and info
'''


""" Shapiro-Wilk test for normality on SA CO2 dataset """

from scipy.stats import shapiro

# performing shapiro-wilk test
shapiro_test = (sa_df.groupby('Country Name')
                .apply(lambda x: pd.Series(shapiro(x), index = ['W', 'P'])))

# printing the results of the test to the console
print(shapiro_test)

'''
Countries accept null hypothesis: Guyana, Uruguay, and Colombia
(countries are normality distributed across the years)

Countries reject null hypothesis: the rest
(countries are not normality distributed across the years)
'''


""" Skewness and Kurtosis on SA Population dataset """

# performing shapiro-wilk test
skewness_test1 = (sa_df.groupby('Country Name')
                  .apply(lambda x: pd.Series(skew(x))))

kurtosis_test1 = (sa_df.groupby('Country Name')
                  .apply(lambda x: pd.Series(kurtosis(x))))

# printing the results of the test to the console
print(skewness_test1, '\n', 
      kurtosis_test1)


""" Boxplot to look for outliers in each COUNTRY (over the years) """

sa_box = sa_df.set_index('Country Name').transpose()

# resetting the index
sa_box.reset_index(level = 0,
                      inplace = True)

# the column had wrong name so have to rename
sa_box.rename(columns = {"index": "Year"},
                 inplace = True)

# plotting the boxplot
fig12 = sa_box.boxplot(rot = 45,
                       fontsize = 10,
                       grid = True,
                       color = 'purple')

# adding extra things for visualisation
plt.title('The CO2 emissions of South American Countries')
plt.ylabel('CO2 emissions')
plt.xlabel('Country')

# show the graph
plt.show()

# saving the figure
plt.savefig('C:/Users/sjjby/Documents/Applied Data Science 1/Assignment 2/figure12.png')

'''
from the boxplot there appears to be outliers above the boxplot
in the countries:
    - Bolivia
    - Guyana
    - Peru
    - Suriname 
    - Uruguay
this makes sense owing to the CO2 emissions data values being 
very close together and thus some values are 2 standard deviations above
the mean. Interesting no outliers below the boxplot.
'''


""" Population Dataset and Pre-Processing """

# reading in the new dataframe
df2 = pd.read_csv('C:/Users/sjjby/Documents/Applied Data Science 1/Assignment 2/total population.csv',
                  header = 2,
                  engine = "python",
                  dtype = "str")

# performing pre-processing and some basic statistics
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


""" Subsetting the data Population data to be only South American countries """

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


""" Summary Statistics on SA Dataset """
print(pre_processing(sa_df2))


""" Shapiro-Wilk test for normality on SA Population dataset """

# performing shapiro-wilk test
pop_shapiro_test = (sa_df2.groupby('Country Name')
                    .apply(lambda x: pd.Series(shapiro(x), index = ['W', 'P'])))

# printing the results of the test to the console
print(pop_shapiro_test)

'''
Countries accept null hypothesis: None
(countries are normality distributed across the years)

Countries reject null hypothesis: All
(countries are not normality distributed across the years)
'''


""" Skewness and Kurtosis on SA Population dataset """

# performing shapiro-wilk test
skewness_test2 = (sa_df2.groupby('Country Name')
                  .apply(lambda x: pd.Series(skew(x))))

kurtosis_test2 = (sa_df2.groupby('Country Name')
                  .apply(lambda x: pd.Series(kurtosis(x))))

# printing the results of the test to the console
print(skewness_test2, '\n', 
      kurtosis_test2)


""" Boxplot to look for outliers in each COUNTRY (over the years) """

# transposing the data
sa_pop_box = sa_df2.set_index('Country Name').transpose()

# resetting the index
sa_pop_box.reset_index(level = 0,
                       inplace = True)

# the column had wrong name so have to rename
sa_pop_box.rename(columns = {"index": "Year"},
                  inplace = True)

# plotting the boxplot
fig13 = sa_pop_box.boxplot(rot = 45,
                           fontsize = 10,
                           grid = True,
                           color = 'orange')

# adding extra things for visualisation
plt.title('The Population of South American Countries')
plt.ylabel('Population (billions)')
plt.xlabel('Country')

# show the graph
plt.show()

'''
from the boxplot there appears to only be outliers above the boxplot
in the country Guyana.
'''

# saving the boxplot
plt.savefig('C:/Users/sjjby/Documents/Applied Data Science 1/Assignment 2/figure13.png')



""" Double axis plot (trying something new) """

""" Data Manipulation of CO2 emissions"""

# copy original data
co2_df = df.copy()

# adding total rows and columns for plots
co2_df.loc['Column_Total']= co2_df.sum(numeric_only=True, axis=0)

# transposing data to make it easy
co2_df = co2_df.set_index('Country Name').transpose()
co2_df.reset_index(level = 0,
              inplace = True)
co2_df.rename(columns = {"index": "Year"},
         inplace = True)

# rename the NaN column
co2_df.rename(columns = {np.nan: "Column Total" }, 
         inplace = True)

# subset new dataframe 
co2_df = co2_df[['Year', 'Column Total']]


""" Data Manipulation of Population """

# copy original population data to add to the plot above
pop_df = df2.copy()

# adding total rows and columns for plots
pop_df.loc['Column_Total']= pop_df.sum(numeric_only=True, axis=0)
#co2_df.loc[:,'Row_Total'] = co2_df.sum(numeric_only=True, axis=1)

# transposing data to make it easy
pop_df = pop_df.set_index('Country Name').transpose()

# resetting the index
pop_df.reset_index(level = 0,
                   inplace = True)

# renaming the column to access it 
pop_df.rename(columns = {"index": "Year"},
             inplace = True)

# rename the NaN column
pop_df.rename(columns = {np.nan: "Column Total" }, 
             inplace = True)

# subset new dataframe 
pop_df = pop_df[['Year', 'Column Total']]


""" Total CO2 line plot """

# plotting the TOTAL co2 emissions through the years
fig1, axs1 = plt.subplots(figsize = (20, 10))
axs1.plot(co2_df['Year'], co2_df['Column Total'],
         color = 'magenta', 
         marker = 'o', 
         linestyle = 'dashed',
         linewidth = 2, 
         markersize = 12,
         label = 'World CO2 emissions')

# adding extra items to the plot
plt.xticks(rotation = 90)
plt.xlabel('Years',
           fontsize = 20)
plt.ylabel('CO2 emissions (kt in 100 million)',
           fontsize = 20)
plt.legend(loc = 'upper left',
           fontsize = 15)

# twinning the axs to make two plots on one figure (CO2 emissions and pop)
axs11 = axs1.twinx()


""" Total Population line plot """

# plotting the TOTAL pop emissions through the years
axs11.plot(pop_df['Year'], pop_df['Column Total'],
           color = 'orange', 
           marker = 'o', 
           linestyle = 'dashed',
           linewidth = 2, 
           markersize = 12,
           label = 'World Population')

# adding extra things to the plot for better visualisation
plt.xticks(rotation = 90)
plt.xlabel('Years',
           fontsize = 20)
plt.ylabel('Population (in billions)',
           fontsize = 20)
plt.legend(loc = 'lower right',
           fontsize = 15)

# show the figure with two different y-axis
plt.show()

# saving the figure
plt.savefig('C:/Users/sjjby/Documents/Applied Data Science 1/Assignment 2/figure11.png')


""" Correlation between WORLD CO2 and Pop datasets overtime """

# using the .corrwith function to do the whole of the subsetted dataframe
print('WORLD: correlation between population and CO2 emissions')
print(co2_df.corrwith(pop_df))



""" Trying something new for a map of south america CO2 emissions (1980 and 2018)"""

# copying south american dataset
sa_df2 = sa_df.copy()

# renaming the a column so it is easier to manipulate
sa_df2.rename(columns = {'Country Name': 'country'},
          inplace = True)

#function to convert to alpah2 country codes
def cont_convert(row):
    # convert country name to country code 
    for country in row.country:
        try:
            cn_code = pc.country_name_to_country_alpha2(row.country, 
                                                        cn_name_format = "default")
        # if an error occurs (prevention of code not running)
        except KeyError:
            pass
    return cn_code

# applying function to dataframe
sa_df2['code'] = sa_df2.apply(cont_convert, 
                                      axis = 1)
# making sure it has worked
print(sa_df2.tail())

# libraries for specific method
import plotly
import plotly.express as px

# create figure for the newest measurements
fig2 = px.choropleth(sa_df2,
                    locations = 'country',
                    locationmode = "country names",
                    color = '2018', 
                    scope = "south america")

# plotly is browser based so had to write as a html to view image
fig2.write_html("C:/Users/sjjby/Documents/Applied Data Science 1/Assignment 2/figure2.html")

# create figure for the oldest measurements
fig3 = px.choropleth(sa_df2,
                    locations = 'country',
                    locationmode = "country names",
                    color = '1960', 
                    scope = "south america")

# plotly is browser based so had to write as a html to view image
fig3.write_html("C:/Users/sjjby/Documents/Applied Data Science 1/Assignment 2/figure3.html")

'''
these two figures when looked at using the HTML are interactive,
whereby using the mouse and hovering over a particular 
south american country the map displays the country name and 
the actual CO2 emissions of the respective country.  However,
this cannot be observed within the A4 report.
'''


""" Correlation between 1980 and 2018 for CO2 emissions """

#adding row total into the table
sa_df2.loc[:,'Row_Total'] = sa_df2.sum(numeric_only=True, axis=1)

co2_1980 = sa_df2['1980']
print("CO2 emissions in 1980 =", '\n', co2_1980)
co2_2018 = sa_df2['2018']
print("CO2 emissions in 2018 =", '\n', co2_2018)

print('CO2 EMISSIONS: Correlation between 1980 and 2018 in SA countries')
a, b = stats.pearsonr(co2_1980, co2_2018)
print(f'pearson rank test: r={a}, p={b}')
c, d = stats.spearmanr(co2_1980, co2_2018)
print(f'spearman rank test: r={c}, p={d}')



""" Stacked area plot of CO2 production """

# transposing data for ease of manipulation
sa_co2_df = sa_df.set_index('Country Name').transpose()

# resetting the index
sa_co2_df.reset_index(level = 0,
                      inplace = True)

# the column had wrong name so have to rename
sa_co2_df.rename(columns = {"index": "Year"},
                 inplace = True)

# plot
fig4 = sa_co2_df.plot(kind = 'area',
                 legend = True,
                 figsize = (15, 8),
                 colormap = 'plasma')
axs4 = plt.subplot()

# adding on extra things to the plot
plt.xticks(range(0,len(sa_co2_df.Year.values)), sa_co2_df.Year.values)
plt.xticks(rotation = 90)
plt.xlabel('Year',
           fontsize = 20)
plt.ylabel('CO2 Emissions (kt per million)',
           fontsize = 20)
plt.legend(fontsize = 15)

# show the graph
plt.show()

# saving the figure
plt.savefig('C:/Users/sjjby/Documents/Applied Data Science 1/Assignment 2/figure4.png')


""" Stacked area map of Population """

# transposing data for ease of manipulation
sa_pop_df = sa_df2.set_index('Country Name').transpose()
sa_pop_df.reset_index(level = 0,
                      inplace = True)
sa_pop_df.rename(columns = {"index": "Year"},
                 inplace = True)

labels = sa_co2_df['Year']

# define plot and axis 
fig5 = sa_pop_df.plot(kind = 'area',
                 legend = True,
                 figsize = (15, 8),
                 colormap = 'plasma')
axs5 = plt.subplot()

# adding extra things for visualisation
plt.xticks(range(0,len(sa_pop_df.Year.values)), sa_pop_df.Year.values)
plt.xticks(rotation = 90)
plt.xlabel('Year',
           fontsize = 20)
plt.ylabel('Population (per 100 million)',
           fontsize = 20)
plt.legend(fontsize = 15)

# show the graph
plt.show()

# saving the figure
plt.savefig('C:/Users/sjjby/Documents/Applied Data Science 1/Assignment 2/figure5.png')


""" Line Plot of CO2 Production """

# copying CO2 dataset so not override previous versions
sa_df3 = sa_df.copy()

# transposing the data
sa_df3 = sa_df3.set_index('Country Name').transpose()

# resetting the index
sa_df3.reset_index(level = 0,
                   inplace = True)

# the first column had no name so now naming it to access it
sa_df3.rename(columns = {"index": "Year"},
              inplace = True)

# defining plot 
fig6 = sa_df3.plot(kind = 'line',
                   legend = True,
                   figsize = (15, 8),
                   colormap = 'plasma')

# adding extra items for visualisation
plt.xticks(range(0,len(sa_df3.Year.values)), sa_df3.Year.values)
plt.xticks(rotation = 90)
plt.xlabel('Year',
           fontsize = 20)
plt.ylabel('CO2 Emissions (kt)',
           fontsize = 20)
plt.legend(fontsize = 15)

# show the plot
plt.show()

# saving the figure
plt.savefig('C:/Users/sjjby/Documents/Applied Data Science 1/Assignment 2/figure6.png')


""" Line Plot of Population """

# define new plot             
fig7 = sa_pop_df.plot(kind = 'line',  
                      linestyle = 'dashed',
                      legend = True,
                      figsize = (15, 8),
                      colormap = 'plasma',
                      layout = 'tight')

# adding extra things to fig7 for visualisation
plt.xticks(range(0,len(sa_pop_df.Year.values)), sa_pop_df.Year.values)
plt.xticks(rotation = 90)
plt.xlabel('Year',
           fontsize = 20)
plt.ylabel('Population (per 100 million)',
           fontsize = 20)
plt.legend(fontsize = 15)

# show the graph
plt.show()

# saving the figure
plt.savefig('C:/Users/sjjby/Documents/Applied Data Science 1/Assignment 2/figure7.png')



""""   Correlation Analysis between Indicators    """


""" Correlation Figure on the two datasets (for visualisation) """

# plot correlation matrix of SA CO2 dataframe
plt.figure(figsize = (15, 8))
plt.matshow(sa_co2_df.corr())
plt.title("Correlation between South American CO2 emissions")
print("Correlation =", '\n', sa_co2_df.corr())

#plotting colorbar
cb = plt.colorbar()
cb.ax.tick_params(labelsize = 14)

# showing the plot
plt.show()

# saving the figure
plt.savefig('C:/Users/sjjby/Documents/Applied Data Science 1/Assignment 2/figure8.png')



# plot correlation matrix of SA Population dataframe
plt.figure(figsize = (15, 8))
plt.matshow(sa_pop_df.corr())
plt.title("Correlation between South American Population")
print("Correlation =", '\n', sa_co2_df.corr())

#plotting colorbar
cb = plt.colorbar()
cb.ax.tick_params(labelsize = 14)

plt.show()

# saving the figure
plt.savefig('C:/Users/sjjby/Documents/Applied Data Science 1/Assignment 2/figure9.png')



""" Manipulation of CO2 emission """

#Identifying rows to use 
arg_row = df.loc[df['Country Name'] == 'Argentina']
bra_row = df.loc[df['Country Name'] == 'Brazil']
per_row = df.loc[df['Country Name'] == 'Peru']
uru_row = df.loc[df['Country Name'] == 'Uruguay']
bol_row = df.loc[df['Country Name'] == 'Bolivia']
chil_row = df.loc[df['Country Name'] == 'Chile']
col_row = df.loc[df['Country Name'] == 'Colombia']
ven_row = df.loc[df['Country Name'] == 'Venezuela']
ecu_row = df.loc[df['Country Name'] == 'Ecuador']
guy_row = df.loc[df['Country Name'] == 'Guyana']
par_row = df.loc[df['Country Name'] == 'Paraguay']
sur_row = df.loc[df['Country Name'] == 'Suriname']

'''
the following code for CO2 emissions and Population follow:
    1. country_values = identifying the cells in the row 
    (values of each variable)
    2. country_list = making the values into a numpy list 
    3. country_list = flatterning the values into a 1D array
'''

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

bol_values = bol_row.iloc[0:, 1:]
bol_list = np.array(bol_values.values.tolist())
bol_list = np.ndarray.flatten(bol_list)

chil_values = chil_row.iloc[0:, 1:]
chil_list = np.array(chil_values.values.tolist())
chil_list = np.ndarray.flatten(chil_list)

col_values = col_row.iloc[0:, 1:]
col_list = np.array(col_values.values.tolist())
col_list = np.ndarray.flatten(col_list) 

ven_values = ven_row.iloc[0:, 1:]
ven_list = np.array(ven_values.values.tolist())
ven_list = np.ndarray.flatten(ven_list) 

par_values = par_row.iloc[0:, 1:]
par_list = np.array(par_values.values.tolist())
par_list = np.ndarray.flatten(par_list)

guy_values = guy_row.iloc[0:, 1:]
guy_list = np.array(guy_values.values.tolist())
guy_list = np.ndarray.flatten(guy_list)

sur_values = sur_row.iloc[0:, 1:]
sur_list = np.array(sur_values.values.tolist())
sur_list = np.ndarray.flatten(sur_list) 

ecu_values = ecu_row.iloc[0:, 1:]
ecu_list = np.array(ecu_values.values.tolist())
ecu_list = np.ndarray.flatten(ecu_list) 


""" Manipulation of Population dataframe """

arg_row2 = df2.loc[df['Country Name'] == 'Argentina']
bra_row2 = df2.loc[df['Country Name'] == 'Brazil']
per_row2 = df2.loc[df['Country Name'] == 'Peru']
uru_row2 = df2.loc[df['Country Name'] == 'Uruguay']
bol_row2 = df2.loc[df['Country Name'] == 'Bolivia']
chil_row2 = df2.loc[df['Country Name'] == 'Chile']
col_row2 = df2.loc[df['Country Name'] == 'Colombia']
ven_row2 = df2.loc[df['Country Name'] == 'Venezuela']
ecu_row2 = df2.loc[df['Country Name'] == 'Ecuador']
guy_row2 = df2.loc[df['Country Name'] == 'Guyana']
par_row2 = df2.loc[df['Country Name'] == 'Paraguay']
sur_row2 = df2.loc[df['Country Name'] == 'Suriname']

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

bol_values2 = bol_row2.iloc[0:, 1:]
bol_list2 = np.array(bol_values2.values.tolist())
bol_list2 = np.ndarray.flatten(bol_list2)

chil_values2 = chil_row2.iloc[0:, 1:]
chil_list2 = np.array(chil_values2.values.tolist())
chil_list2 = np.ndarray.flatten(chil_list2)

col_values2 = col_row2.iloc[0:, 1:]
col_list2 = np.array(col_values2.values.tolist())
col_list2 = np.ndarray.flatten(col_list2) 

ven_values2 = ven_row2.iloc[0:, 1:]
ven_list2 = np.array(ven_values2.values.tolist())
ven_list2 = np.ndarray.flatten(ven_list2) 

par_values2 = par_row2.iloc[0:, 1:]
par_list2 = np.array(par_values2.values.tolist())
par_list2 = np.ndarray.flatten(par_list2)

guy_values2 = guy_row2.iloc[0:, 1:]
guy_list2 = np.array(guy_values2.values.tolist())
guy_list2 = np.ndarray.flatten(guy_list2)

sur_values2 = sur_row2.iloc[0:, 1:]
sur_list2 = np.array(sur_values2.values.tolist())
sur_list2 = np.ndarray.flatten(sur_list2) 

ecu_values2 = ecu_row2.iloc[0:, 1:]
ecu_list2 = np.array(ecu_values2.values.tolist())
ecu_list2 = np.ndarray.flatten(ecu_list2) 


""" Correlation for the countries in South America """

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

print('URUGUAY: correlation between population and CO2 emissions')
a, b = stats.pearsonr(uru_list, uru_list2)
print(f'pearson rank test: r={a}, p={b}')
c, d = stats.spearmanr(uru_list, uru_list2)
print(f'spearman rank test: r={c}, p={c}')

print('BOLIVIA: correlation between population and CO2 emissions')
a, b = stats.pearsonr(bol_list, bol_list2)
print(f'pearson rank test: r={a}, p={b}')
c, d = stats.spearmanr(bol_list, bol_list2)
print(f'spearman rank test: r={c}, p={d}')

print('CHILE: correlation between population and CO2 emissions')
a, b = stats.pearsonr(chil_list, chil_list2)
print(f'pearson rank test: r={a}, p={b}')
c, d = stats.spearmanr(chil_list, chil_list2)
print(f'spearman rank test: r={c}, p={d}')

print('COLOMBIA: correlation between population and CO2 emissions')
a, b = stats.pearsonr(col_list, col_list2)
print(f'pearson rank test: r={a}, p={b}')
c, d = stats.spearmanr(col_list, col_list2)
print(f'spearman rank test: r={c}, p={c}')

print('VENEZUELA: correlation between population and CO2 emissions')
a, b = stats.pearsonr(ven_list, ven_list2)
print(f'pearson rank test: r={a}, p={b}')
c, d = stats.spearmanr(ven_list, ven_list2)
print(f'spearman rank test: r={c}, p={d}')

print('ECUADOR: correlation between population and CO2 emissions')
a, b = stats.pearsonr(ecu_list, ecu_list2)
print(f'pearson rank test: r={a}, p={b}')
c, d = stats.spearmanr(ecu_list, ecu_list2)
print(f'spearman rank test: r={c}, p={d}')

print('GUYANA: correlation between population and CO2 emissions')
a, b = stats.pearsonr(guy_list, guy_list2)
print(f'pearson rank test: r={a}, p={b}')
c, d = stats.spearmanr(guy_list, guy_list2)
print(f'spearman rank test: r={c}, p={c}')

print('PARAGUAY: correlation between population and CO2 emissions')
a, b = stats.pearsonr(par_list, par_list2)
print(f'pearson rank test: r={a}, p={b}')
c, d = stats.spearmanr(par_list, par_list2)
print(f'spearman rank test: r={c}, p={d}')

print('SURINAME: correlation between population and CO2 emissions')
a, b = stats.pearsonr(sur_list, sur_list2)
print(f'pearson rank test: r={a}, p={b}')
c, d = stats.spearmanr(sur_list, sur_list2)
print(f'spearman rank test: r={c}, p={c}')



""" Experimenting with Interpolation """

'''
why? because the data I selected unfortunately did not have
any NA/ missing values therefore, I still wanted to ensure
I knew how to perform interpolation and thus, selected a 
new dataset that I observed to have several NA values.
'''

df3 = pd.read_csv('C:/Users/sjjby/Documents/Applied Data Science 1/Assignment 2/forest area.csv',
                  header = 4,    # it removes the top 4 lines of jargon as its not data
                  engine = "python")

# renaming a column
df3.loc[254, 'Country Name'] = "Venezuela"

# removing unwanted columns
df3.drop(['Country Code', 'Indicator Name', 'Indicator Code'],
        axis = 1,
        inplace = True)

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


""" Summary statistics of the Forest Area Dataset """

# performing pre-processing and basic statistics using functions
#print(pre_processing(sa_for_df))


""" Shapiro-Wilk test for normality on SA Forest Area dataset """

# copying the dataset
sa_for_df1 = sa_df3.copy()
print(sa_for_df1.head())

from scipy.stats import shapiro

# performing shapiro-wilk test
shapiro_test3 = (sa_for_df1.groupby('Country Name')
                 .apply(lambda x: pd.Series(shapiro(x), index = ['W', 'P'])))

# printing the results of the test to the console
print(shapiro_test3)

'''
Countries accept null hypothesis: 
    Bolivia, Colombia, Ecuador, Paraguay, Peru, 
    Uruguay, and Venezuela
(countries are normality distributed across the years)

Countries reject null hypothesis: 
    Argentina, Brazil, Chile, Guyana, and Suriname
(countries are not normality distributed across the years)
'''


""" Skewness and Kurtosis on SA Population dataset """

# performing shapiro-wilk test
skewness_test3 = (sa_for_df1.groupby('Country Name')
                  .apply(lambda x: pd.Series(skew(x))))

kurtosis_test3 = (sa_for_df1.groupby('Country Name')
                  .apply(lambda x: pd.Series(kurtosis(x))))

# printing the results of the test to the console
print(skewness_test3, '\n', 
      kurtosis_test3)


""" Boxplot to look for outliers in each COUNTRY (over the years) """

sa_for_box = sa_for_df1.set_index('Country Name').transpose()

# resetting the index
sa_for_box.reset_index(level = 0,
                       inplace = True)

# the column had wrong name so have to rename
sa_for_box.rename(columns = {"index": "Year"},
                  inplace = True)

# plotting the boxplot
fig14 = sa_for_box.boxplot(rot = 45,
                           fontsize = 10,
                           grid = True,
                           color = 'navy')

# adding extra things for visualisation
plt.title('The Forest Area of South American Countries')
plt.ylabel('Forest Area (in millions)')
plt.xlabel('Country')

# show the graph
plt.show()

# saving the figure
plt.savefig('C:/Users/sjjby/Documents/Applied Data Science 1/Assignment 2/figure14.png')

'''
from the boxplot there appears to have no outliers above 
or below the boxplots for the countries.
'''

""" Interpolation Manipulation """

# transposing data for ease of manipulation
sa_for_df = sa_df3.set_index('Country Name').transpose()

# resetting the matrix
sa_for_df.reset_index(level = 0,
                      inplace = True)

# had no name so now naming it
sa_for_df.rename(columns = {"index": "Year"},
                 inplace = True)

# copying the original data
sa_for_df2 = sa_for_df.copy()

# dropping any NA values
sa_for_df2.dropna(axis = 0,
                 inplace = True)

# changing some values to NA in the dataset
sa_for_df2['Argentina'].replace(['348388', '333780', '317960', '299906', '290970', '288990'], np.NaN,
                                inplace = True)

# copying the data for subplots
sa_for_df3 = sa_for_df.copy()
sa_for_df4 = sa_for_df.copy()
sa_for_df5 = sa_for_df.copy()

# performing interpolation on the SA country: Argentina
sa_for_df3['Argentina'].interpolate(method = 'linear',
                                   limit_direction = 'backward',
                                   inplace = True)
sa_for_df4['Argentina'].interpolate(method = 'nearest',
                                    limit_direction = 'backward',
                                    inplace = True)
sa_for_df5['Argentina'].interpolate(method = 'cubic',
                                    limit_direction = 'backward',
                                    inplace = True)

# defining figure and axis for plotting
fig10, ax10 = plt.subplots(2,2,
                         figsize = (16, 8),
                         sharey = True,
                         sharex = True)

# plotting using subplots
ax10[0,0].plot(sa_for_df2['Year'],
              sa_for_df2['Argentina'],
              color = 'purple')
ax10[0,1].plot(sa_for_df3['Year'],
              sa_for_df3['Argentina'], 
              color = 'pink')
ax10[1,0].plot(sa_for_df4['Year'],
              sa_for_df4['Argentina'], 
              color = 'green')
ax10[1,1].plot(sa_for_df5['Year'],
              sa_for_df5['Argentina'], 
              color = 'orange')

# setting extras for visualisation of figure 10
ax10[0,0].set_ylabel('Forest Area')
ax10[1,0].set_ylabel('Forest Area')

# rotating the x axis
ax10[1,0].tick_params(axis = 'x',
                labelrotation = 90)
ax10[1,1].tick_params(axis = 'x',
                     labelrotation = 90)

# set titles 
ax10[0,0].set_title('Original Data')
ax10[0,1].set_title('Linear Interpolation')
ax10[1,0].set_title('K-Nearest Neighbour Interpolation')
ax10[1,1].set_title('Cubic Interpolation')

# showing the figure
plt.show()

'''
cubic interpolation appears to fit the data the best. 
Therefore, if i was using this data any further I would 
apply the cubic interpolation for analysis/ comparative studies.
'''

# saving the figure
plt.savefig('C:/Users/sjjby/Documents/Applied Data Science 1/Assignment 2/figure10.png')


""" The END """
