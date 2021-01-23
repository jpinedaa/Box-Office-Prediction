import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('data/old/Movies_data_Merged_12-17.csv', sep=';', thousands=',')

data2 = pd.read_csv('data/movies_data_12-18.csv', sep=';', thousands=',')

data['Box_Office'] = pd.to_numeric(data['Box_Office'], errors='coerce')


#Director vs year vs box office

data_copy = data.copy().dropna()
director_box = data_copy.groupby(data_copy['directorsNames'])['Box_Office'].sum()
director_box_index = director_box.sort_values(ascending=False)[:20].index
director_box_pivot = pd.pivot_table(data = data_copy[data_copy['directorsNames'].isin(director_box_index)],index=['Year'], columns = ['directorsNames'], values= ['Box_Office'], aggfunc = 'sum')


fig, ax = plt.subplots()
sns.heatmap(director_box_pivot['Box_Office'],vmin = 0, annot= False, linewidth=.5, ax=ax)
plt.title('Director vs Year vs Box Office')
plt.ylabel('Year')

plt.show()

#Writer vs year vs Box Office

writers_box = data_copy.groupby(data_copy['writers'])['Box_Office'].sum()
writers_box_index = writers_box.sort_values(ascending=False)[:20].index
writers_box_pivot = pd.pivot_table(data = data_copy[data_copy['writers'].isin(writers_box_index)],index=['Year'], columns = ['writers'], values= ['Box_Office'], aggfunc = 'sum')

fig, ax = plt.subplots()
sns.heatmap(writers_box_pivot['Box_Office'],vmin = 0, annot= False, linewidth=.5, ax=ax)
plt.title('Writers vs Year vs Box Office')
plt.ylabel('Year')

plt.show()

# Year vs box office with budget and average rating

cmap = sns.cubehelix_palette(rot=-.2, as_cmap=True)
ax = sns.scatterplot(x="year", y="boxOffice",
                     hue="budget", size="averageRating",
                     palette=cmap, sizes=(10, 200),
                     data=data2)

plt.show()

# Genre vs box office and year

data2['topgenre'] = data2['genres'].apply(lambda x: x.split(',')[0].strip('[').strip(']').strip("'"))

sns.set(style="white")

sns.relplot(x="year", y="boxOffice", hue="topgenre", palette="muted",
            height=6, data=data2)
plt.title('Genre vs box office and year')
plt.xlabel('Year')
plt.ylabel('Box office')

plt.show()

# Genre vs box office/budget ratio and year

data2['bo/budget'] = data2['boxOffice'] / data2['budget']

sns.set(style="white")

sns.relplot(x="year", y="bo/budget", hue="topgenre", palette="muted",
            height=6, data=data2)
plt.title('Genre vs box office budget ratio and year')
plt.xlabel('Year')
plt.ylabel('Box office / budget')

plt.show()


# Box office vs budget, with and without outliers
plt.scatter(data2['boxOffice'], data2['budget'], color='black')
plt.title('Data with outliers')
plt.xlabel('Box office')
plt.ylabel('Budget')
plt.show()

data2 = data2.drop(data2[data2.boxOffice > 1500000000].index)
data2= data2.drop(data2[data2.budget > 300000000].index)
plt.scatter(data2['boxOffice'], data2['budget'], color='black')
plt.title('Data without outliers')
plt.xlabel('Box office')
plt.ylabel('Budget')
plt.show()