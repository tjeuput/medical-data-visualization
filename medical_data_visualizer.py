import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Import data
df = pd.read_csv('medical_examination.csv')
df['id'] = pd.to_numeric(df['id'], downcast='unsigned')
df['age'] = pd.to_numeric(df['age'], downcast='unsigned')
df['gender'] = pd.to_numeric(df['gender'], downcast='unsigned')
df['height'] = pd.to_numeric(df['height'], downcast='unsigned')
df['weight'] = pd.to_numeric(df['weight'], downcast='float')
df['ap_hi'] = pd.to_numeric(df['ap_hi'], downcast='unsigned')
df['ap_lo'] = pd.to_numeric(df['ap_lo'], downcast='unsigned')
df['cholesterol'] = pd.to_numeric(df['cholesterol'], downcast='unsigned')
df['gluc'] = pd.to_numeric(df['gluc'], downcast='unsigned')
df['smoke'] = pd.to_numeric(df['smoke'], downcast='unsigned')
df['alco'] = pd.to_numeric(df['alco'], downcast='unsigned')
df['active'] = pd.to_numeric(df['active'], downcast='unsigned')
df['cardio'] = pd.to_numeric(df['cardio'], downcast='unsigned')

dtypes = df.dtypes
colnames = dtypes.index
types = [i.name for i in dtypes.values]
column_types = dict(zip(colnames, types))

# Add 'overweight' column
overweight = (df.weight/df.height/df.height)*10000
df['overweight'] = round(overweight,1)
df['overweight'] = df['overweight'].apply(lambda x: 1 if x > 25 else 0)


# Normalize data by making 0 always good and 1 always bad. If the value of 'cholesterol' or 'gluc' is 1, make the value 0. If the value is more than 1, make the value 1.
def normalized(df):
    
    if (df['gluc'] > 1) | (df['cholesterol'] > 1):
        return 1
    elif (df['gluc'] == 1) | (df['cholesterol'] == 1):
        return 0
    
df['gluc'] = df.apply(normalized, axis = 1)
df['cholesterol'] = df.apply(normalized, axis = 1)   

# Draw Categorical Plot
def draw_cat_plot():
    # Create DataFrame for cat plot using `pd.melt` using just the values from 'cholesterol', 'gluc', 'smoke', 'alco', 'active', and 'overweight'.
    
    df_cat = pd.melt(df, id_vars=["cardio"], value_vars = ['active', 'alco', 'cholesterol', 'gluc', 'overweight', 'smoke'])

  

  

    # Group and reformat the data to split it by 'cardio'. Show the counts of each feature. You will have to rename one of the columns for the catplot to work correctly.
    df_cat = df_cat.value_counts().reset_index(name='total').sort_values(by='variable')
    # Draw the catplot with 'sns.catplot()'

    fig = sns.catplot(data=df_cat, y='total', kind='bar',
      x='variable',   hue ='value', col= "cardio", height = 5, aspect = 1.5)


 # Do not modify the next two lines
    fig.savefig('catplot.png')
    return fig


# Draw Heat Map
def draw_heat_map():
    # Clean the data
    df_heat = df.loc[
    (df['ap_lo'] <= df['ap_hi']) & 
    (df['height'] >= df['height'].quantile(0.025)) & 
    (df['height'] <= df['height'].quantile(0.975)) & 
    (df['weight'] >= df['weight'].quantile(0.025)) & 
    (df['weight'] <= df['weight'].quantile(0.975))
     ]  

    # Calculate the correlation matrix
    corr = df_heat.corr()

    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr)



    # Set up the matplotlib figure
    triangle_indices = np.triu_indices_from(mask)

    mask[triangle_indices] = True

    plt.figure(figsize=(16,10))
    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)




    # Draw the heatmap with 'sns.heatmap()'
    fig = sns.heatmap(corr, fmt='.1f',mask=mask, annot=True)

    plt.show()


    # Do not modify the next two lines
    fig.savefig('heatmap.png')
    return fig
