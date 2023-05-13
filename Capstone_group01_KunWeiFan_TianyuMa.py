# -*- coding: utf-8 -*-
"""
Created on Mon May  8 09:09:14 2023

@author: JD578

Dataset from: https://data.cdc.gov/NCHS/Provisional-COVID-19-Deaths-by-Sex-and-Age/9bhg-hcku
"""

#%% Import Library
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.dates import YearLocator, DateFormatter, MonthLocator
import seaborn as sns
# Model 1
from statsmodels.tsa.arima.model import ARIMA
from pmdarima.arima import ADFTest, auto_arima
# Model 2
#import statsmodels.formula.api as smf
import statsmodels.api as sm    

#%%

# definding the target age group
Age_Group = ['15-24 years','25-34 years','35-44 years','45-54 years','55-64 years'
             ,'65-74 years','75-84 years','85 years and over']
        
# Defines states in the east, central, and west of the United States
eastern_states = ['Connecticut', 'Maine', 'Massachusetts', 'New Hampshire', 'Rhode Island', 'Vermont',
                  'New Jersey', 'New York', 'New York City', 'Pennsylvania']
        
central_states = ['Illinois', 'Indiana', 'Michigan', 'Ohio', 'Wisconsin',
                  'Iowa', 'Kansas', 'Minnesota', 'Missouri', 'Nebraska', 'North Dakota', 'South Dakota']
        
western_states = ['Arizona', 'Colorado', 'Idaho', 'Montana', 'Nevada', 'New Mexico', 'Utah', 'Wyoming',
                  'Alaska', 'California', 'Hawaii', 'Oregon', 'Washington']
        
southern_states = ['Delaware', 'Florida', 'Georgia', 'Maryland', 'North Carolina', 'South Carolina', 'Virginia',
                   'Washington', 'District of Columbia', 'West Virginia', 'Alabama', 'Kentucky', 'Mississippi', 'Tennessee',
                   'Arkansas', 'Louisiana', 'Oklahoma', 'Texas']

#%% Functions of Calculation

# for chart thousands display
def format_K_M_ann(x, decimal_places=0):
    if x >= 1000000:
        return f"{x / 1000000:.{decimal_places}f}M"
    elif x >= 1000:
        return f"{x / 1000:.0f}K"
    else:
        return f"{x:.0f}"

# For chart millions display
def format_K_M(decimal_places):
    def formatter(x, pos=0):
        if x >= 1000000:
            return f"{x / 1000000:.{decimal_places}f}M"
        elif x >= 1000:
            return f"{x / 1000:.0f}K"
        else:
            return f"{x:.0f}"
    return formatter

# Cumulative sum
def cumulative_stores(df, sort_by, column_name, group_by,):
    try:
        df = df.sort_values(sort_by)
        df.reset_index(drop=True, inplace=True)
        
        df = df.groupby(group_by)[column_name].sum().reset_index(name="Group_sum")
    
        # Create a new column called 'cumsum' that contains the cumulative sum of the number of stores opened in that state
        df['cumsum'] = df['Group_sum'].cumsum()
        
        # Create a new dataframe with only the 'OPENDATE' and 'cumsum' columns and return it
        result_df = df[[group_by, 'cumsum']]
        return result_df
    
    # exception
    except FileNotFoundError as e:
        raise FileNotFoundError(f"{e}")
    except pd.errors.EmptyDataError as e:
        raise pd.errors.EmptyDataError(f"Error: {e}")
    except pd.errors.ParserError as e:
        raise pd.errors.ParserError(f"Error: {e}")
        return None

#%% Strat the main function
def main_function_Group1_Clean(df):
    try:            
        #%% Dataset structuring and cleaning
        
        # Using 'Group' by Month
        df_clean = df[df['Group']=='By Month']
        df_clean = df_clean.reset_index(drop=True)
        
        # Drop the columns that have only one unique value
        df_clean = df_clean.drop(['Data As Of','Footnote'], axis=1)
        
        # Correct the format of date
        df_clean['Start Date'] = pd.to_datetime(df_clean['Start Date'], format='%m/%d/%Y')
        df_clean['End Date'] = pd.to_datetime(df_clean['End Date'], format='%m/%d/%Y')
        
        # fill the na value of those death count column by 0
        df_clean['COVID-19 Deaths'].fillna(0, inplace=True)
        df_clean['Total Deaths'].fillna(0, inplace=True)
        df_clean['Pneumonia Deaths'].fillna(0, inplace=True)
        df_clean['Pneumonia and COVID-19 Deaths'].fillna(0, inplace=True)
        df_clean['Influenza Deaths'].fillna(0, inplace=True)
        df_clean['Pneumonia, Influenza, or COVID-19 Deaths'].fillna(0, inplace=True)
        
        # Create a new column "Region" 
        df_clean['Region'] = df_clean['State'].apply(lambda x: 'Eastern' if x in eastern_states
                                                     else ('Central' if x in central_states 
                                                           else ('Western' if x in western_states 
                                                                 else ('Southern' if x in southern_states 
                                                                       else None))))
        
        # filter the total value
        df_clean = df_clean.loc[df_clean['State'] != 'United States']
        df_clean = df_clean.loc[df_clean['Sex'] != 'All Sexes']
        df_clean = df_clean.loc[df_clean["Age Group"].isin(Age_Group)]
        
        # remove "years" from the age column and make the format unified
        df_clean['Age Group'] = df_clean['Age Group'].str.replace(' years', '')
        df_clean['Age Group'] = df_clean['Age Group'].str.replace('85 and over', '>85')
        
        df_clean['IPC_Sum'] = df_clean['Influenza Deaths']+df_clean['Pneumonia Deaths']+df_clean['COVID-19 Deaths']
        
        # combine two columns with separator '/'
        #df_clean['Year_Month'] = df_clean.apply(lambda x: str(str(f"{x['Year']:.0f}") + '/' + str(f"{x['Month']:.0f}")), axis=1)
        #df_clean['Year_Month'] = pd.to_datetime(df_clean['Start Date'], format='%Y/%m')
        
        #%% success end the main function
        return df_clean  
    
    #%% exception
    except FileNotFoundError as e:
        raise FileNotFoundError(f"{e}")
    except pd.errors.EmptyDataError as e:
        raise pd.errors.EmptyDataError(f"Error: {e}")
    except pd.errors.ParserError as e:
        raise pd.errors.ParserError(f"Error: {e}")
        return None    
          
#%% Strat the main function
def main_function_Group1_Part1(df_clean):
    try:   
        #%% Plot 1 - Deaths by Category (% breakdown)
        
        df_Death_Cat = pd.DataFrame({
            'labels': ['Influenza Deaths','Pneumonia Deaths','COVID-19 Deaths','Others'],
            'Values': [sum(df_clean['Influenza Deaths']), sum(df_clean['Pneumonia Deaths']), sum(df_clean['COVID-19 Deaths']), (sum(df_clean['Total Deaths'])-sum(df_clean['IPC_Sum']))],
        })
        
        colors = ['#12239E', '#118DFF', '#ff7f7f', '#5DADE2']
        
        # Create pie chart
        plt.pie(df_Death_Cat['Values'], labels=df_Death_Cat['labels'], autopct='%.1f%%', colors=colors)
        
        # Add title
        plt.title('Death Count by Category (% breakdown)')
        
        # Display chart
        plt.show()
        
        #%% Plot 2 - Cumulative Deaths over Time

        # call the function cumulative_stores
        
        df_DeathCumSum_byDate = cumulative_stores(df_clean, 'Start Date', 'IPC_Sum','Start Date')
        
        # Create a new figure and axes object
        fig, ax = plt.subplots()
        
        # Plot the data using the plot function
        ax.plot(df_DeathCumSum_byDate["Start Date"], df_DeathCumSum_byDate["cumsum"], color = '#12239E')
        
        # Add axis labels and a title to the plot
        ax.set(title="Cumulative Deaths over Time",
               xlabel="Year", ylabel="Cumulative Deaths",
               ylim = (0,None))
        
        # Set the x-axis ticks to show every year
        years = YearLocator(1)   # every year
        years_fmt = DateFormatter('%Y')
        ax.xaxis.set_major_locator(years)
        ax.xaxis.set_major_formatter(years_fmt)
        
        # Fill the area
        ax.fill_between(df_DeathCumSum_byDate["Start Date"], df_DeathCumSum_byDate["cumsum"], color='lightblue', alpha=0.5)
        
        # Add grid lines to the plot
        ax.grid(True, color='gray', linestyle='--', linewidth=0.5)
        
        # Apply the function to the y-axis tick labels
        formatter = ticker.FuncFormatter(format_K_M(1))
        plt.gca().yaxis.set_major_formatter(formatter)
        
        
        # Show the plot
        plt.show()
        
        #%% Plot 3 - Cumulative Deaths by Category over Time
        
        # call the function cumulative_stores
        df_PneCumSum_byDate = cumulative_stores(df_clean, 'Start Date', 'Pneumonia Deaths','Start Date')
        df_CovCumSum_byDate = cumulative_stores(df_clean, 'Start Date', 'COVID-19 Deaths','Start Date')
        df_InfCumSum_byDate = cumulative_stores(df_clean, 'Start Date', 'Influenza Deaths','Start Date')
        # Create a new figure and axes object
        fig, ax = plt.subplots()
        
        # Plot the data using the plot function
        ax.plot(df_PneCumSum_byDate["Start Date"], df_PneCumSum_byDate["cumsum"], label='Pneumonia', color='#118DFF')
        ax.plot(df_CovCumSum_byDate["Start Date"], df_CovCumSum_byDate["cumsum"], label='COVID-19', color='#12239E')
        ax.plot(df_InfCumSum_byDate["Start Date"], df_InfCumSum_byDate["cumsum"], label='Influenza', color='#ff7f7f')
        
        # Add axis labels and a title to the plot
        ax.set(title="Cumulative Deaths by Category over Time",
               xlabel="Year", ylabel="Cumulative Deaths",
               ylim = (0,None))
        
        # Set the x-axis ticks to show every year
        years = YearLocator(1)   # every year
        years_fmt = DateFormatter('%Y')
        ax.xaxis.set_major_locator(years)
        ax.xaxis.set_major_formatter(years_fmt)
        
        # Add grid lines to the plot
        ax.grid(True, color='gray', linestyle='--', linewidth=0.5)
        
        # Apply the function to the y-axis tick labels
        formatter = ticker.FuncFormatter(format_K_M(0))
        plt.gca().yaxis.set_major_formatter(formatter)
        
        plt.legend()
        
        # Show the plot
        plt.show()
        
        #%% Plot 4 - COVID-19 Deaths by Year
        # Group the data by year and sum the COVID-19 deaths
        df_year = df_clean.groupby(df_clean['Start Date'].dt.year)['COVID-19 Deaths'].sum()
        
        # Create a line chart
        plt.plot(df_year.index, df_year.values, color='#12239E')
        
        # Set the x-axis label and tick marks
        plt.title('COVID-19 Deaths by Year')
        plt.xlabel('Year')
        plt.xticks(df_year.index)
        
        # Set the y-axis label and format the values to be displayed in thousands
        plt.ylabel('COVID-19 Deaths')
        plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: format_K_M_ann(x, 1)))
        
        # Label each data point with its value
        for x, y in zip(df_year.index, df_year.values):
            plt.annotate(format_K_M_ann(y), (x, y), textcoords="offset points", xytext=(1,1), ha='center')
        
        # Show the plot
        plt.show()
        
        #%% Plot 5 - COVID-19 Deaths by Month
        
        df_COVID_byMonth = df_clean.groupby(['Start Date'])['COVID-19 Deaths'].sum().reset_index()
        
        # Create a new figure and axes object
        fig, ax = plt.subplots()
        
        # Plot the data using the plot function
        ax.plot(df_COVID_byMonth["Start Date"], df_COVID_byMonth["COVID-19 Deaths"], color='#12239E')
        
        # Add axis labels and a title to the plot
        ax.set(title="COVID-19 Deaths by Month",
               xlabel="", ylabel="Deaths",
               ylim = (0,None))
        
        # Set the x-axis ticks to show every year
        months = MonthLocator(interval=2)   # every month
        months_fmt = DateFormatter('%Y/%m')
        ax.xaxis.set_major_locator(months)
        ax.xaxis.set_major_formatter(months_fmt)
        
        # Apply the function to the y-axis tick labels
        formatter = ticker.FuncFormatter(format_K_M(0))
        plt.gca().yaxis.set_major_formatter(formatter)
        
        # rotate x-axis labels by 45 degrees
        plt.xticks(rotation=90)
        
        # Get the index of the highest value
        max_idx = df_COVID_byMonth["COVID-19 Deaths"].idxmax()
        
        # Get the x and y values of the highest point
        max_x = df_COVID_byMonth.iloc[max_idx]["Start Date"]
        max_x2 = df_COVID_byMonth.iloc[max_idx+1]["Start Date"]
        max_y = df_COVID_byMonth.iloc[max_idx]["COVID-19 Deaths"]
        
        # Add the label using the annotate function
        ax.annotate(f"{format_K_M_ann(max_y,0)}", xy=(max_x, max_y), xytext=(max_x2, max_y-20000), ha='left', fontsize=12, arrowprops=dict(facecolor='black', arrowstyle='->'))
        
        # Show the plot
        plt.show()
        
        #%% Plot 6 - Predictive Model #########################
        
        df_Model2 = df_CovCumSum_byDate.copy()
        
        # Prepare the data
        dates = pd.to_datetime(df_Model2['Start Date'])  # Assuming your date column is named 'Date'
        values = df_Model2['cumsum']  # Assuming your value column is named 'Value'
        
        # Split the data into training and testing sets
        train_size = int(len(values) * 0.8)  # 80% for training, 20% for testing
        train_data = values[:train_size]
        test_data = values[train_size:]
        
        # Checking time series stationary (True means d should be 1)
        ADFTest(alpha = 0.05).should_diff(values)
        
        p, d, q = auto_arima(train_data, seasonal=False, suppress_warnings=True).order
        
        # Build the ARIMA model
        model = ARIMA(train_data, order=(1, d, q))  # Set the order of the model (p, d, q)
        model_fit = model.fit()
        
        # Forecast future values
        forecast = model_fit.forecast(steps=len(test_data))
        
        # Create a chart
        
        # Create a new figure and axes object
        fig, ax = plt.subplots()
        
        ax.plot(dates, values, label='Actual Data', color = '#118DFF')
        ax.plot(dates[train_size:], forecast, label='Forecasted Data', color='#12239E')
        
        ax.set(title="Cumulative COVID-19 Deaths over Time - Arima Model",
               xlabel="Year", ylabel="Deaths",
               ylim = (0,None))
        
        # Set the x-axis ticks to show every year
        years = YearLocator(1)   # every year
        years_fmt = DateFormatter('%Y')
        ax.xaxis.set_major_locator(years)
        ax.xaxis.set_major_formatter(years_fmt)
        
        # Add grid lines to the plot
        ax.grid(True, color='gray', linestyle='--', linewidth=0.5)
        
        # Apply the function to the y-axis tick labels
        formatter = ticker.FuncFormatter(format_K_M(0))
        plt.gca().yaxis.set_major_formatter(formatter)
        
        plt.legend()
        plt.show()
        
        #%% success end the main function
        return 'Success End Main Function Part1'    
    
    #%% exception
    except FileNotFoundError as e:
        raise FileNotFoundError(f"{e}")
    except pd.errors.EmptyDataError as e:
        raise pd.errors.EmptyDataError(f"Error: {e}")
    except pd.errors.ParserError as e:
        raise pd.errors.ParserError(f"Error: {e}")
        return None
        
#%%
def main_function_Group1_Part2(df_clean):
    try:
        #%% Plot 7 - COVID-19 Deaths by Region (pie chart)
        # Group the data by region and sum the COVID-19 deaths
        df_region = df_clean.groupby('Region')['COVID-19 Deaths'].sum().reset_index()
        
        # Set up the figure and axes
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Create a pie chart of the COVID-19 deaths by region
        explode = (0, 0, 0.1, 0)
        colors = ['#a4a7ed', '#ADD8E6', '#FFD21B', '#eda4d5']
        #sns.set_palette(sns.color_palette(colors2))
        ax.pie(df_region['COVID-19 Deaths'], labels=df_region['Region'], colors=colors, 
               autopct='%1.1f%%', startangle=90, explode=explode, pctdistance=0.85, labeldistance=1.1)
        
        # Add a title
        ax.set_title("COVID-19 Deaths by Region", fontsize=16, fontweight='bold')
        
        # Add a legend
        ax.legend(loc='lower left')
        
        # Show the plot
        plt.show()
        
        #%% Plot 8 - COVID-19 Deaths by Sex and Region (bar plot)
        # Group by state and gender
        df_grouped = df_clean.groupby(['Sex', 'Region'])['COVID-19 Deaths'].sum().reset_index()
        
        # Define the color palette for the hue categories
        palette = {'Male': '#a4a7ed', 'Female': '#ff7f7f'}  # Specify colors for each category
        
        # Draw the bar graph
        fig, ax = plt.subplots()
        sns.barplot(data=df_grouped, x='Region', y='COVID-19 Deaths', palette=palette, hue='Sex', ax=ax, edgecolor='black')
        
        # Add title and labels
        plt.title('COVID-19 Deaths by Sex and Region', fontsize=16)
        plt.xlabel('Region', fontsize=14)
        plt.ylabel('COVID-19 Deaths', fontsize=14)
        
        # Add value labels
        for p in ax.patches:
            x = p.get_x() + p.get_width() / 2
            y = p.get_height()
            value = format_K_M_ann(y)
            ax.annotate(value, (x, y), ha='center', va='bottom')
        # another way
        #for p in ax.containers:
        #    ax.bar_label(p, label_type='edge', fontsize=12, padding=4)
            
        # Apply the function to the y-axis tick labels
        formatter = ticker.FuncFormatter(format_K_M(1))
        plt.gca().yaxis.set_major_formatter(formatter)
             
        # Display 
        plt.show()
        
        #%% Plot 9 - Influenza, Pneumonia and COVID-19 Deaths by Age Group
        
        # calculate each age group deaths sum
        df_Age_Group_Sum = df_clean.groupby("Age Group")[['IPC_Sum']].sum().reset_index()
        
        # Create a new figure and axes object
        fig, ax = plt.subplots()
        
        # Plot the data using the plot function
        # Add color changed and frame
        ax.bar(df_Age_Group_Sum["Age Group"], df_Age_Group_Sum["IPC_Sum"], color='#2471A3', edgecolor='black')
        
        # Add axis labels and a title to the plot
        ax.set(title="Influenza, Pneumonia and COVID-19 Deaths by Age Group",
               xlabel="", ylabel="Deaths")
        
        # Add labels to the bars
        for i in range(len(df_Age_Group_Sum)):
            plt.annotate(str(format_K_M_ann(df_Age_Group_Sum["IPC_Sum"][i],1)), xy=(df_Age_Group_Sum["Age Group"][i], df_Age_Group_Sum["IPC_Sum"][i]), ha='center', va='bottom')
        
        # Apply the function to the y-axis tick labels
        formatter = ticker.FuncFormatter(format_K_M(1))
        plt.gca().yaxis.set_major_formatter(formatter)
        
        # remove the spines (square around the chart)
        plt.gca().spines["top"].set_visible(False)
        plt.gca().spines["right"].set_visible(False)
        
        # rotate x-axis labels by 45 degrees
        #plt.xticks(rotation=45)
        
        # Adjust  the size
        plt.xticks(fontsize=11)
        plt.yticks(fontsize=11)
        
        # Show the chart
        plt.show()
        
        #%% Plot 10 - COVID-19 Deaths by Age Group
        
        # calculate each age group deaths sum
        df_Age_Group_Sum = df_clean.groupby("Age Group")['COVID-19 Deaths'].sum().reset_index()
        
        # Create a new figure and axes object
        fig, ax = plt.subplots()
        
        # Plot the data using the plot function
        # Add color changed and frame
        ax.bar(df_Age_Group_Sum["Age Group"], df_Age_Group_Sum["COVID-19 Deaths"], color='#E67E22', edgecolor='black')
        
        # Add axis labels and a title to the plot
        ax.set(title="COVID-19 Deaths by Age Group",
               xlabel="", ylabel="Deaths")
        
        # Add labels to the bars
        for i in range(len(df_Age_Group_Sum)):
            plt.annotate(str(format_K_M_ann(df_Age_Group_Sum["COVID-19 Deaths"][i],1)), xy=(df_Age_Group_Sum["Age Group"][i], df_Age_Group_Sum["COVID-19 Deaths"][i]), ha='center', va='bottom')
        
        # Apply the function to the y-axis tick labels
        formatter = ticker.FuncFormatter(format_K_M(1))
        plt.gca().yaxis.set_major_formatter(formatter)
        
        # remove the spines (square around the chart)
        plt.gca().spines["top"].set_visible(False)
        plt.gca().spines["right"].set_visible(False)
        
        # rotate x-axis labels by 45 degrees
        #plt.xticks(rotation=45)
        
        # Adjust  the size
        plt.xticks(fontsize=11)
        plt.yticks(fontsize=11)
        
        # Show the chart
        plt.show()
        
        #%% Plot 11 - COVID-19 Deaths by Age Group in the Eastern
        # filter dataset
        Eastern_df = df_clean[df_clean['Region'] == 'Eastern']
        
        # calculate each age group deaths sum
        df_Age_Group_Sum = Eastern_df.groupby("Age Group")['COVID-19 Deaths'].sum().reset_index()
        
        # Create a new figure and axes object
        fig, ax = plt.subplots()
        
        # Plot the data using the plot function
        # Add color changed and frame
        ax.bar(df_Age_Group_Sum["Age Group"], df_Age_Group_Sum["COVID-19 Deaths"],  color='#ADD8E6', edgecolor='black')
        
        # Add axis labels and a title to the plot
        ax.set(title="COVID-19 Deaths by Age Group (Eastern)",
               xlabel="Age Group", ylabel="Deaths")
        
        # Add labels to the bars
        for i in range(len(df_Age_Group_Sum)):
            plt.annotate(str(format_K_M_ann(df_Age_Group_Sum["COVID-19 Deaths"][i],1)), 
                         xy=(df_Age_Group_Sum["Age Group"][i], df_Age_Group_Sum["COVID-19 Deaths"][i]), ha='center', va='bottom')
        
        # Apply the function to the y-axis tick labels
        formatter = ticker.FuncFormatter(format_K_M(1))
        plt.gca().yaxis.set_major_formatter(formatter)
        
        # remove the spines (square around the chart)
        plt.gca().spines["top"].set_visible(False)
        plt.gca().spines["right"].set_visible(False)
        
        # rotate x-axis labels by 45 degrees
        #plt.xticks(rotation=45)
        
        # Adjust  the size
        plt.xticks(fontsize=11)
        plt.yticks(fontsize=11)
        
        # Show the chart
        plt.show()
        
        #%% Plot 12 - COVID-19 Deaths by Age Group in the Central
        # filter dataset
        df_central = df_clean[df_clean['Region'] == 'Central']
        df_agegroup = df_central.groupby('Age Group')['COVID-19 Deaths'].sum().reset_index()
        
        plt.figure()
        
        # Plot the data 
        ax = sns.barplot(data=df_agegroup, x='Age Group', y='COVID-19 Deaths', color='#a4a7ed', edgecolor='black')
        ax.set_title('COVID-19 Deaths by Age Group (Central)')
        ax.set_xlabel('Age Group')
        ax.set_ylabel('COVID-19 Deaths')
        
        # Add value labels
        for p in ax.patches:
            x = p.get_x() + p.get_width() / 2
            y = p.get_height()
            value = format_K_M_ann(y)
            ax.annotate(value, (x, y), ha='center', va='bottom')
        
        # Apply the function to the y-axis tick labels
        formatter = ticker.FuncFormatter(format_K_M(1))
        plt.gca().yaxis.set_major_formatter(formatter)
        
        # remove the spines (square around the chart)
        plt.gca().spines["top"].set_visible(False)
        plt.gca().spines["right"].set_visible(False)
        
        # Adjust  the size
        plt.xticks(fontsize=11)
        plt.yticks(fontsize=11)
        
        # show the plot
        plt.show()
        
        #%% Plot 13 - COVID-19 Deaths by Age Group in the Western
        # filter dataset
        df_grouped = df_clean[df_clean['Region'] == 'Western'].groupby(['Age Group'])['COVID-19 Deaths'].sum().reset_index()
        
        # Plot the data 
        plt.figure()
        ax = sns.barplot(data=df_grouped, x='Age Group', y='COVID-19 Deaths', color='#eda4d5', edgecolor='black')
        
        # Add value labels
        for i, row in df_grouped.iterrows():
            ax.text(i, row['COVID-19 Deaths'], format_K_M_ann(int(row['COVID-19 Deaths'])), ha='center', va='bottom', fontsize=12)

        # Add a chart title and axis labels
        plt.title('COVID-19 Deaths by Age Group (Western)')
        plt.xlabel('Age Group')
        plt.ylabel('COVID-19 Deaths')
        
        # Apply the function to the y-axis tick labels
        formatter = ticker.FuncFormatter(format_K_M(1))
        plt.gca().yaxis.set_major_formatter(formatter)
        
        # remove the spines (square around the chart)
        plt.gca().spines["top"].set_visible(False)
        plt.gca().spines["right"].set_visible(False)
        
        # Adjust  the size
        plt.xticks(fontsize=11)
        plt.yticks(fontsize=11)
        
        # Show the plot
        plt.show()
        
        #%% Plot 14 - COVID-19 Deaths by Age Group in the Southern
        # filter dataset
        df_grouped = df_clean[df_clean['Region'] == 'Southern'].groupby(['Age Group'])['COVID-19 Deaths'].sum().reset_index()
        
        # Plot the data 
        plt.figure()
        ax = sns.barplot(data=df_grouped, x='Age Group', y='COVID-19 Deaths', color='#FFD21B', edgecolor='black')
        
        # Add value labels
        for i, row in df_grouped.iterrows():
            ax.text(i, row['COVID-19 Deaths'], format_K_M_ann(int(row['COVID-19 Deaths'])), ha='center', va='bottom', fontsize=12)
        
        # Add a chart title and axis labels
        plt.title('COVID-19 Deaths by Age Group (Southern)')
        plt.xlabel('Age Group')
        plt.ylabel('COVID-19 Deaths')
        
        # Apply the function to the y-axis tick labels
        formatter = ticker.FuncFormatter(format_K_M(1))
        plt.gca().yaxis.set_major_formatter(formatter)
        
        # remove the spines (square around the chart)
        plt.gca().spines["top"].set_visible(False)
        plt.gca().spines["right"].set_visible(False)
        
        # Adjust  the size
        plt.xticks(fontsize=11)
        plt.yticks(fontsize=11)
        
        # Show the plot
        plt.show()
        
        #%% Chart 15 - OLS Regression  ############################
        '''Ordinary Least Squares (OLS) regression is a statistical method used to analyze the relationship between a dependent variable and one or more independent variables. OLS regression involves finding the best-fitting line through the data by minimizing the sum of the squared distances between the observed values and the predicted values. This method assumes that the relationship between the dependent variable and independent variables is linear and additive.
        
        The OLS regression model estimates the coefficients of the independent variables, which can be used to predict the values of the dependent variable. It also provides information on the statistical significance and goodness of fit of the model, which can be used to assess the reliability of the predictions.
        
        OLS regression is a widely used technique in many fields, including economics, finance, social sciences, and engineering. It is a simple and powerful method that can be used to analyze and model a wide range of data sets.
        '''
        
        # import filter data
        df_regression = df_clean[['COVID-19 Deaths', 'Age Group']]
        
        #create dummy variables
        # Perform dummy variable encoding
        df_regression = pd.get_dummies(df_regression, columns=['Age Group'])
        # drop the 'Age Group_Under 1 year' column
        df_regression = df_regression.drop('Age Group_15-24', axis=1)
        
        #----------------------------
        
        # Define the endogenous and exogenous variables
        y = df_regression['COVID-19 Deaths']
        x = df_regression.drop('COVID-19 Deaths', axis=1)
        
        # Add a constant to the exogenous variables
        x = sm.add_constant(x)
        
        # Fit the OLS model
        model = sm.OLS(y, x).fit()
        
        # Print the summary of the model
        print("OLS Regression Model of COVID-19 Deaths by Age Group:\n",model.summary(alpha=0.05))
        
        #y=0.1290+1.5288D_1+5.2814D_2+14.9068D_3+35.8427D_4+58.7592D_5+68.6160D_6+71.0842D_7
        
        #%% success end the main function
        return 'Success End Main Function Part2'    
    
    #%% exception
    except FileNotFoundError as e:
        raise FileNotFoundError(f"{e}")
    except pd.errors.EmptyDataError as e:
        raise pd.errors.EmptyDataError(f"Error: {e}")
    except pd.errors.ParserError as e:
        raise pd.errors.ParserError(f"Error: {e}")
        return None
    
#%% Testing
if __name__ == "__main__":
    # Input dataset
    df = pd.read_csv("Provisional_COVID-19_Deaths_by_Sex_and_Age.csv")
    df_clean = main_function_Group1_Clean(df)
    main_function_Group1_Part1(df_clean)
    main_function_Group1_Part2(df_clean)
    
