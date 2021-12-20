# Author :Islam Amar
# Date :"1/12/2021"

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import numpy as np
import plotly.express as px

confirmed_df = pd.read_csv('dataset/time_series_covid19_confirmed_global.csv');
death_df = pd.read_csv('dataset/time_series_covid19_deaths_global.csv');
country_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/web-data/data/cases_country.csv')
# data cleaning
country_df.columns = map(str.lower, country_df.columns)
# changing province/state to state and country/region to country
country_df = country_df.rename(columns={'country_region': 'country'})
new_dff = pd.read_csv('dataset/owid-covid-data.csv');
total_deaths_per_million = new_dff['total_deaths_per_million']
new_vaccinations_smoothed_per_million = new_dff['new_vaccinations_smoothed_per_million']
total_vaccinations_per_hundred = new_dff['total_vaccinations_per_hundred']
people_vaccinated_per_hundred = new_dff['people_vaccinated_per_hundred']
labels = list(confirmed_df['Country/Region']);
new_df = confirmed_df[['Country/Region', '3/15/20']];
new_death_df = death_df[['Country/Region', '3/15/20']];
spain_df = new_df[new_df['Country/Region'] == 'Spain'];
spain_val = spain_df['3/15/20'].values[0]
spain_death_df = new_death_df[new_death_df['Country/Region'] == 'Spain'];
spain_death_val = spain_death_df['3/15/20'].values[0]
germany_df = new_df[new_df['Country/Region'] == 'Germany']
germany_val = germany_df['3/15/20'].values[0]
germany_death_df = new_death_df[new_death_df['Country/Region'] == 'Germany'];
germany_death_val = germany_death_df['3/15/20'].values[0]
Austria_df = new_df[new_df['Country/Region'] == 'Austria']
Austria_val = Austria_df['3/15/20'].values[0]
Austria_death_df = new_death_df[new_death_df['Country/Region'] == 'Austria'];
Austria_death_val = Austria_death_df['3/15/20'].values[0]
US_df = new_df[new_df['Country/Region'] == 'US']
US_val = US_df['3/15/20'].values[0]
US_death_df = new_death_df[new_death_df['Country/Region'] == 'US'];
US_death_val = US_death_df['3/15/20'].values[0]
Brazil_df = new_df[new_df['Country/Region'] == 'Brazil']
Brazil_val = Brazil_df['3/15/20'].values[0]
Brazil_death_df = new_death_df[new_death_df['Country/Region'] == 'Brazil'];
Brazil_death_val = Brazil_death_df['3/15/20'].values[0]
labels = ['spain', 'germany', 'Austria', 'US', 'Brazil']
Confirmed_List = [spain_val, germany_val, Austria_val, US_val, Brazil_val]
Death_List = [spain_death_val, germany_death_val, Austria_death_val, US_death_val, Brazil_death_val]
ger_death = death_df[death_df['Country/Region'] == 'Germany']
ger_death = ger_death.iloc[:, 4:].values[0].sum();
spain_death = death_df[death_df['Country/Region'] == 'Spain']
spain_death = spain_death.iloc[:, 4:].values[0].sum();
US_death = death_df[death_df['Country/Region'] == 'US']
US_death = US_death.iloc[:, 4:].values[0].sum();
Austria_death = death_df[death_df['Country/Region'] == 'Austria']
Austria_death = Austria_death.iloc[:, 4:].values[0].sum();
Brazil_death = death_df[death_df['Country/Region'] == 'Brazil']
Brazil_death = Brazil_death.iloc[:, 4:].values[0].sum();
deaths_list = [ger_death, spain_death, US_death, Austria_death, Brazil_death]
Countrylabels = ['germany', 'spain', 'US', 'Austria', 'Brazil']
ger_death_line = confirmed_df[confirmed_df['Country/Region'] == 'Germany']
ger_death_line = ger_death_line.iloc[:, 4:].values[0]
spain_death_line = confirmed_df[confirmed_df['Country/Region'] == 'Spain']
spain_death_line = spain_death_line.iloc[:, 4:].values[0]
Date_list = confirmed_df.iloc[:, 4:]
Date = Date_list.columns;
ger_death_line = ger_death_line[-10:]
spain_death_line = spain_death_line[-10:]
Date = Date[-10:]
sorted_country_df = country_df.sort_values('confirmed', ascending=False)
x = np.arange(len(labels))
width = 0.25

def bar_chart():
    fig, ax = plt.subplots()
    ax.bar(x, Confirmed_List, width, label='Country', color="blue")
    ax.set_ylabel(' COVID-19 Confirmed cases')
    ax.set_title('Number of COVID-19 Confirmed cases in Each country at 15/3/2020 ')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation='vertical')
    ax.legend()
    fig.tight_layout()
    plt.show()


bar_chart()


def stackBar_chart():
    fig2, ax2 = plt.subplots()
    ax2.bar(x, Confirmed_List, width, label='Confirmed_List', color="red")
    ax2.bar(x, Death_List, width, bottom=Confirmed_List, label='Death_List', color="blue")
    ax2.set_ylabel('COVID-19 Confirmed /Death cases')
    ax2.set_title('Number of COVID-19 Confirmed /Death cases in Each country at 15/3/2020')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation='vertical')
    ax2.legend()
    fig2.tight_layout()
    plt.show()


# stackBar_chart()


def pie_chart():
    fig3, ax3 = plt.subplots()
    y = np.array(deaths_list);
    plt.pie(y, labels=Countrylabels, autopct='%1.0f%%', pctdistance=1.1, labeldistance=1.3)
    ax3.set_title('Covid-19 death cases percentage in Each Country 2020-2021')
    plt.show()


# pie_chart()


def line_chart():
    fig4, ax4 = plt.subplots()  # one axis
    ax4.set(xlabel='Date', ylabel=' Death cases', title='Death cases over a [12/4/2021 - 12/3/2021] period ')
    ax4.plot(Date, ger_death_line , label ="Germany")
    ax4.plot(Date, spain_death_line , label ="Spain")
    plt.legend()
    plt.show()


# line_chart()




def Scatter_chart():
    fig3, ax3 = plt.subplots()
    ax3.set(xlabel='people_vaccinated_per_hundred', ylabel='total_vaccinations_per_hundred',
            title='people_vaccinated_per_hundred/total_vaccinations_per_hundred')
    ax3.scatter(people_vaccinated_per_hundred, total_vaccinations_per_hundred)
    fig4, ax4 = plt.subplots()
    ax4.set(xlabel='new_vaccinations_smoothed_per_million', ylabel='total_deaths_per_million',
            title='new_vaccinations_smoothed_per_million/total_deaths_per_million')
    ax4.scatter(new_vaccinations_smoothed_per_million, total_deaths_per_million)
    plt.show()


# Scatter_chart()




def bubble_chart(n):
    fig = px.scatter(sorted_country_df.head(n), x="country", y="confirmed", size="confirmed", color="country",
                     hover_name="country", size_max=60)
    fig.update_layout(
        title=str(n) + " Worst hit countries",
        xaxis_title="Countries",
        yaxis_title="Confirmed Cases",
        width=700
    )
    fig.show();

bubble_chart(10);


fig, axs = plt.subplots(2, 2)

axs[0, 0].set_title('Number of COVID-19 Confirmed /Death cases in Each country at 15/3/2020')
axs[0, 0].bar(x, Confirmed_List, width, label='Confirmed_List', color="red")
axs[0, 0].bar(x, Death_List, width, bottom=Confirmed_List, label='Death_List', color="blue")
axs[0, 0].set_xticks(x)
axs[0, 0].set_xticklabels(labels, rotation='vertical')
axs[0, 0].legend()
axs[1, 0].set_title('Covid-19 death cases percentage in Each Country 2020-2021')
y = np.array(deaths_list);
axs[1, 0].pie(y, labels=Countrylabels, autopct='%1.0f%%', pctdistance=1.1, labeldistance=1.3)
axs[0, 1].set(xlabel='Date', ylabel=' Death cases', title='Death cases over a [12/4/2021 - 12/3/2021] period ')
axs[0, 1].plot(Date, ger_death_line , label ="Germany")
axs[0, 1].plot(Date, spain_death_line , label ="Spain")
axs[0, 1].legend()
axs[1, 1].set(xlabel='people_vaccinated_per_hundred', ylabel='total_vaccinations_per_hundred',
            title='people_vaccinated_per_hundred/total_vaccinations_per_hundred')
axs[1, 1].scatter(people_vaccinated_per_hundred, total_vaccinations_per_hundred)
fig.tight_layout()
plt.show()

