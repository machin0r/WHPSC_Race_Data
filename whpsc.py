'''
This program is for visualising the data from the World Human Powered Speed
Challenge from 2001-2019.
The data is collected from http://ihpva.org/whpsc/ and organised into the 
included excel file
'''
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
from scipy.interpolate import make_interp_spline
import seaborn as sns
import whpsc_random_forest


file_path = Path(__file__).with_name('race_results.xlsx')

colours = ["#011261", "#2c1c79", "#4d248b", "#6c2995", "#8a2f97",
                   "#a63591", "#c03e82", "#d7486d", "#ea5752", "#f86934",
                   "#ff8000"]

class RaceInformation:
    '''
    This class will open the excel file containing all the race data, where
    each sheet is a year. Each sheet is then added separately into the 
    dictionary object race_information_dict by setup_databases().
    The successful runs are compiled into one dataframe for ease of filtering
    by the successful_runs function
    '''
    def __init__(self, excel_path):
        #self.excel_path = Path(__file__).with_name('race_results.xlsx')
        self.excel_path = excel_path
        self.race_information_dict = {}
        self.all_successful_runs = pd.DataFrame()
        self.year_range = []

        self.setup_databases()
        self.successful_runs()

    def setup_databases(self):
        '''This spltis the sheets from the excel document into 
        dictionary items. Each sheet is a race year'''
        full_excel = pd.read_excel(self.excel_path,sheet_name=None)
        for sheet, dataframe in full_excel.items():
            self.race_information_dict[sheet] = dataframe

    def successful_runs(self):
        '''This creates a dataframe containing only successful runs
        It does this by filitering DNS (Did Not Start) and DNF (Did Not Finish)
        results from the main excel sheet (and Timing Errors)'''
        runs = {}
        #df = pd.DataFrame(columns=['Year', 'Country', 'Count'])
        for key, value in self.race_information_dict.items():
            runs[key] = value[value['Elapsed Time (s)'] != "DNF"]
            runs[key] = runs[key][runs[key]['Elapsed Time (s)'] != "DNS"]
            runs[key] = runs[key][runs[key]['Elapsed Time (s)'] != "Timing Error"]
            self.year_range.append(key)
        self.all_successful_runs = pd.concat(runs.values(), ignore_index=True)

class RaceAnalysis:
    '''
    This class contains all the analysis functions that can be performed on the
    race data, and takes a RaceInformation object as an argument
    The functions will usually return a figure or data 
    '''
    def __init__(self, race_data):
        self.race_date = race_data
        self.race_information_dict = race_data.race_information_dict
        self.all_successful_runs = race_data.all_successful_runs
        self.year_range = race_data.year_range

    def plot_formatter(self, plot, y_label ="", x_label = "", decimal=0):
        '''Provides a uniform look for all the figures that are proeduced
        Takes a plot object and the y axis label, and returns the 
        formatted plot'''
        formatted_plot = plot
        # Set the x axis label formatting
        formatted_plot.set_xlabel(x_label, labelpad=10, fontsize=14,
                                       fontweight='bold', color='white',
                                       verticalalignment='center')
        formatted_plot.xaxis.set_label_position("bottom")
        formatted_plot.tick_params(axis='x', colors='white')

        # Set the y axis parameters, including making numbers whole for
        # aesthetics
        formatted_plot.set_ylabel(y_label, fontsize=14,
                                       labelpad=10, fontweight='bold',
                                       color='white')
        formatted_plot.yaxis.set_label_position("left")
        formatted_plot.yaxis.set_major_formatter(lambda s, i : f'{s:,.{decimal}f}')
        formatted_plot.yaxis.set_major_locator(MaxNLocator(integer=False))
        formatted_plot.yaxis.set_tick_params(pad=2, labeltop=False,
                                                  labelbottom=True,
                                                  bottom=False,
                                                  labelsize=14, color='white')
        formatted_plot.tick_params(axis='y', colors='white')

        return formatted_plot

    def figure_creator(self, x_dim, y_dim):
        '''
        Creates figures with a uniform look
        '''
        figure = plt.figure(figsize=(x_dim,y_dim), linewidth=10,
                        edgecolor='#393d5c',
                        facecolor='#25253c')
        rect = [0.1,0.1,0.8,0.8]
        return figure, rect

    def rider_nationality(self):
        '''This fucntion produces a stackplot of the rider nationalities 
        for each year
        It returns the stackplot figure'''
        # Set up dictionaries to store the data from the dataframe
        # The dictionaries store each year as a dataframe
        riders = {}
        countries = {}
        countries_melted = {}  # Used for smoothing the line plot

        # Iterate through the race information and remove duplicate riders each
        # year, adding the unique riders nationality to the countries
        # dataframe for that year
        # The melted dataframe adds the year of each entry in a new column
        for key, value in self.race_information_dict.items():
            riders[key] = value.drop_duplicates(subset=['Rider'])
            countries[key] = (riders[key]['Country'].value_counts().
                              rename_axis('Country').reset_index(name=key))
            countries_melted[key] = (countries[key].
                                     melt(id_vars=['Country'],
                                          var_name='Year',
                                          value_name='Count'))

        # Concat all the countries years into one dataframe
        # and remove any NaN values with a 0 for that year
        countries_melted_df = pd.concat(countries_melted.values(),
                                        ignore_index=True)
        countries_df = (pd.concat(countries.values(), ignore_index=True).
                        groupby('Country').sum())
        countries_df.fillna(0, inplace=True)
        countries_melted_df.fillna(0, inplace=True)
        countries_melted_pivot = countries_melted_df.pivot(index='Year',
                                                           columns='Country',
                                                           values='Count')
        countries_melted_pivot.fillna(0, inplace=True)

        # Create a smooth line between the data years to remove sharp edges
        x_axis_smooth = np.linspace(int(self.year_range[0]),
                                    int(self.year_range[-1]), 500)

        # Make a spline between the data points, using x_axis_smmoth
        # k value controls the spline, make 1 for sharp edged again
        countries_melt_piv_smooth = (pd.DataFrame({country: make_interp_spline(
                                        countries_melted_pivot.index,
                                        countries_melted_pivot[country],
                                        k=1)(x_axis_smooth)
                                        for country in
                                        countries_melted_pivot.columns}))

        # Transpose the countries dataframe and convert to a list to get
        # the labels for the plot
        countries_list = countries_df.T.columns.values
        nationality_colours = ['#002f61', '#2f3a49', "#011261", '#AA5161',
                               '#425c7c', '#AA689a', '#695fcb', '#b133da',
                               '#e70091', '#ff8000']

        # Create the figure for the plot
        fig_nationality, rect = self.figure_creator(20,10)

        # Create the plot
        ax_nationality = fig_nationality.add_axes(rect, frameon=False)
        ax_nationality.stackplot(x_axis_smooth,
                                 countries_melt_piv_smooth.values.T,
                                 colors=nationality_colours, labels = countries_list,
                                 edgecolor="#000000", linewidth=1.5)

        # Format the plot, and invert the legend so it's in the same order
        # as the data
        ax_nationality = self.plot_formatter(ax_nationality, y_label="Entrants")
        ax_nationality.legend()
        handles, labels = ax_nationality.get_legend_handles_labels()
        ax_nationality.legend(handles[::-1], labels[::-1], loc='lower right')
        ax_nationality.xaxis.set_major_formatter(lambda s, i : f'{s:.0f}')

        return fig_nationality

    def unique_bikes(self):
        '''
        This functions will determine the top 10 most ridden bikes across
        all years of the competition
        It returns a radial bar chart figure
        '''

        # Count up the number of occurances of each bikes name in the
        # "Vehicle name" column, and then return the top 10
        unique_bikes_top10 = (self.all_successful_runs['Vehicle Name'].
                              value_counts().rename_axis('Vehicle Name').
                              reset_index(name='Count').head(10))
        # Invert the order so the plot moves outwards
        unique_bikes_top10 = unique_bikes_top10.reindex(
                              index=unique_bikes_top10.index[::-1])

        # Get the count for the most ridden bike
        max_value = max(unique_bikes_top10['Count'])
        # colours = ["#011261", "#2c1c79", "#4d248b", "#6c2995", "#8a2f97",
        #            "#a63591", "#c03e82", "#d7486d", "#ea5752", "#f86934",
        #            "#ff8000"]
        # Make labels for the plot of the Name + Count
        bike_labels = [f'   {x} ({v})' for x, v in
                       zip(list(unique_bikes_top10['Vehicle Name']),
                           list(unique_bikes_top10['Count']))]


        # Set up the figure
        fig_unique_bikes, rect = self.figure_creator(10,10)

        # Add axis for a polar plot
        # Start bars at top ('N') going CCW (1)
        ax_unique_bikes_bg = fig_unique_bikes.add_axes(rect,
                                                       polar=True,
                                                       frameon=False)
        ax_unique_bikes_bg.set_theta_zero_location('N')
        ax_unique_bikes_bg.set_theta_direction(1)

        # Create a grey background for the number of items to plot (10)
        for i in range(len(unique_bikes_top10)):
            ax_unique_bikes_bg.barh(i, max_value*1.5*np.pi/max_value,
                            color='grey',
                            alpha=0.1)
        ax_unique_bikes_bg.axis('off')

        # Add axis for radial chart
        ax_unique_bikes = fig_unique_bikes.add_axes(rect, polar=True,
                                                    frameon=False)
        ax_unique_bikes.set_theta_zero_location('N')
        ax_unique_bikes.set_theta_direction(1)
        # Add the radial grid with the labels created earlier
        ax_unique_bikes.set_rgrids([0, 1, 2, 3, 4, 5, 6, 7 ,8, 9],
                                   labels=bike_labels, angle=0,
                                   fontsize=14, fontweight='bold',
                                   color='white', verticalalignment='center')

        # Loop through each entry in the dataframe and create a bar
        for i in range(len(unique_bikes_top10)):
            ax_unique_bikes.barh(i, list(unique_bikes_top10['Count'])[i]
                                 *1.5*np.pi/max_value, color=colours[i])

        # Hide all grid elements to make it look nicer
        ax_unique_bikes.grid(False)
        ax_unique_bikes.tick_params(axis='both', left=False, bottom=False,
                                    labelbottom=False, labelleft=True)

        return fig_unique_bikes

    def unique_riders(self):
        '''
        This functions will determine the top 10 riders
        all years of the competition
        It returns a radial bar chart figure
        '''

        # Count up the number of occurances of each riders name in the
        # "Rider" column, and then return the top 10
        unique_riders_top10 = (self.all_successful_runs['Rider'].
                              value_counts().rename_axis('Rider').
                              reset_index(name='Count').head(10))
        # Invert the order so the plot moves outwards
        unique_riders_top10 = unique_riders_top10.reindex(
                              index=unique_riders_top10.index[::-1])

        # Get the count for the riders with most runs
        max_value = max(unique_riders_top10['Count'])
        # colours = ["#011261", "#2c1c79", "#4d248b", "#6c2995", "#8a2f97",
        #            "#a63591", "#c03e82", "#d7486d", "#ea5752", "#f86934",
        #            "#ff8000"]
        # Make labels for the plot of the Name + Count
        rider_labels = [f'   {x} ({v})' for x, v in
                       zip(list(unique_riders_top10['Rider']),
                           list(unique_riders_top10['Count']))]


        # Set up the figure
        fig_unique_riders, rect = self.figure_creator(10,10)

        # Add axis for a polar plot
        # Start bars at top ('N') going CCW (1)
        ax_unique_riders_bg = fig_unique_riders.add_axes(rect,
                                                       polar=True,
                                                       frameon=False)
        ax_unique_riders_bg.set_theta_zero_location('N')
        ax_unique_riders_bg.set_theta_direction(1)

        # Create a grey background for the number of items to plot (10)
        for i in range(len(unique_riders_top10)):
            ax_unique_riders_bg.barh(i, max_value*1.5*np.pi/max_value,
                            color='grey',
                            alpha=0.1)
        ax_unique_riders_bg.axis('off')

        # Add axis for radial chart
        ax_unique_riders = fig_unique_riders.add_axes(rect, polar=True,
                                                    frameon=False)
        ax_unique_riders.set_theta_zero_location('N')
        ax_unique_riders.set_theta_direction(1)
        # Add the radial grid with the labels created earlier
        ax_unique_riders.set_rgrids([0, 1, 2, 3, 4, 5, 6, 7 ,8, 9],
                                   labels=rider_labels, angle=0,
                                   fontsize=14, fontweight='bold',
                                   color='white', verticalalignment='center')

        # Loop through each entry in the dataframe and create a bar
        for i in range(len(unique_riders_top10)):
            ax_unique_riders.barh(i, list(unique_riders_top10['Count'])[i]
                                 *1.5*np.pi/max_value, color=colours[i])

        # Hide all grid elements to make it look nicer
        ax_unique_riders.grid(False)
        ax_unique_riders.tick_params(axis='both', left=False, bottom=False,
                                    labelbottom=False, labelleft=True)

        return fig_unique_riders

    def records_broken(self):
        '''
        This function will determine when records are broken in each category
        and then plot them as a line graph to show the increas
        '''
        # Select rows where a record was broken, and keep columns for Date,
        # speed, record (y/n), and what record was attempted
        record_speeds = (self.all_successful_runs[
                         self.all_successful_runs['Record (y/n)']
                         == 'Y'][["Date", "Speed (MPH)",
                                  "Record (y/n)", "Record Attempt"]])

        # Create a list of unique race categories
        race_categories = record_speeds['Record Attempt'].unique().tolist()

        records = {}
        # Loop through each category, selecting rows that broke it, and save
        # them to the records dictionary
        for category in race_categories:
            temp_record = (record_speeds[record_speeds['Record Attempt']
                                         == category])
            records[category] = {'Date': [], 'Speed': []}
            records[category]['Date'] = temp_record['Date'].tolist()
            records[category]['Speed'] = temp_record['Speed (MPH)'].tolist()

        # Plot the times each record has been broken
        for value in records.items():
            plt.plot(value['Date'], value['Speed'])

    def average_speed_per_year(self):
        '''
        This will calcualte the average speed per year of the men's and women's
        leg powered runs (the two most popular categories)
        It will also calculate the first and third quartile to plot around the
        mean

        It will return a line chart with the mean and quartiles plotted
        '''
        # colours = ["#011261", "#2c1c79", "#4d248b", "#6c2995", "#8a2f97",
        #            "#a63591", "#c03e82", "#d7486d", "#ea5752", "#f86934",
        #            "#ff8000"]

        # Create the figure
        fig_average_speed_per_year, rect = self.figure_creator(10,10)

        # Create the axis
        ax_avg_speed_per_year = (fig_average_speed_per_year
                                 .add_axes(rect, frameon=False))

        # Set up the dictionaries for holding the speeds and quartiles
        men_avg_speed = {}
        men_quartiles = {}
        women_avg_speed = {}
        women_quartiles = {}

        # Long loop function that selects the category (Men then Women),
        # Converts the speeds to a number and replaces non-numerical values
        # with NaNs which are then removed
        # The 1st and 3rd quartile, and mean are then calcualted
        # This process is done for the Mens Leg then Womens Leg
        for key, value in self.race_information_dict.items():
            men_avg_speed[key] = value[value['Record Attempt']  == 'Mens Leg']
            men_avg_speed[key] = (men_avg_speed[key][men_avg_speed[key]
                                                     ['Course']  == 5])
            men_avg_speed[key]['Speed (MPH)'] = (pd.to_numeric(
                                            men_avg_speed[key]['Speed (MPH)'],
                                            errors='coerce'))
            men_avg_speed[key] = (men_avg_speed[key]
                                  .dropna(subset=['Speed (MPH)']))

            men_quartiles[key] = (men_avg_speed[key]['Speed (MPH)']
                                  .quantile([0.25,0.75], interpolation='lower'))
            men_avg_speed[key] = (men_avg_speed[key].loc[:, 'Speed (MPH)']
                                  .mean())

            women_avg_speed[key] = value[value
                                         ['Record Attempt']  == 'Womens Leg']
            women_avg_speed[key] = (women_avg_speed[key][women_avg_speed[key]
                                                         ['Course']  == 5])
            women_avg_speed[key]['Speed (MPH)'] = (pd.to_numeric(
                                            women_avg_speed[key]
                                            ['Speed (MPH)'],
                                            errors='coerce'))
            women_avg_speed[key] = (women_avg_speed[key]
                                  .dropna(subset=['Speed (MPH)']))
            women_quartiles[key] = (women_avg_speed[key]['Speed (MPH)']
                                    .quantile([0.25,0.75], interpolation='lower'))
            women_avg_speed[key] = (women_avg_speed[key].loc[:, 'Speed (MPH)']
                                  .mean())

        # The means of the Mens and Womens catefories are plotted by year
        ax_avg_speed_per_year.plot(list(men_avg_speed.keys()),
                                   list(men_avg_speed.values()),
                                   color=colours[3],
                                   label = "Average Men's Speed")
        ax_avg_speed_per_year.plot(list(women_avg_speed.keys()),
                                   list(women_avg_speed.values()),
                                   color=colours[-1],
                                   label = "Average Women's Speed")


        # Format the plot style
        ax_avg_speed_per_year = self.plot_formatter(ax_avg_speed_per_year,
                                                    y_label="Speed (MPH)")

        women_fill_list_lower = []
        women_fill_list_upper = []
        men_fill_list_lower = []
        men_fill_list_upper = []
        # This loops through the quartiles items and adds them to the
        # lower (1st) and upper (3rd) lists for plotting an area
        for key, value in women_quartiles.items():
            women_fill_list_lower.append(women_quartiles[key][0.25])
            women_fill_list_upper.append(women_quartiles[key][0.75])
            men_fill_list_lower.append(men_quartiles[key][0.25])
            men_fill_list_upper.append(men_quartiles[key][0.75])

        # Plots the areas between the 1st and 3rd quartiles for the men
        # and women
        ax_avg_speed_per_year.fill_between(list(women_quartiles.keys()),
                                           women_fill_list_upper,
                                           women_fill_list_lower,
                                           interpolate=True, color=colours[-2],
                                           alpha=0.3)
        ax_avg_speed_per_year.fill_between(list(men_quartiles.keys()),
                                           men_fill_list_upper,
                                           men_fill_list_lower,
                                           interpolate=True, color=colours[4],
                                           alpha=0.3)
        ax_avg_speed_per_year.set_ylim(0,100)

        ax_avg_speed_per_year.legend()

        return fig_average_speed_per_year

    def records_per_year(self):
        '''
        Determines how many records were broken each year. Records broken
        multiple times in the same year are coutned individually
        
        Returns a bar chart of records broken by year
        '''
        #colours = ["#011261", "#2c1c79", "#4d248b", "#6c2995", "#8a2f97",
        #           "#a63591", "#c03e82", "#d7486d", "#ea5752", "#f86934",
        #           "#ff8000"]

        colours_records = ["#011261", "#1b186f", "#301d7b", "#422285", "#54268e",
                 "#652893", "#762b97", "#862e98", "#973196", "#a63591",
                 "#b53989", "#c33f80", "#d04475", "#dc4b67", "#e65358",
                 "#ef5d49", "#f66738", "#fc7324", "#ff8000" ]

        # Set up figure
        fig_records_per_year, rect = self.figure_creator(10,10)
        # Set up axis
        ax_records_per_year = fig_records_per_year.add_axes(rect, frameon=False)

        records_per_year = {}
        x_years = []
        # Iterate through the race information and select rows where a record
        # was broken, add to dataframe for each year, save the years as a list
        # Plot a bar for each year
        for index, (key, value) in enumerate(self.race_information_dict.items(),
                                             start=0):
            records_per_year[key] = value[value['Record (y/n)']  == 'Y']
            x_years.append(key)
            ax_records_per_year.bar(key, len(records_per_year[key]),
                                    color=colours_records[index])

        # Format the axis - additional params to the function
        ax_records_per_year = self.plot_formatter(ax_records_per_year,
                                                  y_label="Records Broken")
        ax_records_per_year.xaxis.set_major_formatter(lambda s, i : f'{s:,.0f}')
        ax_records_per_year.xaxis.set_tick_params(pad=2, labelbottom=True,
                                                  bottom=False, labelsize=14,
                                                  labelrotation=0, color='white')
        tick_points = [i for i in range(0, len(x_years))]
        ax_records_per_year.set_xticks(tick_points, x_years,
                                       color='white', fontsize=12,
                                       fontweight='bold', rotation=45,
                                       ha="right")

        return fig_records_per_year

    def stats(self):
        '''
        This function is for determining general stats about the races
        Such as the total distance travelled, and number of unique riders, 
        vehicles, and countries
        '''
        # How many successful runs have there been
        successful_runs = len(self.all_successful_runs)

        # What is the total length ridden by all riders over the years
        total_lenth = self.all_successful_runs['Course'].sum()

        # How many unique vehicle have raced
        unique_vehicles = (len(self.all_successful_runs['Vehicle Name']
                               .value_counts().rename_axis('Vehicle Name')
                               .reset_index(name='Count')))

        # How many unique riders have raced
        unique_riders = (len(self.all_successful_runs['Rider'].value_counts()
                             .rename_axis('Rider').reset_index(name='Count')))

        # How many unique countries have riders come from
        unique_countries = (len(self.all_successful_runs['Country']
                                .value_counts().rename_axis('Country')
                                .reset_index(name='Count')))

        # How many entires there are per category
        category_entries = (self.all_successful_runs['Record Attempt'].
                              value_counts().rename_axis('Record Attempt').
                              reset_index(name='Count'))

        # How many records have been broken
        broken_records = (self.all_successful_runs['Record (y/n)'].
                              value_counts().rename_axis('Record (y/n)').
                              reset_index(name='Count'))

        print(f'There have been {successful_runs} successful runs')
        print(f'{total_lenth} miles have been riden in total')
        print(f'There have been {unique_vehicles} unique vehicles and'
              f' {unique_riders} riders from {unique_countries} countries')
        print(f'The number of entires per category is: {category_entries}')
        print(f'The number of records broken is: {broken_records}')
        return

    def arion(self):
        '''
        This function is specifically for gathering information about the Arion
        series of bikes
        '''
        # Create a dataframe just for Arion, and convert speed and course
        # columns to numbers, dropping the DNF and DNS rows
        arion_df = (self.all_successful_runs[self.all_successful_runs
                                             ['Vehicle Name']
                                             .str.contains('Arion')])
        arion_df['Speed (MPH)'] = (pd.to_numeric(arion_df['Speed (MPH)'],
                                errors='coerce'))
        arion_df = arion_df.dropna(subset=['Speed (MPH)'])
        arion_df['Course'] = (pd.to_numeric(arion_df['Course'],
                                errors='coerce'))

        # Total distance riden by Arion bikes
        dist_ridden = arion_df['Course'].sum()

        # Total number of runs by Arion bikes
        runs = len(arion_df)

        # Top speed attained by Arion bikes
        top_speed = arion_df['Speed (MPH)'].max()
        print(f'runs: {runs}, dist: {dist_ridden} miles, top speed: {top_speed} MPH')

        return

    def wind_to_speed(self):
        '''
        This function investigates how the wind speed affects the riders speeds
        and the wind effects in the am or pm of racing
        
        Returns two plots, a scatter of wind speed vs rider speed, and a bar
        chart of % successful or failed runs for am and pm
        '''
        # colours = ["#011261", "#2c1c79", "#4d248b", "#6c2995", "#8a2f97",
        #            "#a63591", "#c03e82", "#d7486d", "#ea5752", "#f86934",
        #            "#ff8000"]

        # Copys the successful runs dataframe
        wind_df = self.all_successful_runs.copy()


        # Converts speed colum to numerical, changing non-numerical to NaN
        # which are then removed
        # Changes wind speed to a numerical field
        wind_df['Speed (MPH)'] = (pd.to_numeric(wind_df['Speed (MPH)'],
                                errors='coerce'))
        wind_df = wind_df.dropna(subset=['Speed (MPH)'])
        wind_df['Wind Speed (ft/min)'] = (pd.to_numeric(
                                                wind_df['Wind Speed (ft/min)'],
                                                errors='coerce'))

        wind_df = wind_df.dropna(subset=['Wind Speed (ft/min)'])

        # Create a new database of just illegal wind runs, then two more
        # one for am and one for pm
        # Creates lists of the wind speeds and riders speeds for am and pm
        wind_df_illegal = wind_df[wind_df['Legal Wind (y/n)']  == 'N']
        wind_df_illegal_am = wind_df_illegal[wind_df_illegal['Morning/evening']  == 'AM']
        speed_illegal_am = wind_df_illegal_am['Speed (MPH)'].tolist()
        wind_illegal_am = wind_df_illegal_am['Wind Speed (ft/min)'].tolist()
        wind_df_illegal_pm = wind_df_illegal[wind_df_illegal['Morning/evening']  == 'PM']
        speed_illegal_pm = wind_df_illegal_pm['Speed (MPH)'].tolist()
        wind_illegal_pm = wind_df_illegal_pm['Wind Speed (ft/min)'].tolist()

        # Create a new database of just legal wind runs this time
        # Then two more, one for am and one for pm
        # Creates lists of the wind speeds and riders speeds for am and pm
        wind_df_legal = wind_df[wind_df['Legal Wind (y/n)']  == 'Y']
        wind_df_legal_am = wind_df_legal[wind_df_legal['Morning/evening']  == 'AM']
        speed_legal_am = wind_df_legal_am['Speed (MPH)'].tolist()
        wind_legal_am = wind_df_legal_am['Wind Speed (ft/min)'].tolist()
        wind_df_legal_pm = wind_df_legal[wind_df_legal['Morning/evening']  == 'PM']
        speed_legal_pm = wind_df_legal_pm['Speed (MPH)'].tolist()
        wind_legal_pm = wind_df_legal_pm['Wind Speed (ft/min)'].tolist()

        # Create figure for scatter graph
        fig_wind_speed_scatter, rect = self.figure_creator(10,10)

        # Calcualte the total number of runs in the am and pm
        total_am = len(speed_legal_am) + len(speed_illegal_am)
        total_pm = len(speed_legal_pm) + len(speed_illegal_pm)

        # Scatter graph axis
        # Scatter plot of the Legal and Illegal, am and pm runs are plotted
        # independently of one another
        ax_wind_speed_scat = fig_wind_speed_scatter.add_axes(rect, frameon=False)
        ax_wind_speed_scat.scatter(wind_legal_am,speed_legal_am,
                                   label = 'Legal wind am', color=colours[2])
        ax_wind_speed_scat.scatter(wind_illegal_am,speed_illegal_am,
                                   label = 'Illegal wind am', color=colours[4])
        ax_wind_speed_scat.scatter(wind_legal_pm,speed_legal_pm,
                                   label = 'Legal wind pm', color=colours[-3])
        ax_wind_speed_scat.scatter(wind_illegal_pm,speed_illegal_pm,
                                   label = 'Illegal wind pm', color=colours[-1])

        # Plot is formatted, and and x axis label is added
        ax_wind_speed_scat = self.plot_formatter(ax_wind_speed_scat,
                                                 y_label = "Rider Speed (MPH)",
                                                 x_label="Wind Speed (ft/min)",
                                                  decimal=2)
        ax_wind_speed_scat.legend()

        # Figure for bar chart is created
        fig_wind_speed_bar, rect = self.figure_creator(10,10)

        # Axis created
        ax_wind_speed_bar = fig_wind_speed_bar.add_axes(rect, frameon=False)

        # Plot four bars, Legal ad Illegal for each am and pm
        ax_wind_speed_bar.bar("Legal am", len(speed_legal_am)/total_am, color=colours[2])
        ax_wind_speed_bar.bar("Illegal am", len(speed_illegal_am)/total_am, color=colours[3])
        ax_wind_speed_bar.bar("Legal pm", len(speed_legal_pm)/total_pm, color=colours[-2])
        ax_wind_speed_bar.bar("Illegal pm", len(speed_illegal_pm)/total_pm, color=colours[-1])

        # Axis formatted, and y axis label allowed to be decimal
        ax_wind_speed_bar = self.plot_formatter(ax_wind_speed_bar,
                                                y_label="% of am or pm",
                                                decimal=2)

        return fig_wind_speed_scatter, fig_wind_speed_bar

    def wind_to_speed_leg(self):
        '''
        This function investigates how the wind speed affects the riders speeds
        and the wind effects in the am or pm of racing
        It is only for leg powered races (single and multitrack) as the 
        Junior and Arm races are to odifferent to include in the correlation
        
        Returns two plots, a scatter of wind speed vs rider speed, and a bar
        chart of % successful or failed runs for am and pm
        '''
        # colours = ["#011261", "#2c1c79", "#4d248b", "#6c2995", "#8a2f97",
        #            "#a63591", "#c03e82", "#d7486d", "#ea5752", "#f86934",
        #            "#ff8000"]

        # Copys the successful runs dataframe
        wind_df_leg = (self.all_successful_runs[self.all_successful_runs
                                             ['Record Attempt']
                                             .str.contains('Mens Leg*' or 'Womens Leg*')])


        # Converts speed colum to numerical, changing non-numerical to NaN
        # which are then removed
        # Changes wind speed to a numerical field
        wind_df_leg['Speed (MPH)'] = (pd.to_numeric(wind_df_leg['Speed (MPH)'],
                                errors='coerce'))
        wind_df_leg = wind_df_leg.dropna(subset=['Speed (MPH)'])
        wind_df_leg['Wind Speed (ft/min)'] = (pd.to_numeric(
                                                wind_df_leg['Wind Speed (ft/min)'],
                                                errors='coerce'))

        wind_df_leg = wind_df_leg.dropna(subset=['Wind Speed (ft/min)'])

        # Create a new database of just illegal wind runs, then two more
        # one for am and one for pm
        # Creates lists of the wind speeds and riders speeds for am and pm
        wind_df_leg_illegal = wind_df_leg[wind_df_leg['Legal Wind (y/n)']  == 'N']
        wind_df_leg_illegal_am = (wind_df_leg_illegal
                                  [wind_df_leg_illegal['Morning/evening']  == 'AM'])
        speed_illegal_am = wind_df_leg_illegal_am['Speed (MPH)'].tolist()
        wind_illegal_am = wind_df_leg_illegal_am['Wind Speed (ft/min)'].tolist()
        wind_df_leg_illegal_pm = (wind_df_leg_illegal
                                  [wind_df_leg_illegal['Morning/evening']  == 'PM'])
        speed_illegal_pm = wind_df_leg_illegal_pm['Speed (MPH)'].tolist()
        wind_illegal_pm = wind_df_leg_illegal_pm['Wind Speed (ft/min)'].tolist()

        # Create a new database of just legal wind runs this time
        # Then two more, one for am and one for pm
        # Creates lists of the wind speeds and riders speeds for am and pm
        wind_df_leg_legal = wind_df_leg[wind_df_leg['Legal Wind (y/n)']  == 'Y']
        wind_df_leg_legal_am = (wind_df_leg_legal
                                [wind_df_leg_legal['Morning/evening']  == 'AM'])
        speed_legal_am = wind_df_leg_legal_am['Speed (MPH)'].tolist()
        wind_legal_am = wind_df_leg_legal_am['Wind Speed (ft/min)'].tolist()
        wind_df_leg_legal_pm = (wind_df_leg_legal
                                [wind_df_leg_legal['Morning/evening']  == 'PM'])
        speed_legal_pm = wind_df_leg_legal_pm['Speed (MPH)'].tolist()
        wind_legal_pm = wind_df_leg_legal_pm['Wind Speed (ft/min)'].tolist()

        # Create figure for scatter graph
        fig_wind_speed_scatter, rect = self.figure_creator(10,10)

        # Calcualte the total number of runs in the am and pm
        total_am = len(speed_legal_am) + len(speed_illegal_am)
        total_pm = len(speed_legal_pm) + len(speed_illegal_pm)
        legal_perc = (len(speed_legal_am) + len(speed_legal_pm)) / (total_am + total_pm)
        print(f'Percentage of legal runs is: {legal_perc}')
        # Scatter graph axis
        # Scatter plot of the Legal and Illegal, am and pm runs are plotted
        # independently of one another
        ax_wind_speed_scat = fig_wind_speed_scatter.add_axes(rect, frameon=False)
        ax_wind_speed_scat.scatter(wind_legal_am,speed_legal_am,
                                   label = 'Legal wind am', color=colours[2])
        ax_wind_speed_scat.scatter(wind_illegal_am,speed_illegal_am,
                                   label = 'Illegal wind am', color=colours[4])
        ax_wind_speed_scat.scatter(wind_legal_pm,speed_legal_pm,
                                   label = 'Legal wind pm', color=colours[-3])
        ax_wind_speed_scat.scatter(wind_illegal_pm,speed_illegal_pm,
                                   label = 'Illegal wind pm', color=colours[-1])

        # Plot is formatted, and and x axis label is added
        ax_wind_speed_scat = self.plot_formatter(ax_wind_speed_scat,
                                                 y_label="Rider Speed (MPH)",
                                                 x_label="Wind Speed (ft/min)")
        ax_wind_speed_scat.legend(edgecolor='#393d5c', facecolor='#25253c',
                                  framealpha=1, labelcolor="white",
                                  loc="lower right")
        ax_wind_speed_scat.axvline(x=326.7716, color='red', linestyle='--')
        ax_wind_speed_scat.text(330, 90, "Wind Speed Cutoff", fontsize=14,
                                color='white')


        fig_spearman_wind, rect = self.figure_creator(10,10)

        #Spearman ranking calculations
        # Axis created
        ax_spearman_wind = fig_spearman_wind.add_axes(rect, frameon=False)

        # Axis formatted, and y axis label allowed to be decimal
        ax_spearman_wind = self.plot_formatter(ax_spearman_wind,
                                               y_label="Wind",
                                               x_label="Wind Speed (ft/min)",
                                               decimal=2)
        ax_spearman_wind.xaxis.set_label_position("bottom")
        ax_spearman_wind.xaxis.set_major_formatter(lambda s, i : f'{s:,.0f}')
        ax_spearman_wind.xaxis.set_major_locator(MaxNLocator(integer=False))
        ax_spearman_wind.xaxis.set_tick_params(pad=2, labeltop=False,
                                                  labelbottom=True,
                                                  bottom=False,
                                                  labelsize=14, color='white')
        ax_spearman_wind.tick_params(axis='x', colors='white')

        # Create a dataframe of all 5 mile runs
        all_5_runs = wind_df_leg[wind_df_leg['Course']  == 5]
        all_5_runs_legal = all_5_runs[all_5_runs['Legal Wind (y/n)']  == "Y"]
        all_5_runs_illegal = all_5_runs[all_5_runs['Legal Wind (y/n)']  == "N"]

        # Create a dataframe just of the windspeed and rider speed for
        # performing the spearman correlation
        spearman_df_al = all_5_runs[['Wind Speed (ft/min)', 'Speed (MPH)']].copy()
        spearman_df_legal = all_5_runs_legal[['Wind Speed (ft/min)', 'Speed (MPH)']].copy()
        spearman_df_illegal = all_5_runs_illegal[['Wind Speed (ft/min)', 'Speed (MPH)']].copy()

        # Create a heatmap of the spearman ranking using seaborn
        spearman_corr_df_all = spearman_df_al.corr(method = 'spearman')
        spearman_corr_df_legal = spearman_df_legal.corr(method = 'spearman')
        spearman_corr_df_illegal = spearman_df_illegal.corr(method = 'spearman')

        # Create a Seaborn heatmap
        sns.heatmap(spearman_corr_df_all, annot = True, cmap=colours)

        # Figure for bar chart is created
        fig_wind_speed_bar, rect = self.figure_creator(10,10)

        # Axis created
        ax_wind_speed_bar = fig_wind_speed_bar.add_axes(rect, frameon=False)

        # Plot four bars, Legal ad Illegal for each am and pm
        ax_wind_speed_bar.bar("Legal am", len(speed_legal_am)/total_am, color=colours[2])
        ax_wind_speed_bar.bar("Illegal am", len(speed_illegal_am)/total_am, color=colours[3])
        ax_wind_speed_bar.bar("Legal pm", len(speed_legal_pm)/total_pm, color=colours[-2])
        ax_wind_speed_bar.bar("Illegal pm", len(speed_illegal_pm)/total_pm, color=colours[-1])

        # Axis formatted, and y axis label allowed to be decimal
        ax_wind_speed_bar = self.plot_formatter(ax_wind_speed_bar,
                                                y_label="% of am or pm",
                                                x_label="Wind Speed (ft/min)",
                                                decimal=2)

        return fig_wind_speed_scatter, fig_wind_speed_bar

    def random_forest_analysis(self):
        '''
        Creates a random forest regression of the Mens and Womens races
        To determine the correlation between rider, vehicle, wind speed,
        and time of day
        '''
        mens_races = (self.all_successful_runs[self.all_successful_runs
                                                    ['Record Attempt']
                                                    .isin(['Mens Leg'])])
        womens_races = (self.all_successful_runs[self.all_successful_runs
                                                    ['Record Attempt']
                                                    .isin(['Womens Leg'])])
        rfr_analysis = whpsc_random_forest.RaceRandomForest(mens_races, womens_races)

        (fig_rfr_predictions, fig_rfr_residuals,
         partial_dependency_plot_women,
         partial_dependency_plot_men) = rfr_analysis.random_forest_model()

        return (fig_rfr_predictions, fig_rfr_residuals,
                partial_dependency_plot_women, partial_dependency_plot_men)

if __name__ == '__main__':
    race_info = RaceInformation(file_path)
    analysis = RaceAnalysis(race_info)
    # fig_nationality_image = analysis.rider_nationality()
    # fig_unique_bikes_image = analysis.unique_bikes()
    # fig_average_speeds = analysis.average_speed_per_year()
    # fig_records_per_year_image = analysis.records_per_year()
    # analysis.arion()
    analysis.stats()
    # fig_unique_riders_image = analysis.unique_riders()
    # fig_wind_speed_scatter, fig_wind_speed_bar = analysis.wind_to_speed_leg()
    # (fig_rfr_predictions, fig_rfr_residuals,
    #  partial_dependency_plot_women,
    #  partial_dependency_plot_men) = analysis.random_forest_analysis()


    # fig_nationality_image.savefig('images/rider_nationality.png', dpi=300, bbox_inches='tight')
    # fig_unique_bikes_image.savefig('images/unique_bikes.png', dpi=300, bbox_inches='tight')
    # fig_average_speeds.savefig('images/avg_speeds.png', dpi=300, bbox_inches='tight')
    # fig_unique_riders_image.savefig('images/unique_riders.png', dpi=300, bbox_inches='tight')
    # fig_wind_speed_scatter.savefig('images/wind_speed_scatter.png', dpi=300, bbox_inches='tight')
    # fig_wind_speed_bar.savefig('images/wind_speed_bar.png', dpi=300, bbox_inches='tight')
    # fig_rfr_predictions.savefig('images/rfr_plot.png', dpi=300, bbox_inches='tight')
    # fig_rfr_residuals.savefig('images/rfr_residuals.png', dpi=300, bbox_inches='tight')#
    # fig_records_per_year_image.savefig('rimages/ecord.png', dpi=300, bbox_inches='tight')
