'''
This module creates a random forest regression model of the mens and womens
200m flying start races at the WHPSC
Challenge from 2001-2019.
The data is collected from http://ihpva.org/whpsc/
'''
import pickle as cPickle
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, explained_variance_score
from sklearn.inspection import PartialDependenceDisplay

class RaceRandomForest:
    '''
    This class will create two separate regression models, one for the Mens
    200m flyign start and oen for the womens 200m flying start
    Plots are produced of regression sv actual, the residuals, and partial
    dependence plots of each variable
    '''
    def __init__(self, mens_races, womens_races):
        self.mens_races = mens_races
        self.womens_races = womens_races

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

    def label_encoding(self, dataframe_to_encode):
        '''
        Takes in a dataframe and encodes the categorical columns as numbers
        to allow for analysis to be performed. Adds columns with "_Encoded"
        at the end of the dataframe
        Returns encoded_runs which is the encoded dataframe and 
        column_mapping which keeps track of the encoding
        Sister function to decode_dataframe, where the encoded_runs and
        column_mapping are used to turn the numbers back into categories
        '''
        input_dataframe = dataframe_to_encode
        encoded_runs = input_dataframe.copy()
        # Store mappings for each column
        column_mappings = {}
        # Loop through string columns and perform label encoding
        for column in input_dataframe.select_dtypes(include=['object']).columns:
            mapping = {value: index for index, value in
                       enumerate(input_dataframe[column].unique())}
            encoded_runs[column + '_Encoded'] = input_dataframe[column].map(mapping)
            column_mappings[column] = mapping
        return encoded_runs, column_mappings

    def decode_dataframe(self, encoded_dataframe, column_mappings):
        '''
        Takes in an encoded dataframe, and then decodes it back to categories 
        from the column_mappings dictionary
        Returns a decoded dataframe with "_Decoded" columsn at the end
        Sister function to label_encoding, where the encoded_runs and
        column_mapping are created
        '''
        for column, mapping in column_mappings.items():
            reverse_mapping = {index: value for value, index in mapping.items()}
            encoded_dataframe[column + '_Decoded'] = (encoded_dataframe
                                                      [column + '_Encoded']
                                                      .map(reverse_mapping))
        return encoded_dataframe

    def rfr_model_creator(self, model_dataframe, model_name):
        '''
        Creates a Random Forest Regression Model of the Morning/evening, 
        Rider, Vehicle, Wind Speed, to try and predict what the speeds will be
        Only for the Mens and Womens Leg races (single rider)
        '''
        # Encode the categorical variables as numbers for analysis
        # Convert the datetime64 to an int in Unix time
        # Then trim the dataframe down the columns we want and remove any
        # missing values (NaN and "-"), and then only select the 5 mile races
        encoded_runs, column_mappings = (self.label_encoding(
                                        model_dataframe
                                        .dropna(subset=
                                                ['Wind Speed (ft/min)'])))
        encoded_runs['Date_unix'] = ((encoded_runs['Date'] -
                                     pd.Timestamp("1970-01-01"))
                                     // pd.Timedelta('1s'))
        rfr_data = encoded_runs[['Date_unix', 'Course', 'Morning/evening_Encoded',
                                 'Rider_Encoded', 'Vehicle Name_Encoded',
                                 'Speed (MPH)', 'Wind Speed (ft/min)']]
        rfr_data = rfr_data[~rfr_data.apply(lambda row: row.astype(str)
                                            .str.contains('-').any(), axis=1)]
        rfr_data = rfr_data[rfr_data['Course']  == 5]

        # Convert the wind speed column ot a numerical format
        rfr_data.loc[:, 'Wind Speed (ft/min)'] = (rfr_data['Wind Speed (ft/min)']
                                                  .astype(float))

        # Split the dataframe into inputs and outputs
        input_data = rfr_data[['Date_unix', 'Morning/evening_Encoded',
                               'Rider_Encoded', 'Vehicle Name_Encoded',
                               'Wind Speed (ft/min)']]
        output_data = rfr_data[['Speed (MPH)',]].values.ravel()

        # Split the datasets ranomly for training
        in_train, in_test, out_train, out_test = train_test_split(input_data,
                                                                output_data,
                                                                test_size=0.2,
                                                                random_state=4)

        # Create the regression object and train it with the training sets
        rfr = RandomForestRegressor(n_estimators=100)
        rfr.fit(in_train, out_train)
        print("Created Random Forest Regression Fit")
        score = rfr.score(in_test, out_test)
        print(f'R2 score is {score}')

        # Calculate the residuals of the model
        predicted_speed = rfr.predict(in_test)
        actual_speed = out_test
        residuals = out_test - predicted_speed
        print("predicted_speed calculations complete")

        # Calculate MSE and the explained variance of the model
        mse = mean_squared_error(actual_speed.tolist(), predicted_speed.tolist())
        explained_variance = explained_variance_score(actual_speed.tolist(),
                                                      predicted_speed.tolist())
        print("Mean Squared Error:", mse)
        print(f'Explained_variance: {explained_variance}')

        # Determine the importances of each factor
        importances = list(rfr.feature_importances_)
        feature_importances = [(feature, round(importance, 2)) for
                            feature, importance in zip(in_test, importances)]
        feature_importances = sorted(feature_importances,
                                    key = lambda x: x[1], reverse = True)
        print(f'Variable Contributions: {feature_importances}')

        # Save the model to a file
        with open(model_name, 'wb') as file:
            cPickle.dump(rfr, file)

        return (rfr, in_train, predicted_speed, actual_speed, score,
                residuals, mse, explained_variance, feature_importances)

    def random_forest_model(self):
        '''
        Sends the dataframes to the rfr creator function, then plots the
        predictions, residuals, and partial dependence plots
        '''
        # Create a dataframe for the Mens and Womens races
        mens_races = self.mens_races
        womens_races = self.womens_races

        # Create an RFR model for the mens races
        (rfr_men, in_train_men, predicted_speed_men, actual_speed_men,
         score_men, residuals_men, mse_men, explained_variance_men,
         feature_importances_men) = self.rfr_model_creator(mens_races,
                                                           "mens_rfr")

        # Create an RFR model for the womens races
        (rfr_women, in_train_women, predicted_speed_women, actual_speed_women,
         score_women, residuals_women, mse_women, explained_variance_women,
         feature_importances_women) = self.rfr_model_creator(womens_races,
                                                             "womens_rfr")

        # Create partial dependency plots to show how each variable affects
        # the model results for the mens races
        features = [0,1,2,3,4]
        partial_dependency_plot_men = (PartialDependenceDisplay
                                       .from_estimator(rfr_men, in_train_men,
                                                       features,
                                                       line_kw={'color':
                                                                '#ff8000'}))

        # Create a PDP for the womens races
        partial_dependency_plot_women = (PartialDependenceDisplay
                                         .from_estimator(rfr_women,
                                                         in_train_women,
                                                         features,
                                                         line_kw={'color':
                                                                  '#6c2995'}))

        # Iterate through the mens and womens plot to design the plot style
        for plot in [partial_dependency_plot_men, partial_dependency_plot_women]:
            index_pairs = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)]
            for i, j in index_pairs:
                plot.axes_[i][j].set_facecolor("#25253c")
                plot.axes_[i][j].tick_params(axis='x', colors='white')
                for spine in plot.axes_[i][j].spines.values():
                    spine.set_color('white')
            # Set up the plot colours
            plot.figure_.set_facecolor('#25253c')
            plot.figure_.set_edgecolor('#393d5c')
            # Set up date plot
            plot.axes_[0][0].set_xlabel("Date", color="#FFFFFF")
            plot.axes_[0][0].set_xticks([(1.2e9),(1.6e9)],[(2001),(2019)],
                                                           color="#FFFFFF")
            # Set up the AM/PM plot
            plot.axes_[0][1].set_xlabel("Morning/Evening", color="#FFFFFF")
            plot.axes_[0][1].set_xticks([0,1], ['AM', 'PM'], color="#FFFFFF")
            #Set up the Rider plot
            plot.axes_[0][2].set_xlabel("Rider",color="#FFFFFF")
            # Set up the bike plot
            plot.axes_[1][0].set_xlabel("Bike",color="#FFFFFF")
            # Set up the wind speed plot
            plot.axes_[1][1].set_xlabel("Wind Speed", color="#FFFFFF")
            plot.axes_[1][1].set_xticks([0,800], [0,800], color="#FFFFFF")
            # Set up Y axis
            plot.axes_[0][0].set_ylabel("Influence", color="#FFFFFF")
            plot.axes_[1][0].set_ylabel("Influence", color="#FFFFFF")
            plot.axes_[0][0].set_yticks([50,80], [50,80], color="#FFFFFF")
            plot.axes_[1][0].set_yticks([50,80], [50,80], color="#FFFFFF")
            plot.axes_[0][0].tick_params(axis='y', colors='white')
            plot.axes_[1][0].tick_params(axis='y', colors='white')

        # Set up the axis scales for num of riders and bikes that are different
        # between the men and women
        partial_dependency_plot_women.axes_[1][0].set_xticks([0,30], [0,30],
                                                             color="#FFFFFF")
        partial_dependency_plot_women.axes_[0][2].set_xticks([0,35], [0,35],
                                                                    color="#FFFFFF")
        partial_dependency_plot_men.axes_[0][2].set_xticks([0,180], [0,180],
                                                                color="#FFFFFF")
        partial_dependency_plot_men.axes_[1][0].set_xticks([0,150], [0,150],
                                                                color="#FFFFFF")

        # Create the residuals figure
        fig_rfr_residuals, rect = self.figure_creator(10,10)
        ax_rfr_residuals = fig_rfr_residuals.add_axes(rect, frameon=False)
        # Plot the residuals for each prediction
        ax_rfr_residuals.scatter(predicted_speed_men, residuals_men,
                                 color = "#ff8000", label="Men")
        ax_rfr_residuals.scatter(predicted_speed_women, residuals_women,
                                 color = "#6c2995", label="Women")
        ax_rfr_residuals.plot((-1,100),(0,0), color='black', linewidth=1)
        # Format the axis
        ax_rfr_residuals = self.plot_formatter(ax_rfr_residuals,
                                               y_label="Residual",
                                               x_label="Predicted Speed",
                                               decimal=2)
        ax_rfr_residuals.legend(edgecolor='#393d5c', facecolor='#25253c',
                                framealpha=1, labelcolor="white",
                                loc="upper left")

        #Create the prediction vs actual speed figure
        fig_rfr_predictions, rect = self.figure_creator(10,10)
        ax_rfr_predictions = fig_rfr_predictions.add_axes(rect, frameon=False)
        # Plot the actual speed against the predicted speeds for Men and Women
        ax_rfr_predictions.scatter(actual_speed_men, predicted_speed_men,
                                   color = "#ff8000", label="Men")
        ax_rfr_predictions.scatter(actual_speed_women, predicted_speed_women,
                                   color = "#6c2995", label="Women")

        # Plot the R2 line on the graph for the men
        x_residual_line_men = actual_speed_men.tolist()
        y_residual_line_men = predicted_speed_men.tolist()
        coef_men = np.polyfit(x_residual_line_men,y_residual_line_men,1)
        poly1d_fn_men = np.poly1d(coef_men)
        plt.plot(x_residual_line_men, poly1d_fn_men(x_residual_line_men),
                 color='#ea5752')
        r2_text_men = "Men $R^{2}$ = " + str(round(score_men,3))
        ax_rfr_predictions.text(82.5, 70, r2_text_men, fontsize=14, color='white')

        # Plot the R2 line on the graph for the women
        x_residual_line_women = actual_speed_women.tolist()
        y_residual_line_women = predicted_speed_women.tolist()
        coef_women = np.polyfit(x_residual_line_women,y_residual_line_women,1)
        poly1d_fn_women = np.poly1d(coef_women)
        plt.plot(x_residual_line_women, poly1d_fn_women(x_residual_line_women),
                 color='#4d248b')
        r2_text_women = "Women $R^{2}$ = " + str(round(score_women,3))
        ax_rfr_predictions.text(72.5, 60, r2_text_women, fontsize=14, color='white')

        # Axis formatted, and y axis label allowed to be decimal
        ax_rfr_predictions = self.plot_formatter(ax_rfr_predictions,
                                                 y_label="Predicted Speed",
                                                 x_label="Actual Speed",
                                                 decimal=2)
        ax_rfr_predictions.legend(edgecolor='#393d5c', facecolor='#25253c',
                                  framealpha=1, labelcolor="white")

        return (fig_rfr_predictions, fig_rfr_residuals,
                partial_dependency_plot_women, partial_dependency_plot_men)
