import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error


class BikeData:
    def __init__(self,data):
        
        '''
        Initializes the BikeData object with a dataset.
        
        Parameters:
         data (DataFrame): pandas DataFrame
        
        '''
        self.data = data
        
    def pointplot_bikedata(self,x,y,hue,ax,title):
        
        '''
        Displays a point plot for the bike data.
        
        Parameters:
        x ( str ): Column from the dataframe for the x_axis.
        y ( str ): Column from the dataframe for the y_axis.
        hue ( str ): Column from the dataframe for color encoding.
        ax (matplotlib.axes.Axes): Axes object obtained from plt.subplots where the plot will be drawn.
        title ( str ): Appropriate title chosen based on the data provided to the function.
        
        '''
        
        
        sns.pointplot(data=self.data, x=x, y=y, hue=hue, ax=ax)
        ax.set(title=title)
        
        
    def barplot_bikedata(self,x,y,ax,title):
        
        '''
        Displays a bar plots for the bike data.
        
        Parameters:
        x ( str ): Column from the dataframe for the x_axis.
        y ( str ): Column from the dataframe for the y_axis.
        
        ax (matplotlib.axes.Axes): Axes object obtained from plt.subplots where the plot will be drawn.
        title ( str ): Appropriate title chosen based on the data provided to the function.
        
        '''
        
        
        sns.barplot(data=self.data, x=x, y=y, ax=ax)
        ax.set(title=title)
        
    def analyze_correlation(dataframe):
        col_list = []
        ind_list = []
        corr_type = []
        corr_strength = []
        corr_list = []
    
        for i in range(len(dataframe.corr().columns)):
            for j in range(len(dataframe.corr().index)):
    
                if dataframe.corr().columns[i] != dataframe.corr().index[j] and dataframe.corr().iloc[i, j] != 0:
    
                    if dataframe.corr().iloc[i, j] < 0:
    
                        if dataframe.corr().iloc[i, j] < -0.7:
                            col_list.append(dataframe.corr().columns[i])
                            ind_list.append(dataframe.corr().index[j])
                            corr_type.append("Negative")
                            corr_strength.append("Strong")
                            corr_list.append(round(dataframe.corr().iloc[i, j], 2))
    
                        elif dataframe.corr().iloc[i, j] < -0.5:
                            col_list.append(dataframe.corr().columns[i])
                            ind_list.append(dataframe.corr().index[j])
                            corr_type.append("Negative")
                            corr_strength.append("Medium")
                            corr_list.append(round(dataframe.corr().iloc[i, j], 2))
    
                    else:
    
                        if dataframe.corr().iloc[i, j] >= 0.7:
                            col_list.append(dataframe.corr().columns[i])
                            ind_list.append(dataframe.corr().index[j])
                            corr_type.append("Positive")
                            corr_strength.append("Strong")
                            corr_list.append(round(dataframe.corr().iloc[i, j], 2))
    
                        elif dataframe.corr().iloc[i, j] > 0.3:
                            col_list.append(dataframe.corr().columns[i])
                            ind_list.append(dataframe.corr().index[j])
                            corr_type.append("Positive")
                            corr_strength.append("Medium")
                            corr_list.append(round(dataframe.corr().iloc[i, j], 2))
    
        # Variable relationship based on correlation
        df_corr = pd.DataFrame({"Column_1": col_list,
                                "Column_2": ind_list,
                                "Relationship_strength": corr_strength,
                                "Relation_type": corr_type,
                                "Correlation": corr_list})
    
        return df_corr
        
    def evaluate_models(X, y):
        
        '''
        Evaluates the Mean absolute deviation for 'Linear Regression', 'Lasso', 'Ridge', 'Random Forest', 'Gradient Boost', 'SVR',             'XGB' models and also displays the kde plots for the residuals and displays important features for the bike data.
        
        Parameters:
        X ( Dataframe ): dataframe with the features only.
        y ( Dataframe ): dataframe with the Target.
        
        
        
        '''
        
        
        # Split the dataset
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
        # Define models
        linear_reg_model = LinearRegression()
        lasso_model = Lasso(alpha=0.1)
        ridge_model = Ridge(alpha=1.0)
        random_forest_model = RandomForestRegressor(random_state=42)
        gradient_boosting_model = GradientBoostingRegressor(random_state=42)
        svr_model = SVR()
        xgb_model = XGBRegressor()
    
        models = [linear_reg_model, lasso_model, ridge_model, random_forest_model, gradient_boosting_model, svr_model, xgb_model]
        model_names = ['Linear Regression', 'Lasso', 'Ridge', 'Random Forest', 'Gradient Boost', 'SVR', 'XGB']
        mad_scores = []
    
        # Calculate MAD for all models
        for model, name in zip(models, model_names):
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
    
            # Take exponential of the log
            y_pred_org = np.expm1(y_pred)
            y_test_org = np.expm1(y_test)
    
            mad = mean_absolute_error(y_test_org, y_pred_org)
    
            mad_scores.append(mad)
            print(f'{name} - Mean Absolute Deviation: {mad}')
    
            residuals = y_test_org - y_pred_org
            plt.figure(figsize=(10, 6))
            sns.histplot(residuals, kde=True, color='blue')
            plt.title(f'Residuals Distribution - {name}')
            plt.xlabel('Residuals')
            plt.ylabel('Frequency')
            plt.show()
    
            # Extract important features
            if hasattr(model, 'feature_importances_'):
                feature_importances = model.feature_importances_
                feature_names = X.columns
                importance_dict = dict(zip(feature_names, feature_importances))
    
                # Sort feature importances in descending order
                sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
    
                # Plot the top 10 features
                top_n = min(10, len(sorted_importance))  # Plot at most top 10 features
                top_features, top_importance = zip(*sorted_importance[:top_n])
    
                plt.figure(figsize=(10, 6))
                plt.bar(top_features, top_importance, color='blue')
                plt.title(f'Top {top_n} Important Features - {name}')
                plt.xlabel('Features')
                plt.ylabel('Importance')
                plt.show()
    
        # Plot MAD
        plt.figure(figsize=(10, 6))
        plt.bar(model_names, mad_scores, color=['blue', 'green', 'orange', 'red', 'purple', 'brown', 'pink'])
        plt.title('Mean Absolute Deviation for Different Models')
        plt.xlabel('Models')
        plt.ylabel('Mean Absolute Deviation')
        plt.show()
