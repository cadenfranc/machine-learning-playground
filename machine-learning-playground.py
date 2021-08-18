from catboost.core import CatBoostClassifier
from numpy.lib.shape_base import tile
import pandas as pd
from pandas.core.frame import DataFrame
from scipy.sparse.construct import random
import streamlit as st
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from catboost import CatBoostRegressor
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from model_texts import retrieve_text

# TODO
# Finish text and example code for all main models.
# Add ability to modify dataframe (one hot / categorical encode, drop).
# Add parameters into each of the individual model views.
# Organize data so that it can be cached if possible. 

def train_models(task: str, models: list, data: pd.DataFrame, target: str):
    data = data.select_dtypes(['number'])
    X, y = data.drop(target, axis=1), data[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42)

    fig = go.Figure()
    metrics = pd.DataFrame()
    predictions = pd.DataFrame(y_test)

    if len(models) == 1:
            st.markdown(retrieve_text(models[0]))

    for model in models:
        
        if model == 'Linear Regression':
            model_type = LinearRegression()
        
        if model == 'Logistic Regression':
            model_type = LogisticRegression()
        
        if model == 'Decision Tree':
            model_type = DecisionTreeRegressor() if task == 'Regression' else DecisionTreeClassifier()
        
        if model == 'Random Forest':
            model_type = RandomForestRegressor() if task == 'Regression' else RandomForestClassifier()

        if model == 'Gradient Boosting':
            model_type = GradientBoostingRegressor() if task == 'Regression' else GradientBoostingClassifier()
        
        if model == 'CatBoost':
            model_type = CatBoostRegressor() if task == 'Regression' else CatBoostClassifier()
        
        if model == 'k-nearest Neighbors':
            model_type = KNeighborsRegressor() if task == 'Regression' else KNeighborsClassifier()
        
        if model == 'Support Vector Machine':
            model_type = make_pipeline(StandardScaler(), SVR())
        
        if model == 'Naive Bayes':
            pass
        
        if model == 'Neural Network':
            pass

        predictions[model] = model_type.fit(X_train, y_train).predict(X_test)

        fig.add_trace(go.Scatter(x=predictions[target], 
                                 y=predictions[model], 
                                 mode='markers',
                                 name=model))
        fig.update_xaxes(title='Actual')
        fig.update_yaxes(title='Predicted')

        metrics[model] = [mean_absolute_error(predictions[target], predictions[model]),
                          mean_absolute_error(predictions[target], predictions[model]) / predictions[target].mean(),
                          mean_squared_error(predictions[target], predictions[model]),
                          r2_score(predictions[target], predictions[model])]

    fig.update_traces(mode='markers', marker_line_width=1, marker_size=8)
    fig.update_layout(title='Difference Between Actual and Predicted Values')

    metrics = metrics.transpose()
    metrics.columns = ['MAE', 'MAPE', 'MSE', 'R2 Score']
    if len(models) > 1:
        metrics = metrics.style.highlight_min(['MAE', 'MAPE', 'MSE'], color='#F63366', axis=0)
    else:
        st.header('Model Performance')
    st.write(fig, metrics)


def main():
    st.sidebar.header('Import Dataset')
    st.sidebar.write('Choose from an existing dataset or upload your own.')
    file = st.sidebar.file_uploader('')

    task = st.sidebar.radio('', ['Classification', 'Regression'])
    if file is not None:
        df = pd.read_csv(file)
        target = st.sidebar.selectbox('Target Feature', df.keys())
    models = st.sidebar.multiselect('Select Models', [
        'Linear Regression', 
        'Logistic Regression', 
        'Decision Tree', 
        'Random Forest', 
        'Gradient Boosting',
        'CatBoost',
        'k-nearest Neighbors',
        'Support Vector Machine',
        'Naive Bayes',
        'Neural Network'])
    
    if file is not None:
        if len(models):
            if len(models) == 1:
                st.title(models[0])
                st.sidebar.header('Model Parameters')
                st.sidebar.write(f'This will be the parameters for the {models[0]} model.')
            else:
                st.title('Multiple Model Comparison')
            train_models(task, models, df, target)
        else:
            st.write(df)

if __name__ == "__main__":
    main()
