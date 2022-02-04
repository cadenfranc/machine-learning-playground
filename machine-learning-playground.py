from catboost.core import CatBoostClassifier
from numpy.lib.shape_base import tile
import pandas as pd
import math
import streamlit as st
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
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


def format_table(s):
    if s.name == 'MAE' or s.name == 'MAPE' or s.name == 'RMSE':
        return ['background-color: #F63366' if i == s.min() else '' for i in s]
    if s.name == 'Bias':
        min_error = abs(s).min()
        return ['background-color: #F63366' if abs(i) == min_error else '' for i in s]
    else:
        return ['background-color: #F63366' if i == s.max() else '' for i in s]


def train_models(task: str, models: list, data: pd.DataFrame, target: str):
    data = data.select_dtypes(['number']).dropna()
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
        
        if model == 'Neural Network':
            pass

        predictions[model] = model_type.fit(X_train, y_train).predict(X_test)

        fig.add_trace(go.Scatter(x=predictions[target], 
                                y=predictions[model], 
                                mode='markers',
                                name=model))
        fig.update_xaxes(title='Actual')
        fig.update_yaxes(title='Predicted')

        if task == 'Regression':
            metrics[model] = [mean_absolute_error(predictions[target], predictions[model]),
                            mean_absolute_error(predictions[target], predictions[model]) / predictions[target].mean(),
                            math.sqrt(mean_squared_error(predictions[target], predictions[model])),
                            (predictions[model] - predictions[target]).sum() / len(predictions[target]),
                            r2_score(predictions[target], predictions[model])]

        if task == 'Classification':
            cm = confusion_matrix(predictions[target], predictions[model])
            metrics[model] = [accuracy_score(predictions[target], predictions[model]),
                            cm[1][1] / (cm[1][1] + cm[0][1]),
                            cm[1][1] / (cm[1][1] + cm[1][0]),
                            f1_score(predictions[target], predictions[model], zero_division=1)]

    fig.update_traces(mode='markers', marker_line_width=1, marker_size=8)
    fig.update_layout(title='Difference Between Actual and Predicted Values')

    metrics = metrics.transpose()

    if task == 'Regression':
        metrics.columns = ['MAE', 'MAPE', 'RMSE', 'Bias', 'R2 Score']
    if task == 'Classification':
        metrics.columns = ['Accuracy', 'Precision', 'Recall', 'F1 Score']

    if len(models) > 1:
        metrics = metrics.style.apply(format_table)
    else:
        st.header('Model Performance')
    
    st.write(fig, metrics)


def main():
    st.sidebar.header('Import Dataset')
    file = st.sidebar.file_uploader('')
    
    if file is not None:
        df = pd.read_csv(file)
        task = st.sidebar.radio('', ['Regression', 'Classification'])
        target = st.sidebar.selectbox('Target Feature', df.keys())
        models = st.sidebar.multiselect('Select Models', [
        'Linear Regression', 
        'Logistic Regression', 
        'Decision Tree', 
        'Random Forest', 
        'Gradient Boosting',
        'CatBoost',
        'k-nearest Neighbors',
        'Support Vector Machine'])

        if len(models):
            if len(models) == 1:
                st.title(models[0])
            else:
                st.title('Multiple Model Comparison')
            train_models(task, models, df, target)
        else:
            st.write(df)

    else:
        st.title('Machine Learning Playground')
        st.markdown('''
        This will be a title page that will have a brief explanation with an overview of each part.
        ''')

if __name__ == "__main__":
    main()
