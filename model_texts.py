# INSERT IMAGE ![Alt Text](https://annalyzin.files.wordpress.com/2016/07/decision-tree-tutorial-animated3.gif)

linear_regression = """
A **Linear Regression** attempts to explain the relationship between one or more inputs and
and an output through a linear equation (a line). When the relationship has been established, we are
able to determine exactly how many of variable *x* leads to an increase or decrease in
variable *y*. When used in a dataset with such colinearity, it is a very efficient and explainable model.

    from sklearn.linear_model import LinearRegression

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    reg.predict(X_test)

    # Use lr.coef_, lr.intercept_ to see coefficients and intercept.
    coef, intercept = lr.coef_, lr.intercept_

For more information on how to implement a Linear Regression in Python, 
[visit scikit-learn's documentation page.](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)
"""

logistic_regression = """
For more information on how to implement a Logistic Regression in Python, 
[visit scikit-learn's documentation page.](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
"""

decision_tree = """

The **Decision Tree** is an algorithm commonly used in tasks of classification and regression.
Similar to a game of *Guess Who* the model attempts to make a prediction by dividing the data 
into a series of subsets based on if-else statements. Doing so, the model can
determine an answer that most closely approximates past results.
    
    from sklearn.tree import DecisionTreeRegressor

    dt = DecisionTreeRegressor()
    dt.fit(X_train, y_train)
    dt.predict(X_test)

For more information on how to implement a Decision Tree in Python, 
[visit scikit-learn's documentation page.](https://scikit-learn.org/stable/modules/tree.html#)
"""

random_forest = """

The **Random Forest** is a *bootstrap aggregation* or *bagging* ensemble method, meaning that it is a collective average of many
other Decision Trees (or a *forest*). In contrast to a Gradient Boosted Tree, a Random Forest creates multiple trees consecutively,
and then returns a single averaged tree, instead of continually improving upon the same tree.

    from sklearn.tree import RandomForestRegressor

    rf = RandomForestRegressor()
    rf.fit(X_train, y_train)
    rf.predict(X_test)

    # Predict the importances of each feature used.
    importances = rf.feature_importances_

For more information on how to implement a Random Forest in Python, 
[visit scikit-learn's documentation page.](https://scikit-learn.org/stable/modules/ensemble.html#forests-of-randomized-trees)
"""

gradient_boosting = """
For more information on how to implement Gradient Boosting in Python, 
[visit scikit-learn's documentation page.](https://scikit-learn.org/stable/modules/ensemble.html#gradient-tree-boosting)
"""

catboost = """
For more information on how to implement CatBoost in Python, 
[visit catboost's documentation page.](https://catboost.ai/docs/concepts/python-usages-examples.html)
"""

knn = """
For more information on how to implement k-nearest Neighbors in Python, 
[visit scikit-learn's documentation page.](https://scikit-learn.org/stable/modules/neighbors.html#nearest-neighbors-classification)
"""

svm = """
For more information on how to implement Support Vector Machines in Python, 
[visit scikit-learn's documentation page.](https://scikit-learn.org/stable/modules/svm.html)
"""

naive_bayes = """
To be added at a later date.
"""

neural_network = """
To be added at a later date.
"""

texts = {
    'Linear Regression':linear_regression,
    'Logistic Regression':logistic_regression,
    'Decision Tree':decision_tree,
    'Random Forest':random_forest,
    'Gradient Boosting':gradient_boosting,
    'CatBoost':catboost,
    'k-nearest Neighbors':knn,
    'Support Vector Machine':svm,
    'Naive Bayes':naive_bayes,
    'Neural Network':neural_network
}

def retrieve_text(model):
    return texts[model]