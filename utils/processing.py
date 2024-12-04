import pandas as pd
from sklearn.model_selection import cross_val_score, train_test_split, KFold
from sklearn.linear_model import LogisticRegression,
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn import metrics


def modelisation(X, y, model=LogisticRegression(random_state=42)):
    # applying one hot encoding to Pclass which is categorical variable
    X = pd.get_dummies(X, columns=['Sex', 'Pclass', 'isChild', 'title'], drop_first=False)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # trainning model on train set
    model.fit(X_train, y_train)
    # make prediction
    y_pred = model.predict(X_test)

    train_score = model.score(X_train, y_train)
    print("Train score: {:0.2%}".format(train_score))
    
    # test score
    test_score = model.score(X_test, y_test)
    print("Test score: {:0.2%}".format(test_score))

    # compute confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    disp.plot(cmap="Blues")



from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def Random_Forest(X, y, model=RandomForestClassifier(random_state=42, n_estimators=100, max_depth=10)):
    #if model is None:
        #model = RandomForestClassifier(random_state=42, n_estimators=100, max_depth=10)

    # applying one hot encoding to categorical variables
    X = pd.get_dummies(X, columns=['Sex', 'Pclass', 'isChild', 'title'], drop_first=False)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # training model on train set
    model.fit(X_train, y_train)
    
    # making predictions
    y_pred = model.predict(X_test)

    # training score
    train_score = model.score(X_train, y_train)
    print("Train score: {:0.2%}".format(train_score))

    # test score
    test_score = model.score(X_test, y_test)
    print("Test score: {:0.2%}".format(test_score))

    # compute confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot(cmap="Blues")
    return model
