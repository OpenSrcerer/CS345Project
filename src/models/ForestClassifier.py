import pandas as pd
import matplotlib.pyplot as plt
import logging

from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
from sklearn.ensemble import RandomForestClassifier


class ForestClassifier:
    def feature_importance(self, columns, classifier):
        features = list(zip(columns, classifier.feature_importances_))
        sorted_features = sorted(features, key=lambda x: x[1] * -1)

        keys = [value[0] for value in sorted_features]
        values = [value[1] for value in sorted_features]
        return pd.DataFrame(data={'feature': keys, 'value': values})

    def evaluate(self, dataset):
        logging.info("[CLASSIFIER] Dropping fraudRisk on axis 1")
        df_X = dataset.drop('fraudRisk', axis=1)
        df_y = dataset[['fraudRisk']]

        X = df_X.values
        y = df_y.values

        logging.info("[CLASSIFIER] Label Binarizer...")
        y = LabelBinarizer().fit_transform(y)

        # Test/train data split
        logging.info("[CLASSIFIER] Test/train data split...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        # Oversample only the training data
        logging.info("[CLASSIFIER] Oversampling training data...")
        X_train, y_train = SMOTE(random_state=345).fit_resample(X_train, y_train)

        # Random forest classification
        logging.info("[CLASSIFIER] Creating Random Forest Classifier...")
        classifier = RandomForestClassifier(n_estimators=500, random_state=345, max_depth=5, bootstrap=True,
                                       class_weight='balanced')
        logging.info("[CLASSIFIER] Fitting using Random Forest Classifier...")
        classifier = classifier.fit(X_train, y_train)

        # Evaluate the model
        logging.info("[CLASSIFIER] Evaluated model, displaying stats.")

        ConfusionMatrixDisplay.from_estimator(classifier, X_test, y_test)
        RocCurveDisplay.from_estimator(classifier, X_test, y_test, name="RF Model")
        logging.info(f"\n{self.feature_importance(df_X.columns.to_list(), classifier)}")
        plt.show()
