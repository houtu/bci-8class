"""
Machine Learning Pipeline
Adapted for the new 8-class trial design:
- task4  : early 0-2.5s, collapsed 4-class MI task
- state2 : late 2.5-5.0s, MI (no-rest) vs Rest
- joint8 : task4 prediction from 0-2.5s joined with
           state2 prediction from 2.5-5.0s
"""

import numpy as np
import pandas as pd
from moabb.pipelines.utils import FilterBank
from mne.decoding import CSP, UnsupervisedSpatialFilter

from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from sklearn.linear_model import Lasso
from sklearn.base import clone
import mrmr
from pyriemann.classification import MDM, TSclassifier
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from pathlib import Path


SEEDS = 42
KFOLD = 5


class METHODS:
    pipeline = {
        "alphaCSP+LDA": Pipeline(steps=[
            ('pca', UnsupervisedSpatialFilter(PCA(), average=False)),
            ('csp', CSP(n_components=8)),
            ('lda', LDA())
        ]),

        "alphaCSP+CCA+LDA": None,
        "FBCSP+LDA": None,
        "mrmrFBCSP+LDA": None,
        "SparseFBCSP+LDA": None,

        "Cov+Tangent+SVM": Pipeline(steps=[
            ('covariances', Covariances("oas")),
            ('tangent_space', TangentSpace(metric="riemann")),
            ('svm', SVC(kernel='linear', random_state=SEEDS, probability=True))
        ]),

        "Cov+Tangent+LDA": Pipeline(steps=[
            ('covariances', Covariances("oas")),
            ('tangent_space', TangentSpace(metric="riemann")),
            ('lda', LDA())
        ]),

        "Cov+Tangent+LR": Pipeline(steps=[
            ('covariances', Covariances("oas")),
            ('ts', TSclassifier())
        ]),

        "Cov+MDM": Pipeline(steps=[
            ('covariances', Covariances("oas")),
            ('mdm', MDM(metric=dict(mean='riemann', distance='riemann')))
        ]),
    }


class Pipeline_ML:
    """
    Run classical ML pipelines for x, y from one of the adapted formulations:
        - task4
        - state2
        - joint8
    """

    def __init__(self, method="Cov+MDM"):
        self.method = method

    def _score_binary(self, pipe_ml, x_test, y_test):
        if hasattr(pipe_ml, "predict_proba"):
            y_score = pipe_ml.predict_proba(x_test)[:, 1]
            return metrics.roc_auc_score(y_test, y_score)

        # fallback for estimators without predict_proba
        if hasattr(pipe_ml, "decision_function"):
            y_score = pipe_ml.decision_function(x_test)
            return metrics.roc_auc_score(y_test, y_score)

        y_pred = pipe_ml.predict(x_test)
        return metrics.accuracy_score(y_test, y_pred)

    def _score_multiclass(self, pipe_ml, x_test, y_test):
        y_pred = pipe_ml.predict(x_test)
        return metrics.accuracy_score(y_test, y_pred)

    def _pipe(self, x_train, y_train, x_test, y_test):
        """
        Fit one fold and return:
            pipe_ml : trained model
            r       : fold score
        """
        if "FBCSP" in self.method:
            fbcsp = make_pipeline(FilterBank(CSP(n_components=4)))
            fbcsp.fit(x_train, y_train)
            ft_train = fbcsp.transform(x_train)
            ft_test = fbcsp.transform(x_test)

            if "Sparse" in self.method:
                clf = Lasso(alpha=0.05, random_state=SEEDS, selection="random")
                clf.fit(ft_train, y_train)
                idx = [i for i, v in enumerate(clf.coef_) if v != 0]
                if len(idx) == 0:
                    idx = list(range(ft_train.shape[1]))
                ft_train_select = ft_train[:, idx]
                ft_test_select = ft_test[:, idx]

            elif "mrmr" in self.method:
                selected = mrmr.mrmr_classif(
                    X=pd.DataFrame(ft_train),
                    y=pd.Series(y_train),
                    K=max(1, int(0.5 * ft_train.shape[1]))
                )
                ft_train_select = ft_train[:, selected]
                ft_test_select = ft_test[:, selected]

            else:
                ft_train_select = ft_train
                ft_test_select = ft_test

            pipe_ml = Pipeline(steps=[('classifier', LDA())])
            pipe_ml.fit(ft_train_select, y_train)

            if len(np.unique(y_train)) == 2:
                r = metrics.roc_auc_score(
                    y_test,
                    pipe_ml.predict_proba(ft_test_select)[:, 1]
                )
            else:
                r = metrics.accuracy_score(
                    y_test,
                    pipe_ml.predict(ft_test_select)
                )
        else:
            pipe_ml = METHODS.pipeline[self.method]
            if pipe_ml is None:
                raise ValueError(f"Method '{self.method}' is not implemented.")
            pipe_ml.fit(x_train, y_train)

            if len(np.unique(y_train)) == 2:
                r = self._score_binary(pipe_ml, x_test, y_test)
            else:
                r = self._score_multiclass(pipe_ml, x_test, y_test)

        return pipe_ml, r

    def fit_predict(self, x_train, y_train, x_test):
        """
        Fit the selected pipeline and return predictions.

        Returns
        -------
        y_pred : predicted class labels
        y_score : binary probability/decision score when available,
                  otherwise None
        """

        if "FBCSP" in self.method:
            fbcsp = make_pipeline(
                FilterBank(CSP(n_components=4))
            )

            fbcsp.fit(x_train, y_train)

            ft_train = fbcsp.transform(x_train)
            ft_test = fbcsp.transform(x_test)

            if "Sparse" in self.method:
                clf_select = Lasso(
                    alpha=0.05,
                    random_state=SEEDS,
                    selection="random",
                )
                clf_select.fit(ft_train, y_train)

                idx = [
                    i for i, v in enumerate(clf_select.coef_)
                    if v != 0
                ]

                if len(idx) == 0:
                    idx = list(range(ft_train.shape[1]))

                ft_train = ft_train[:, idx]
                ft_test = ft_test[:, idx]

            elif "mrmr" in self.method:
                selected = mrmr.mrmr_classif(
                    X=pd.DataFrame(ft_train),
                    y=pd.Series(y_train),
                    K=max(1, int(0.5 * ft_train.shape[1])),
                )

                ft_train = ft_train[:, selected]
                ft_test = ft_test[:, selected]

            clf = LDA()
            clf.fit(ft_train, y_train)

            y_pred = clf.predict(ft_test)

            if len(np.unique(y_train)) == 2:
                y_score = clf.predict_proba(ft_test)[:, 1]
            else:
                y_score = None

            return y_pred, y_score

        # Non-FBCSP pipelines
        pipe_ml = clone(METHODS.pipeline[self.method])

        if pipe_ml is None:
            raise ValueError(
                f"Method '{self.method}' is not implemented."
            )

        pipe_ml.fit(x_train, y_train)

        y_pred = pipe_ml.predict(x_test)
        y_score = None

        if len(np.unique(y_train)) == 2:

            if hasattr(pipe_ml, "predict_proba"):
                proba = pipe_ml.predict_proba(x_test)

                if proba.ndim == 2 and proba.shape[1] == 2:
                    y_score = proba[:, 1]

            elif hasattr(pipe_ml, "decision_function"):
                y_score = pipe_ml.decision_function(x_test)

        return y_pred, y_score

    def run(self, x, y):
        """
        k-fold CV on already prepared x, y
        """
        x = np.asarray(x)
        y = np.asarray(y).reshape(-1)

        df = pd.DataFrame()
        skf = StratifiedKFold(
            n_splits=KFOLD,
            shuffle=True,
            random_state=SEEDS
        )

        for fold, (train_index, test_index) in enumerate(skf.split(x, y)):
            _, r = self._pipe(
                x[train_index],
                y[train_index],
                x[test_index],
                y[test_index],
            )
            df.loc[f"fold-{fold}", "score"] = r

        return df
