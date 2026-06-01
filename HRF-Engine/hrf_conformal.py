"""
HRFConformalPredictor
=====================
Distribution-free prediction sets with guaranteed marginal coverage
for any fitted HarmonicResonanceClassifier_BEAST_14D instance.

Based on split conformal prediction (Papadopoulos et al., 2002).
Coverage guarantee: P(y_true ∈ prediction_set) >= 1 - alpha, exactly.
No distribution assumptions. No asymptotic approximations.
"""

import numpy as np
from sklearn.utils.validation import check_is_fitted, check_array


class HRFConformalPredictor:
    """
    Wraps a fitted HarmonicResonanceClassifier_BEAST_14D and produces
    prediction sets with guaranteed marginal coverage using split
    conformal prediction.

    Parameters
    ----------
    estimator : fitted HarmonicResonanceClassifier_BEAST_14D
        Must already be fitted. This class does not refit.
    alpha : float, default=0.05
        Miscoverage level. Prediction sets contain the true label
        with probability >= 1 - alpha.

    Attributes
    ----------
    calibration_scores_ : ndarray of shape (n_cal,)
        Nonconformity scores computed on the calibration set.
    classes_ : ndarray
        Class labels from the wrapped estimator.
    n_cal_ : int
        Number of calibration samples used.

    Examples
    --------
    >>> clf = HarmonicResonanceClassifier_BEAST_14D(verbose=False)
    >>> clf.fit(X_train, y_train)
    >>> conformal = HRFConformalPredictor(clf, alpha=0.05)
    >>> conformal.calibrate(X_cal, y_cal)
    >>> prediction_sets = conformal.predict_set(X_test)
    >>> report = conformal.coverage_report(X_test, y_test)
    """

    def __init__(self, estimator, alpha=0.05):
        self.estimator = estimator
        self.alpha = alpha
        self.calibration_scores_ = None
        self.classes_ = None
        self.n_cal_ = None

    def _nonconformity_score(self, proba, y):
        """
        Compute nonconformity scores for calibration samples.

        For class y_i, the score is 1 - p(y_i | x_i).
        Low score = highly conforming (model is confident and correct).
        High score = nonconforming (model is wrong or uncertain).

        Parameters
        ----------
        proba : ndarray of shape (n, n_classes)
        y : ndarray of shape (n,) — true integer class labels

        Returns
        -------
        scores : ndarray of shape (n,)
        """
        n = len(y)
        # Map true labels to class indices
        class_to_idx = {cls: i for i, cls in enumerate(self.classes_)}
        true_idx = np.array([class_to_idx[label] for label in y])
        true_proba = proba[np.arange(n), true_idx]
        return 1.0 - true_proba

    def calibrate(self, X_cal, y_cal):
        """
        Compute nonconformity scores on a held-out calibration set.
        Must be called before predict_set() or predict_p_values().

        Parameters
        ----------
        X_cal : array-like of shape (n_cal, n_features)
        y_cal : array-like of shape (n_cal,)

        Returns
        -------
        self
        """
        check_is_fitted(self.estimator, 'weights_')
        X_cal = check_array(X_cal)
        y_cal = np.asarray(y_cal)

        self.classes_ = self.estimator.classes_
        self.n_cal_ = len(y_cal)

        proba = self.estimator.predict_proba(X_cal)
        self.calibration_scores_ = self._nonconformity_score(proba, y_cal)
        return self

    def _quantile_threshold(self):
        """
        Compute the (1-alpha) adjusted quantile of calibration scores.
        The +1 adjustment ensures finite-sample coverage guarantee.
        """
        if self.calibration_scores_ is None:
            raise RuntimeError("Call calibrate() before predict_set().")
        n = self.n_cal_
        level = np.ceil((n + 1) * (1 - self.alpha)) / n
        level = min(level, 1.0)
        return np.quantile(self.calibration_scores_, level)

    def predict_set(self, X, alpha=None):
        """
        Produce prediction sets with guaranteed coverage >= 1 - alpha.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        alpha : float or None
            Override instance alpha if provided.

        Returns
        -------
        prediction_sets : list of lists
            Each inner list contains the class labels in the prediction set
            for that sample. May contain 0 (impossible) or all classes
            (maximally uncertain) for extreme cases.
        """
        if alpha is not None:
            original_alpha = self.alpha
            self.alpha = alpha

        X = check_array(X)
        threshold = self._quantile_threshold()
        proba = self.estimator.predict_proba(X)

        prediction_sets = []
        for i in range(len(X)):
            sample_set = []
            for c_idx, cls in enumerate(self.classes_):
                # Include class if its nonconformity score is <= threshold
                score = 1.0 - proba[i, c_idx]
                if score <= threshold:
                    sample_set.append(cls)
            prediction_sets.append(sample_set)

        if alpha is not None:
            self.alpha = original_alpha

        return prediction_sets

    def predict_p_values(self, X):
        """
        Compute conformal p-values for each class at each test point.

        The p-value for class c at point x is the fraction of calibration
        scores >= the nonconformity score of (x, c). A large p-value means
        the sample is highly conforming to class c.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        p_values : ndarray of shape (n_samples, n_classes)
        """
        X = check_array(X)
        proba = self.estimator.predict_proba(X)
        n_cal = self.n_cal_
        p_values = np.zeros((len(X), len(self.classes_)))

        for i in range(len(X)):
            for c_idx in range(len(self.classes_)):
                test_score = 1.0 - proba[i, c_idx]
                # p-value: fraction of calibration scores >= test score
                p_values[i, c_idx] = (
                    np.sum(self.calibration_scores_ >= test_score) + 1
                ) / (n_cal + 1)

        return p_values

    def coverage_report(self, X_test, y_test, alphas=None):
        """
        Empirically validate coverage and efficiency across alpha levels.

        Parameters
        ----------
        X_test : array-like of shape (n_test, n_features)
        y_test : array-like of shape (n_test,)
        alphas : list of float or None
            Alpha levels to evaluate. Defaults to [0.01, 0.05, 0.10, 0.20].

        Returns
        -------
        report : dict
            Keys: alpha level (float)
            Values: dict with 'coverage', 'avg_set_size', 'empty_sets',
                    'singleton_sets', 'full_sets'
        """
        if alphas is None:
            alphas = [0.01, 0.05, 0.10, 0.20]

        y_test = np.asarray(y_test)
        report = {}

        print(f"\n{'='*60}")
        print(f"  HRF CONFORMAL COVERAGE REPORT")
        print(f"  Calibration size: {self.n_cal_} | Test size: {len(y_test)}")
        print(f"  Classes: {self.classes_}")
        print(f"{'='*60}")
        print(f"  {'Alpha':<8} {'Target':<10} {'Actual':<10} "
              f"{'Avg Set':<10} {'Empty':<8} {'Single':<8} {'Full'}")
        print(f"  {'-'*58}")

        for alpha in alphas:
            sets = self.predict_set(X_test, alpha=alpha)
            covered = sum(
                y_test[i] in sets[i] for i in range(len(y_test))
            )
            coverage = covered / len(y_test)
            set_sizes = [len(s) for s in sets]
            n_classes = len(self.classes_)

            stats = {
                'coverage': coverage,
                'target_coverage': 1 - alpha,
                'avg_set_size': np.mean(set_sizes),
                'empty_sets': sum(s == 0 for s in set_sizes),
                'singleton_sets': sum(s == 1 for s in set_sizes),
                'full_sets': sum(s == n_classes for s in set_sizes),
            }
            report[alpha] = stats

            status = "✅" if coverage >= (1 - alpha) else "❌"
            print(f"  {alpha:<8.2f} {1-alpha:<10.2%} {coverage:<10.2%} "
                  f"{np.mean(set_sizes):<10.2f} "
                  f"{stats['empty_sets']:<8} "
                  f"{stats['singleton_sets']:<8} "
                  f"{stats['full_sets']} {status}")

        print(f"{'='*60}\n")
        return report
