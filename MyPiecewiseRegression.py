import numpy as np
import sklearn.base
from pyomo.environ import (
    ConcreteModel,
    Var, Set,
    Reals, NonNegativeReals, Binary,
    Constraint, ConstraintList,
    Objective, minimize, maximize,
    SolverFactory, SOSConstraint
)
from sklearn import preprocessing

# The generation of the initial set is similar to the algorithm
# used in k-means, which we directly use here.
# https://flothesof.github.io/k-means-numpy.html


def closest_centroid(points, centroids):
    """returns an array containing the index to the nearest centroid for each point"""
    distances = ((points - centroids[:, np.newaxis])**2).sum(axis=2)
    return np.argmin(distances, axis=0)


def sse(y, y_hat):
    return ((y - y_hat) ** 2).sum()


class LSPPiecewiseRegression(sklearn.base.BaseEstimator, sklearn.base.RegressorMixin):
    """
    Fit a convex piecewise linear regression to the data,
    by least-squares partition algorithm.
    
    Reference:
    Convex piecewise-linear fitting
    (Alessandro Magnani, Stephen P. Boyd)
    """

    def __init__(self, n_pieces: int, convex: bool,
                 l_max: int = 100, n_trials: int = 100,
                 reject_merge: bool = False):
        """
        :param n_pieces: max number of linear segments
        :param convex: fit to a convex function if set to True, otherwise concave
        :param l_max: max rounds of fitting for each trial
        :param n_trials: number of repeated trials
        :param reject_merge: when set to True, retry when merge happens, so that the result will exactly consists of n_pieces segments.
        """
    
        self.n_pieces = n_pieces
        self.convex = convex
        self.l_max = l_max
        self.n_trials = n_trials
        self.reject_merge = reject_merge

    def get_initial_partition(self, X, y=None, scale=True):
        if scale:
            X_scale = preprocessing.StandardScaler().fit_transform(X)
        else:
            X_scale = X

        centroids = np.random.normal(X_scale.mean(axis=0), X_scale.std(axis=0),
                                     size=[self.n_pieces, X_scale.shape[1]])
        partition = closest_centroid(X, centroids)
        return partition

    def fit_once(self, X, y, verbose=False):
        """
        Run one trial of least-squares partition algorithm.

        :param X: 2-D numpy array of shape (N, d)
        :param y: 1-D numpy array of shape (d, )
        :returns: the coefficients and SSE.
        """
        partition = self.get_initial_partition(X, y)

        last_coef = np.array([])
        for step in range(self.l_max):
            coef = []
            # Fit an OLS for each partition
            for p in range(self.n_pieces):
                idx = (partition == p)
                # Remove empty partitions
                if not idx.any():
                    #print("Partition %d is empty!" % p)
                    continue
                b, _, _, _ = np.linalg.lstsq(X[idx], y[idx], rcond=None)
                coef.append(b)

            coef = np.array(coef)

            # Generate the new partition
            piece_values = X.dot(np.array(coef).T)
            if self.convex:
                partition = piece_values.argmax(axis=1)
            else:
                partition = piece_values.argmin(axis=1)
            y_hat = piece_values[np.arange(len(X)), partition]

            # Sort the coefficient in lexicographical order
            # before checking stopping condition to detect "switching"
            coef = np.array(sorted(coef.tolist()))

            # Quit if coefficients does not change
            # If reject_merge, we give up when the solution merges
            quit_cond = (
                (
                    self.reject_merge
                    and len(coef) < self.n_pieces
                )
                or
                (
                    len(last_coef) == len(coef)
                    and np.allclose(last_coef, coef)
                )
            )

            # Print debug information if needed
            if verbose:
                print("Step {} SSE {}".format(step, sse(y, y_hat)))

            if quit_cond:
                break
            else:
                last_coef = coef.copy()

        return coef, sse(y, y_hat), len(coef)

    def fit(self, X, y):
        """
        Fit the data repeatedly and keep the best fit.
        To get the coefficients, use attribute reg.coef_

        :param X: 2-D numpy array of shape (N, d)
        :param y: 1-D numpy array of shape (d, )
        :returns: the regressor
        """

        if X.shape[0] != y.shape[0]:
            raise ValueError("Dimension Mismatch: " +
                             str(X.shape) + str(y.shape))

        self.sse_ = np.inf
        self.coef_ = None

        for _ in range(self.n_trials):
            if self.reject_merge:
                # Sample until the coefficient length is what we want
                coef, err, len_coef = self.fit_once(X, y)
                while len_coef != self.n_pieces:
                    coef, err, len_coef = self.fit_once(X, y)
            else:
                coef, err, _ = self.fit_once(X, y)

            if err < self.sse_:
                self.coef_ = coef
                self.sse_ = err
        return self

    def predict(self, X, y=None):
        """
        Returns the prediction given new data.
        :param X: 2-D numpy array of shape (N, d)
        :param y: ignored
        :returns: 1-D numpy array of (N, ) representing the prediction
        """
        piece_values = X.dot(np.array(self.coef_).T)
        if self.convex:
            return piece_values.max(axis=1)
        else:
            return piece_values.min(axis=1)

    def predict_region(self, X, y=None):
        """
        Returns the piece index on the fitted plane.
        :param X: 2-D numpy array of shape (N, d)
        :param y: ignored
        :returns: 1-D numpy array of (N, ) representing the index of the plane
        """

        piece_values = X.dot(np.array(self.coef_).T)
        if self.convex:
            return piece_values.argmax(axis=1)
        else:
            return piece_values.argmin(axis=1)


class OptPiecewiseRegression(sklearn.base.BaseEstimator, sklearn.base.RegressorMixin):
    """
    Fit a convex piecewise linear regression to the data,
    by solving an exact mixed-integer quadratic program.
    """

    def __init__(self, convex, n_pieces, M=1000.0):
        """
        :param convex: fit to a convex function if set to True, otherwise concave
        :param n_pieces: number of linear segments
        :param M: big-M constant in integer constraints to represent logical constraints
        """
        self.convex = convex
        self.M = M
        self.n_pieces = n_pieces

    def build_opt_model(self, X, y):
        """
        Build an pyomo MIQP model that minimizes the MSE for the data.
        The model then can be solved or exported. model.b are the coefficients.

        :param X: 2-D numpy array of shape (N, d)
        :param y: 1-D numpy array of shape (d, )
        :returns: (unsolved) pyomo model of the fitting problem

        """
        N_rows, N_cols = X.shape
        row_index = list(range(N_rows))
        column_index = list(range(N_cols))
        piece_index = list(range(self.n_pieces))

        X = X.tolist()
        y = y.tolist()

        model = ConcreteModel()
        model.t = Var(row_index, within=Reals)
        model.m = Var(row_index, within=Reals)
        model.b = Var(piece_index, column_index, within=Reals)
        model.s = Var(row_index, piece_index, within=Binary)

        def MinLTRule(model, i, p):
            return model.m[i] <= sum(model.b[p, j] * X[i][j] for j in column_index)
        model.MinLT = Constraint(row_index, piece_index, rule=MinLTRule)

        def MinGTRule(model, i, p):
            return model.m[i] >= sum(model.b[p, j] * X[i][j] for j in column_index) - (1-model.s[i, p]) * self.M
        model.MinGT = Constraint(row_index, piece_index, rule=MinGTRule)

        # def SOS1Rule(model, i):
        #    return sum(model.s[i, p] for p in piece_index) == 1
        # model.SOS1 = Constraint(row_index, rule=SOS1Rule)
        # =====
        # Alternative:
        model.SOS1 = SOSConstraint(
            row_index, var=model.s, index=Set(piece_index), sos=1)
        model.lg = Constraint(row_index, rule=lambda model, i: sum(
            model.s[i, p] for p in piece_index) >= 1)

        def TConstraintRule(model, i):
            return model.t[i] == y[i] - model.m[i] 
        model.TConstraint = Constraint(row_index, rule=TConstraintRule)

        # Maintain the order of the pieces so there might be
        # no duplicates
        def OrderConstraintRule(model, p):
            return model.b[p, 0] <= model.b[p+1, 0]
        model.OrderConstraint = Constraint(
            list(range(self.n_pieces - 1)),
            rule=OrderConstraintRule
        )

        model.obj = Objective(
            expr=(1 / N_rows) * sum(model.t[i]
                                    * model.t[i] for i in row_index),
            sense=minimize
        )
        return model

    def set_warmstart_values(self, X: np.array, y: np.array,
                             warmstart_coef: np.array):
        """
        Internal method used to warm start the model.
        Set the variables to match warmstart_coef.
        """

        # Shape checks
        assert(warmstart_coef.shape == (self.n_pieces, X.shape[1]))

        # Need to negate the coefficient for convex models
        # in order to simulate "min" function
        flip_coef = -1 if self.convex else 1
        coef = flip_coef * warmstart_coef

        # Need to set:
        # t: (row) residues
        # m: (row) predicted min values
        # b: (n_piece, column) coefficients
        # s: (row, piece) active component in min function
        pred_values = X.dot(coef.T)
        active_index = pred_values.argmin(axis=1)
        b = coef
        s = np.zeros((X.shape[0], self.n_pieces), dtype=int)
        s[np.arange(len(X)), active_index] = 1
        m = pred_values[np.arange(len(X)), active_index]
        t = y - m

        def init_model_param(model_var: Var, arr: np.array):
            """
            Copy array into model initializer.
            """
            # Size checking
            block_size = np.array(model_var._index).max(axis=0) + 1
            array_size = arr.shape
            assert(np.all(block_size == array_size))

            if arr.ndim == 1:
                for i, x in enumerate(arr):
                    model_var[i] = x
            elif arr.ndim == 2:
                for i, x in enumerate(arr):
                    for j, y in enumerate(x):
                        model_var[i, j] = y
            else:
                raise IndexError("Param Dimension Mismatch" + str(arr))

        # Assign it to the model
        init_model_param(self.opt_model_.b, b)
        init_model_param(self.opt_model_.s, s)
        init_model_param(self.opt_model_.m, m)
        init_model_param(self.opt_model_.t, t)

    def fit(self, X_arr: np.array, y_arr: np.array,
            warmstart_coef: np.array = None, verbose:bool=False):
        """
        Fit the convex/concave PWA model of n_pieces to data (X_arr, y_arr).
        Requires pyomo cplex solver.
        The resulting coefficients will be stored in attribute coef_

        :param X_arr: 2-D numpy array of shape (N, d)
        :param y_arr: 1-D numpy array of shape (d, )
        :param warmstart_coef: initialize the model with min(b^T x) before solving if not None.
        :param verbose: print pyomo solving details.
        """
        # Dimension check
        X = np.array(X_arr)
        y = np.array(y_arr)

        assert(len(X) == len(y))
        assert(X.ndim == 2)

        # We assume we fit a concave PWA function only.
        # i.e. y = min_i(a_i^T X + b)
        # For convex function we fit on -y instead, and
        # negate the coefficients
        if self.convex:
            y = -y

            # Invert the warmstart_coef as well
            if warmstart_coef is not None:
                warmstart_coef = -warmstart_coef

        # Construct the model
        self.opt_model_ = self.build_opt_model(X, y)

        # Solve the generated model
        opt = SolverFactory("cplex")

        # Set variable values for warmstart if required
        if warmstart_coef is not None:
            self.set_warmstart_values(X, y, warmstart_coef)
            status = opt.solve(self.opt_model_, warmstart=True)
        else:
            status = opt.solve(self.opt_model_)

        if verbose:
            print(status)

        opt_coef = np.reshape(
            self.opt_model_.b[:, :](),
            (self.n_pieces, -1)
        )

        # Invert the coefficient back for convex cases
        if self.convex:
            opt_coef = -opt_coef

        self.coef_ = np.array(sorted(opt_coef.tolist()))
        return self

    def predict(self, X, y=None):
        """
        Predicts the value of the regression funciton on new data.
    
        :param X: 2-D numpy array of shape (N, d)
        :param y: ignored
        """
        if self.convex:
            return X.dot(self.coef_.T).max(axis=1)
        else:
            return X.dot(self.coef_.T).min(axis=1)
