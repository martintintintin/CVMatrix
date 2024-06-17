"""
Contains the CVMatrix class which implements methods for fast computation of training
set kernel matrices in cross-validation using the fast algorithms described in the
paper by O.-C. G. Engstrøm: https://arxiv.org/abs/2401.13185

The implementation is written using NumPy.

Author: Ole-Christian Galbo Engstrøm
E-mail: ole.e@di.ku.dk
"""

from multiprocessing.managers import SyncManager, SharedMemoryManager
from multiprocessing.shared_memory import SharedMemory
from typing import Hashable, Iterable, Union
from multiprocessing import Array, Manager, Value

import numpy as np
from numpy import typing as npt
import time as pytime


class CVMatrix:
    """
    Implements the fast cross-validation algorithms for kernel matrix-based models such
    as PCA, PCR, PLS, and OLS. The algorithms are based on the paper by O.-C. G.
    Engstrøm: https://arxiv.org/abs/2401.13185

    Parameters
    ----------
    cv_splits : Iterable of Hashable with N elements
        An iterable defining cross-validation splits. Each unique value in
        `cv_splits` corresponds to a different fold.

    center_X : bool, optional, default=True
        Whether to center `X` before computation of
        :math:`\mathbf{X}^{\mathbf{T}}\mathbf{X}` and
        :math:`\mathbf{X}^{\mathbf{T}}\mathbf{Y}` by subtracting its row of column-wise
        means from each row. The row of column-wise means is computed on the training
        set for each fold to avoid data leakage.

    center_Y : bool, optional, default=True
        Whether to center `Y` before computation of
        :math:`\mathbf{X}^{\mathbf{T}}\mathbf{Y}` by subtracting its row of column-wise
        means from each row. The row of column-wise means is computed on the training
        set for each fold to avoid data leakage. This parameter is ignored if `Y` is
        `None`.

    scale_X : bool, optional, default=True
        Whether to scale `X` before computation of
        :math:`\mathbf{X}^{\mathbf{T}}\mathbf{X}` and
        :math:`\mathbf{X}^{\mathbf{T}}\mathbf{Y}` by dividing each row with the row of
        `X`'s column-wise standard deviations. Bessel's correction for the unbiased
        estimate of the sample standard deviation is used. The row of column-wise
        standard deviations is computed on the training set for each fold to avoid data
        leakage.

    scale_Y : bool, optional, default=True
        Whether to scale `Y` before computation of
        :math:`\mathbf{X}^{\mathbf{T}}\mathbf{Y}` by dividing each row with the row of
        `X`'s column-wise standard deviations. Bessel's correction for the unbiased
        estimate of the sample standard deviation is used. The row of column-wise
        standard deviations is computed on the training set for each fold to avoid data
        leakage. This parameter is ignored if `Y` is `None`.

    dtype : np.floating, optional, default=np.float64
        The data type used for the computations. The default is `np.float64`.

    copy : bool, optional, default=True
        Whether to make a copy of the input arrays. If `False` and the input arrays are
        already NumPy arrays of type `dtype`, then no copy is made. If `False` and the
        input arrays are not NumPy arrays of type `dtype`, then a copy is made. If
        `True` a copy is always made. If no copy is made, then external modifications
        to `X` or `Y` will result in undefined behavior.
    """

    def __init__(
        self,
        cv_splits: Iterable[Hashable],
        center_X: bool = True,
        center_Y: bool = True,
        scale_X: bool = True,
        scale_Y: bool = True,
        dtype: np.floating = np.float64,
        copy: bool = True,
    ) -> None:
        self.center_X = center_X
        self.center_Y = center_Y
        self.scale_X = scale_X
        self.scale_Y = scale_Y
        self.dtype = dtype
        self.copy = copy
        self.X_total = None
        self.Y_total = None
        self.N = None
        self.K = None
        self.M = None
        self.X_total_mean = None
        self.Y_total_mean = None
        self.XTX_total = None
        self.XTY_total = None
        self.sum_X_total = None
        self.sum_Y_total = None
        self.sum_sq_X_total = None
        self.sum_sq_Y_total = None
        self.val_folds_dict = None
        self._init_val_folds_dict(cv_splits)

    def export_to_multiprocessing_types(
        self, manager: SyncManager, smm: SharedMemoryManager
    ):
        typecode_map = {int: "i", float: "d", bool: "b", str: "s"}
        managed_values = {}
        shared_memory_objects = {}  # Store the SharedMemory objects
        for attr, value in self.__dict__.items():
            if attr == "val_folds_dict":
                continue
                pure_dict = {}
                for k, v in value.items():
                    if isinstance(v, np.ndarray):
                        shm = smm.SharedMemory(size=v.nbytes)
                        np_array = np.ndarray(v.shape, dtype=v.dtype, buffer=shm.buf)
                        np_array[:] = v[:]
                        pure_dict[k] = (v.shape, v.dtype)
                        shared_memory_objects[k] = shm
                    else:
                        assert False
                managed_values[attr] = manager.dict(pure_dict)
            elif isinstance(value, np.ndarray):
                shm = smm.SharedMemory(size=value.nbytes)
                np_array = np.ndarray(value.shape, dtype=value.dtype, buffer=shm.buf)
                np_array[:] = value[:]
                managed_values[attr] = (value.shape, value.dtype)
                shared_memory_objects[attr] = shm
            elif isinstance(value, bool):
                managed_values[attr] = manager.Value("b", int(value))
            elif np.issubdtype(type(value), np.dtype):
                managed_values[attr] = manager.Value(str, str(value))
            elif isinstance(value, dict):
                managed_values[attr] = manager.dict(value)
            else:
                typecode = typecode_map.get(type(value))
                if typecode is not None:
                    managed_values[attr] = manager.Value(typecode, value)
                else:
                    assert True
        return managed_values, shared_memory_objects

    @staticmethod
    def create_from_multiprocessing_types(
        managed_values, shared_memory_names, cv_splits
    ):
        new_instance = CVMatrix(cv_splits=cv_splits)
        shm_objects = []  # Don't want these garbage collected
        time_spent_attr = {}
        for attr, mp_type in reversed(managed_values.items()):
            start_attr_time = pytime.time()
            if attr == "val_folds_dict":
                continue
                val_folds_dict = {}
                for k, v in dict(mp_type).items():
                    if isinstance(v, tuple):
                        shm = SharedMemory(shared_memory_names[k])
                        shm_objects.append(shm)
                        array = np.ndarray(v[0], dtype=v[1], buffer=shm.buf[:])
                        val_folds_dict[k] = array
                    else:
                        assert False
                setattr(new_instance, attr, val_folds_dict)
            elif isinstance(mp_type, tuple):
                if attr in shared_memory_names:
                    shm = SharedMemory(shared_memory_names[attr])
                    shm_objects.append(shm)
                    array = np.ndarray(mp_type[0], dtype=mp_type[1], buffer=shm.buf[:])
                    setattr(new_instance, attr, array)
                else:
                    assert False
            elif hasattr(mp_type, "_typecode"):
                if mp_type._typecode == "b":
                    setattr(new_instance, attr, bool(mp_type.value))
                elif mp_type._typecode == "i":
                    setattr(new_instance, attr, int(mp_type.value))
                elif mp_type._typecode == "d":
                    setattr(new_instance, attr, float(mp_type.value))
                elif mp_type._typecode == "s":
                    setattr(new_instance, attr, str(mp_type.value))
            else:
                setattr(new_instance, attr, mp_type.value)

            end_attr_time = pytime.time()
            time_spent_attr[attr] = end_attr_time - start_attr_time

        # sort time_spent_attr[attr] by max
        time_spent_attr = dict(
            sorted(time_spent_attr.items(), key=lambda x: x[1], reverse=True)
        )
        worst_two_time_spent = list(time_spent_attr.keys())[:2]
        worst = {k: time_spent_attr[k] for k in worst_two_time_spent}
        # print(f"Time spent on worst two attributes: {worst}")

        return new_instance, shm_objects

    def fit(self, X: npt.ArrayLike, Y: Union[None, npt.ArrayLike] = None) -> None:
        """
        Loads and stores `X` and `Y` for cross-validation. Computes dataset-wide
        :math:`\mathbf{X}^{\mathbf{T}}\mathbf{X}` and, if `Y` is not `None`,
        :math:`\mathbf{X}^{\mathbf{T}}\mathbf{Y}`. If `center_X`, `center_Y`,
        `scale_X`, or `scale_Y` is `True`, the corresponding global statistics are also
        computed.

        Parameters
        ----------
        X : Array-like of shape (N, K) or (N,)
            Predictor variables for the entire dataset.

        Y : None or array-like of shape (N, M) or (N,), optional, default=None
            Response variables for the entire dataset. If `None`, subsequent calls to
            training_XTY and training_XTX_XTY will raise a `ValueError`.
        """
        self.X_total = self._init_mat(X)
        self.N, self.K = self.X_total.shape
        self.XTX_total = self.X_total.T @ self.X_total
        if Y is not None:
            self.Y_total = self._init_mat(Y)
            self.M = self.Y_total.shape[1]
            self.XTY_total = self.X_total.T @ self.Y_total
        self._init_total_stats()

    def training_XTX(self, val_fold: Hashable) -> np.ndarray:
        """
        Returns the training set :math:`\mathbf{X}^{\mathbf{T}}\mathbf{X}`
        corresponding to every sample except those belonging to the given validation
        fold.

        Parameters
        ----------
        val_fold : Hashable
            The validation fold for which to return the corresponding training set
            :math:`\mathbf{X}^{\mathbf{T}}\mathbf{X}`.

        Returns
        -------
        Array of shape (K, K)
            The training set :math:`\mathbf{X}^{\mathbf{T}}\mathbf{X}`.

        Raises
        ------
        ValueError
            If `val_fold` was not provided as a cross-validation split in the
            `cv_splits` parameter of the constructor.

        See Also
        --------
        training_XTY :
            Returns the training set :math:`\mathbf{X}^{\mathbf{T}}\mathbf{Y}`
        training_XTX_XTY :
            Returns the training set :math:`\mathbf{X}^{\mathbf{T}}\mathbf{X}` and
            :math:`\mathbf{X}^{\mathbf{T}}\mathbf{Y}` for a given validation fold. This
            method is faster than calling `training_XTX` and `training_XTY` separately.
        """
        return self._training_matrices(True, False, val_fold)

    def training_XTY(self, val_fold: Hashable) -> np.ndarray:
        """
        Returns the training set :math:`\mathbf{X}^{\mathbf{T}}\mathbf{Y}`
        corresponding to every sample except those belonging to the given validation
        fold.

        Parameters
        ----------
        val_fold : Hashable
            The validation fold for which to return the corresponding training set
            :math:`\mathbf{X}^{\mathbf{T}}\mathbf{Y}`.

        Returns
        -------
        Array of shape (K, M)
            The training set :math:`\mathbf{X}^{\mathbf{T}}\mathbf{Y}`.

        Raises
        ------
        ValueError
            If `Y` is `None`.

        ValueError
            If `val_fold` was not provided as a cross-validation split in the
            `cv_splits` parameter of the constructor.

        See Also
        --------
        training_XTX :
            Returns the training set :math:`\mathbf{X}^{\mathbf{T}}\mathbf{X}`
        training_XTX_XTY :
            Returns the training set :math:`\mathbf{X}^{\mathbf{T}}\mathbf{X}` and
            :math:`\mathbf{X}^{\mathbf{T}}\mathbf{Y}` for a given validation fold. This
            method is faster than calling `training_XTX` and `training_XTY` separately.
        """
        return self._training_matrices(False, True, val_fold)

    def training_XTX_XTY(self, val_fold: Hashable) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns the training set :math:`\mathbf{X}^{\mathbf{T}}\mathbf{X}` and
        :math:`\mathbf{X}^{\mathbf{T}}\mathbf{Y}` corresponding to every sample except
        those belonging to the given validation fold.

        Parameters
        ----------
        val_fold : Hashable
            The validation fold for which to return the corresponding training set
            :math:`\mathbf{X}^{\mathbf{T}}\mathbf{X}` and
            :math:`\mathbf{X}^{\mathbf{T}}\mathbf{Y}`.

        Returns
        -------
        tuple of arrays of shapes (K, K) and (K, M)
            The training set :math:`\mathbf{X}^{\mathbf{T}}\mathbf{X}` and
            :math:`\mathbf{X}^{\mathbf{T}}\mathbf{Y}`.

        Raises
        ------
        ValueError
            If `Y` is `None`.

        ValueError
            If `val_fold` was not provided as a cross-validation split in the
            `cv_splits` parameter of the constructor.

        See Also
        --------
        training_XTX :
            Returns the training set :math:`\mathbf{X}^{\mathbf{T}}\mathbf{X}`
        training_XTY :
            Returns the training set :math:`\mathbf{X}^{\mathbf{T}}\mathbf{Y}`
        """
        return self._training_matrices(True, True, val_fold)

    def _training_matrices(
        self, return_XTX: bool, return_XTY: bool, val_fold: Hashable
    ) -> Union[np.ndarray, tuple[np.ndarray, np.ndarray]]:
        """
        Returns the training set :math:`\mathbf{X}^{\mathbf{T}}\mathbf{X}` and/or
        :math:`\mathbf{X}^{\mathbf{T}}\mathbf{Y}` corresponding to every sample except
        those belonging to the given validation fold.

        Parameters
        ----------
        return_XTX : bool
            Whether to return the training set
            :math:`\mathbf{X}^{\mathbf{T}}\mathbf{X}`.

        val_fold : Hashable
            The validation fold for which to return the corresponding training set
            :math:`\mathbf{X}^{\mathbf{T}}\mathbf{X}` and
            :math:`\mathbf{X}^{\mathbf{T}}\mathbf{Y}`

        return_XTY : bool, optional, default=False
            Whether to return the training set
            :math:`\mathbf{X}^{\mathbf{T}}\mathbf{Y}`.

        Returns
        -------
        Array of shape (K, K) or (K, M) or tuple of arrays of shapes (K, K) and (K, M)
            The training set :math:`\mathbf{X}^{\mathbf{T}}\mathbf{X}` and/or
            training set :math:`\mathbf{X}^{\mathbf{T}}\mathbf{Y}`.

        Raises
        ------
        ValueError
            If both `return_XTX` and `return_XTY` are `False` or if `return_XTY` is
            `True` and `Y` is `None`.

        ValueError
            If `val_fold` was not provided as a cross-validation split in the
            `cv_splits` parameter of the constructor.
        """
        if not return_XTX and not return_XTY:
            raise ValueError(
                "At least one of `return_XTX` and `return_XTY` must be True."
            )
        if return_XTY and self.Y_total is None:
            raise ValueError("Response variables `Y` are not provided.")
        X_train_mean = None
        Y_train_mean = None
        X_train_std = None
        Y_train_std = None
        N_train = None
        try:
            val_indices = self.val_folds_dict[val_fold]
        except KeyError as e:
            raise ValueError(f"Validation fold {val_fold} not found.") from e
        X_val = self.X_total[val_indices]
        if return_XTY:
            Y_val = self.Y_total[val_indices]
        if self.center_X or self.center_Y or self.scale_X or self.scale_Y:
            N_val = val_indices.size
            N_train = self.N - N_val
            N_total_over_N_train = self.N / N_train
            N_val_over_N_train = N_val / N_train
        if self.center_X or self.center_Y or self.scale_X:
            X_train_mean = self._compute_training_mat_mean(
                X_val, self.X_total_mean, N_total_over_N_train, N_val_over_N_train
            )
        if return_XTY and (self.center_X or self.center_Y or self.scale_Y):
            Y_train_mean = self._compute_training_mat_mean(
                Y_val, self.Y_total_mean, N_total_over_N_train, N_val_over_N_train
            )
        if self.scale_X:
            X_train_std = self._compute_training_mat_std(
                X_val, X_train_mean, self.sum_X_total, self.sum_sq_X_total, N_train
            )
        if self.scale_Y and return_XTY:
            Y_train_std = self._compute_training_mat_std(
                Y_val, Y_train_mean, self.sum_Y_total, self.sum_sq_Y_total, N_train
            )
        if return_XTX and return_XTY:
            return (
                self._training_kernel_matrix(
                    self.XTX_total,
                    X_val,
                    X_val,
                    X_train_mean,
                    X_train_mean,
                    X_train_std,
                    X_train_std,
                    N_train,
                    center=self.center_X,
                ),
                self._training_kernel_matrix(
                    self.XTY_total,
                    X_val,
                    Y_val,
                    X_train_mean,
                    Y_train_mean,
                    X_train_std,
                    Y_train_std,
                    N_train,
                    center=self.center_X or self.center_Y,
                ),
            )
        if return_XTX:
            return self._training_kernel_matrix(
                self.XTX_total,
                X_val,
                X_val,
                X_train_mean,
                X_train_mean,
                X_train_std,
                X_train_std,
                N_train,
                center=self.center_X,
            )
        return self._training_kernel_matrix(
            self.XTY_total,
            X_val,
            Y_val,
            X_train_mean,
            Y_train_mean,
            X_train_std,
            Y_train_std,
            N_train,
            center=self.center_X or self.center_Y,
        )

    def _training_kernel_matrix(
        self,
        total_kernel_mat: np.ndarray,
        X_val: np.ndarray,
        mat2_val: np.ndarray,
        X_train_mean: Union[None, np.ndarray] = None,
        mat2_train_mean: Union[None, np.ndarray] = None,
        X_train_std: Union[None, np.ndarray] = None,
        mat2_train_std: Union[None, np.ndarray] = None,
        N_train: Union[None, int] = None,
        center: bool = False,
    ) -> np.ndarray:
        """
        Computes the training set kernel matrix for a given fold.

        Parameters
        ----------
        total_kernel_mat : Array of shape (N, K) or (N, M)
            The total kernel matrix :math:`\mathbf{X}^{\mathbf{T}}\mathbf{X}` or
            :math:`\mathbf{X}^{\mathbf{T}}\mathbf{Y}`.

        X_val : Array of shape (N_val, K)
            The validation set of predictor variables.

        mat2_val : Array of shape (N_val, K) or (N_val, M)
            The validation set of predictor or resoponse variables.

        X_train_mean : None or array of shape (1, K), optional, default=None
            The row of column-wise means of the training set of predictor variables.

        mat2_train_mean : None or array of shape (1, K) or (1, M), optional,
        default=None
            The row of column-wise means of the training set of predictor or response
            variables.

        X_train_std : None or array of shape (1, K), optional, default=None
            The row of column-wise standard deviations of the training set of predictor
            variables.

        mat2_train_std : None or array of shape (1, K) or (1, M), optional, default=None
            The row of column-wise standard deviations of the training set of predictor
            or response variables.

        N_train : None or int, optional, default=None
            The size of the training set. Only required if `X_train_mean` or
            `mat2_train_mean` is not `None`.

        center : bool, optional, default=False
            Whether to center the kernel matrix. If `True`, the kernel matrix is
            centered. Setting this parameter to `True` requires that `X_train_mean` and
            `mat2_train_mean` are not `None`.

        Returns
        -------
        Array of shape (K, K) or (K, M)
            The training set kernel matrix.
        """
        XTmat2_train = total_kernel_mat - X_val.T @ mat2_val
        if center:
            XTmat2_train -= N_train * (X_train_mean.T @ mat2_train_mean)
        if X_train_std is not None and mat2_train_std is not None:
            return XTmat2_train / (X_train_std.T @ mat2_train_std)
        if X_train_std is not None:
            return XTmat2_train / X_train_std.T
        if mat2_train_std is not None:
            return XTmat2_train / mat2_train_std
        return XTmat2_train

    def _compute_training_mat_mean(
        self,
        mat_val: np.ndarray,
        mat_total_mean: np.ndarray,
        N_total_over_N_train: float,
        N_val_over_N_train: float,
    ) -> np.ndarray:
        """
        Computes the row of column-wise means of a matrix for a given fold.

        Parameters
        ----------
        mat_val : Array of shape (N_val, K) or (N_val, M)
            The validation set of `X` or `Y`.

        mat_total_mean : Array of shape (1, K) or (1, M)
            The row of column-wise means of the total matrix.

        N_total_over_N_train : float
            The ratio of the total number of samples to the number of samples in the
            training set.

        N_val_over_N_train : float
            The ratio of the number of samples in the validation set to the number of
            samples in the training set.

        Returns
        -------
        Array of shape (1, K) or (1, M)
            The row of column-wise means of the training set matrix.
        """
        return (
            N_total_over_N_train * mat_total_mean
            - N_val_over_N_train * mat_val.mean(axis=0, keepdims=True)
        )

    def _compute_training_mat_std(
        self,
        mat_val: np.ndarray,
        mat_train_mean: np.ndarray,
        sum_mat_total: np.ndarray,
        sum_sq_mat_total: np.ndarray,
        N_train: int,
    ) -> np.ndarray:
        """
        Computes the row of column-wise standard deviations of a matrix for a given
        fold.

        Parameters
        ----------
        mat_val : Array of shape (N_val, K) or (N_val, M)
            The validation set of `X` or `Y`.

        mat_train_mean : Array of shape (1, K) or (1, M)
            The row of column-wise means of the training matrix.

        sum_mat_total : Array of shape (1, K) or (1, M)
            The row of column-wise sums of the total matrix.

        sum_sq_mat_total : Array of shape (1, K) or (1, M)
            The row of column-wise sums of squares of the total matrix.

        N_train : int
            The size of the training set.

        Returns
        -------
        Array of shape (1, K) or (1, M)
            The row of column-wise standard deviations of the training set matrix.
        """
        train_sum_mat = sum_mat_total - np.expand_dims(
            np.einsum("ij -> j", mat_val), axis=0
        )
        train_sum_sq_mat = sum_sq_mat_total - np.expand_dims(
            np.einsum("ij,ij -> j", mat_val, mat_val), axis=0
        )
        mat_train_std = np.sqrt(
            1
            / (N_train - 1)
            * (
                -2 * mat_train_mean * train_sum_mat
                + N_train * np.einsum("ij,ij -> ij", mat_train_mean, mat_train_mean)
                + train_sum_sq_mat
            )
        )
        mat_train_std[mat_train_std == 0] = 1
        return mat_train_std

    def _init_mat(self, mat: np.ndarray) -> np.ndarray:
        """
        Casts the matrix to the dtype specified in the constructor and reshapes it if
        the matrix is one-dimensional.

        Parameters
        ----------
        mat : Array of shape (N, K) or (N, M) or (N,)
            The matrix to be initialized.

        Returns
        -------
        Array of shape (N, K) or (N, M) or (N, 1)
            The initialized matrix.
        """
        mat = np.asarray(mat, dtype=self.dtype)
        if self.copy and mat.dtype == self.dtype:
            mat = mat.copy()
        if mat.ndim == 1:
            mat = mat.reshape(-1, 1)
        return mat

    def _init_total_stats(self) -> None:
        """
        Initializes the global statistics for `X` and `Y`.
        """
        if self.center_X or self.center_Y or self.scale_X:
            self.X_total_mean = self.X_total.mean(axis=0, keepdims=True)
        else:
            self.X_total_mean = None
        if (
            self.center_X or self.center_Y or self.scale_Y
        ) and self.Y_total is not None:
            self.Y_total_mean = self.Y_total.mean(axis=0, keepdims=True)
        else:
            self.Y_total_mean = None
        if self.scale_X:
            self.sum_X_total = np.expand_dims(
                np.einsum("ij -> j", self.X_total), axis=0
            )
            self.sum_sq_X_total = np.expand_dims(
                np.einsum("ij,ij -> j", self.X_total, self.X_total), axis=0
            )
        else:
            self.sum_X_total = None
            self.sum_sq_X_total = None
        if self.scale_Y and self.Y_total is not None:
            self.sum_Y_total = np.expand_dims(
                np.einsum("ij -> j", self.Y_total), axis=0
            )
            self.sum_sq_Y_total = np.expand_dims(
                np.einsum("ij,ij -> j", self.Y_total, self.Y_total), axis=0
            )
        else:
            self.sum_Y_total = None
            self.sum_sq_Y_total = None

    def _init_val_folds_dict(
        self, cv_splits: Iterable[Hashable]
    ) -> dict[Hashable, npt.NDArray[np.int_]]:
        """
        Generates a dictionary of validation indices for each fold. The dictionary is
        stored in the `val_folds_dict` attribute. The dictionary is used to quickly
        access the validation indices for each fold.

        Parameters
        ----------
        cv_splits : Iterable of Hashable with N elements
            An iterable defining cross-validation splits. Each unique value in
            `cv_splits` corresponds to a different fold.
        """
        val_folds_dict = {}
        for i, num in enumerate(cv_splits):
            try:
                val_folds_dict[num].append(i)
            except KeyError:
                val_folds_dict[num] = [i]
        for key in val_folds_dict:
            val_folds_dict[key] = np.asarray(val_folds_dict[key], dtype=int)
        self.val_folds_dict = val_folds_dict
