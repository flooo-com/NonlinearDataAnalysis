"""
Module for recurrence-based nonlinear time series analysis
==========================================================

"""

# Created: Fri Aug 31, 2018
# Last modified: Wed Dec 18, 2024
# Source: Bedartha Goswami
# Modified: Norbert Marwan

import sys
import numpy as np

from scipy.spatial.distance import pdist, squareform
from itertools import chain

# disable dive by zero warnings
np.seterr(divide="ignore")


def mi(x, maxlag, binrule="fd"):
    """
    Computes the self mutual information of a time series up to a specified maximum lag.

    Args:
        x (array-like): 
            The input time series.
        maxlag (int): 
            The maximum lag up to which mutual information is computed.
        binrule (str, optional): 
            The binning rule for histogram estimation. Default is "fd" (Freedman-Diaconis).

    Returns:
        tuple:
            - mi (numpy.ndarray): Array of mutual information values for each lag.
            - lags (numpy.ndarray): Array of lag values corresponding to the computed mutual information.

    Example:
        >>> time_series = np.sin(np.linspace(0, 10 * np.pi, 1000)) + np.random.normal(0, 0.1, 1000)
        >>> mutual_info, lags = mi(time_series, maxlag=200, binrule="fd")
        >>> plt.plot(lags, mutual_info)
    """

    # initialize variables
    n = len(x)
    lags = np.arange(0, maxlag, dtype="int")
    mi = np.zeros(len(lags))

    # loop over lags and get MI
    for i, lag in enumerate(lags):

        # extract lagged data
        y1 = x[:n - lag].copy()
        y2 = x[lag:].copy()

        # use np.histogram to get individual entropies
        H1, be1 = entropy1d(y1, binrule)
        H2, be2 = entropy1d(y2, binrule)
        H12, _, _ = entropy2d(y1, y2, [be1, be2])
        # use the entropies to estimate MI
        mi[i] = H1 + H2 - H12

    return mi, lags


def entropy1d(x, binrule):
    """
    Computes the Shannon entropy of a one-dimensional dataset using a specified binning rule.

    Args:
        x (array-like): 
            The input data for which the entropy is to be calculated.
        binrule (str): 
            The binning rule to be used for histogram estimation (e.g., "fd" for Freedman-Diaconis).

    Returns:
        tuple:
            - H (float): The computed Shannon entropy.
            - be (numpy.ndarray): The bin edges used for the histogram.
    """

    p, be = np.histogram(x, bins=binrule, density=True)
    r = be[1:] - be[:-1]
    P = p * r
    L = np.log2(P)
    i = ~ np.isinf(L)
    H = -(P[i] * L[i]).sum()

    return H, be


def entropy2d(x, y, bin_edges):
    """
    Computes the Shannon entropy of a two-dimensional dataset using specified bin edges.

    Args:
        x (array-like): 
            The first dataset (e.g., one dimension of a bivariate distribution).
        y (array-like): 
            The second dataset (e.g., the other dimension of a bivariate distribution).
        bin_edges (list or tuple): 
            A pair of arrays specifying the bin edges for `x` and `y`.

    Returns:
        tuple:
            - H (float): The computed Shannon entropy of the joint distribution.
            - bex (numpy.ndarray): The bin edges used for the first dimension (`x`).
            - bey (numpy.ndarray): The bin edges used for the second dimension (`y`).
    """

    p, bex, bey = np.histogram2d(x, y, bins=bin_edges, density=True)
    r = np.outer(bex[1:] - bex[:-1], bey[1:] - bey[:-1])
    P = p * r
    H = np.zeros(P.shape)
    i = ~np.isinf(np.log2(P))
    H[i] = -(P[i] * np.log2(P[i]))
    H = H.sum()

    return H, bex, bey


def first_minimum(y):
    """
    Finds the first local minimum in a given data series.

    Args:
        y (array-like): 
            The input data series.

    Returns:
        int or float: 
            - The index of the first local minimum if found.
            - NaN if no local minimum exists.
    """

    try:
        fmin = np.where(np.diff(np.sign(np.diff(y))) == 2.)[0][0] + 2
    except IndexError:
        fmin = np.nan
    return fmin


def acf(x, maxlag):
    """
    Computes the autocorrelation function (ACF) of a time series up to a specified maximum lag.

    Args:
        x (array-like): 
            The input time series.
        maxlag (int): 
            The maximum lag up to which the ACF is computed.

    Returns:
        tuple:
            - acf (numpy.ndarray): The autocorrelation values for each lag.
            - lags (numpy.ndarray): The corresponding lag values.

    Example:
        >>> time_series = np.sin(np.linspace(0, 10 * np.pi, 1000)) + np.random.normal(0, 0.1, 1000)
        >>> c, lags = acf(time_series, maxlag=400)
        >>> plt.plot(lags, c)
    """

    # normalize data
    n = len(x)
    a = (x - x.mean()) / (x.std() * n)
    b = (x - x.mean()) / x.std()

    # get acf
    cor = np.correlate(a, b, mode="full")
    acf = cor[n:n+maxlag+1]
    lags = np.arange(maxlag + 1)

    return acf, lags


def fnn(x: np.ndarray, tau: int, maxdim: int, r: float = 10.0) -> (np.ndarray, np.ndarray):
    """
    Computes the fraction of false nearest neighbors (FNN) for a time series 
    up to a specified maximum embedding dimension.

    Args:
        x (array-like): 
            The input time series.
        tau (int): 
            The time delay for phase space reconstruction.
        maxdim (int): 
            The maximum embedding dimension to evaluate.
        r (float, optional): 
            Threshold for distance increase. Defaults to 0.10.

    Returns:
        tuple:
            - fnn (numpy.ndarray): The fraction of false nearest neighbors for each embedding dimension.
            - dims (numpy.ndarray): The corresponding embedding dimensions considered.

    Example:
        >>> time_series = np.sin(np.linspace(0, 10 * np.pi, 1000)) + np.random.normal(0, 0.1, 1000)
        >>> f, m = fnn(time_series, tau=50, maxdim=5)
        >>> plt.plot(m, f)
    """

    # initialize params
    sd = x.std()
    r = r * (x.max() - x.min())
    e = sd / r
    fnn = np.zeros(maxdim)
    dims = np.arange(1, maxdim + 1, dtype="int")

    # ensure that (m-1) tau is not greater than N = length(x)
    N = len(x)
    K = (maxdim + 1 - 1) * tau
    if K >= N:
        m_c = N / tau
        i = np.where(dims >= m_c)
        fnn[i] = np.nan
        j = np.where(dims < m_c)
        dims = dims[j]

    # get first values of distances for m = 1
    d_m, k_m = mindist(x, 1, tau)

    # loop over dimensions and get FNN values
    for m in dims:
        # get minimum distances for one dimension higher
        d_m1, k_m1 = mindist(x, m + 1, tau)
        # remove those indices in the m-dimensional calculations which cannot
        # occur in the m+1-dimensional arrays as the m+1-dimensional arrays are
        # smaller
        cond1 = k_m[1] > k_m1[0][-1]
        cond2 = k_m[0] > k_m1[0][-1]
        j = np.where(~(cond1 + cond2))[0]
        k_m_ = (k_m[0][j], k_m[1][j])
        d_k_m, d_k_m1 = d_m[k_m_], d_m1[k_m_]
        n_m1 = d_k_m.shape[0]
        # calculate quantities in Eq. 3.8 of Kantz, Schreiber (2004) 2nd Ed.
        j = d_k_m > 0.
        y = np.zeros(n_m1, dtype="float")
        y[j] = (d_k_m1[j] / d_k_m[j] > r)
        w = (e > d_k_m)
        num = float((y * w).sum())
        den = float(w.sum())
        # assign FNN value depending on whether denominator is zero
        if den != 0.:
            fnn[m - 1] = num / den
        else:
            #fnn[m - 1] = np.nan
            fnn[m - 1] = 0
        # assign higher dimensional values to current one before next iteration
        d_m, k_m = d_m1, k_m1
    fnn[0]=1
    return fnn, dims


def mindist(x, m, tau):
    """
    Computes the minimum distances for each point in a given embedding.

    Args:
        x (array-like): 
            The input time series data or points to be embedded.
        m (int): 
            The embedding dimension.
        tau (int): 
            The time delay for the embedding.

    Returns:
        tuple: A tuple containing two elements:
            - d (ndarray): A 2D array of pairwise distances between the embedded points.
            - k (tuple): A tuple of two arrays:
                - The first array contains the indices of the points in the distance matrix.
                - The second array contains the indices of the closest point for each point in the embedding.
    """

    z = embed(x, m, tau)
    # d = squareform(pdist(z))
    n = len(z)
    d = np.zeros((n, n))
    for i in range(n):
        d[i] = np.max(np.abs(z[i] - z), axis=1)

    np.fill_diagonal(d, 99999999.)
    k = (np.arange(len(d)), np.argmin(d, axis=1))

    return d, k


def embed(x, m, tau):
    """
    Embeds a scalar time series into a multidimensional space using time delay embedding.

    Args:
        x (array-like): 
            The input scalar time series data.
        m (int): 
            The embedding dimension (number of dimensions to embed the time series into).
        tau (int): 
            The time delay, i.e., the step size between points in the embedded space.

    Returns:
        ndarray: 
            A 2D array where each row represents a point in the m-dimensional embedded space.

    Example:
        >>> time_series = np.sin(np.linspace(0, 10 * np.pi, 1000))
        >>> y = embed(time_series, m=2, tau=50)
        >>> plt.plot(y[:,0], y[:,1])
    """

    n = len(x)
    k = n - (m - 1) * tau
    z = np.zeros((k, m), dtype="float")
    for i in range(k):
        z[i] = [x[i + j * tau] for j in range(m)]

    return z


def first_zero(y):
    """
    Returns the index of the first occurrence of zero in the array.

    Args:
        y (array-like): 
            The input array in which to search for the first zero.

    Returns:
        int: 
            The index of the first zero in the array. If no zero is found, returns 0.
    """

    try:
        fzero = np.where(y == 0.)[0][0]
    except IndexError:
        fzero = 0
    return fzero


def rp(x, m, tau, e, norm="euclidean", threshold_by="distance", normed=True):
    """
    Computes the recurrence plot of a given time series using various thresholding methods.

    Args:
        x (array-like): 
            The input time series data.
        m (int): 
            The embedding dimension.
        tau (int): 
            The time delay for the embedding.
        e (float or int): 
            The threshold value used for determining recurrence, which can vary based on `threshold_by` method.
        norm (str, optional): 
            The distance metric to use for the recurrence plot (default is "euclidean").
        threshold_by (str, optional): 
            The method for thresholding the distance matrix. 
            The available options are:
                - "distance": Applies a threshold directly to the distance matrix.
                - "fan": Uses a fixed number of nearest neighbors, specified as a fraction of the total number of points.
                - "frr": Automatically selects the threshold to achieve a preselected recurrence rate.
            The default option is "distance".
        normed (bool, optional): 
            If True, normalizes the time series before embedding (default is True).

    Returns:
        ndarray: 
            A 2D binary matrix representing the recurrence plot, where 1 indicates recurrence and 0 indicates no recurrence.

    Example:
        >>> time_series = np.sin(np.linspace(0, 10 * np.pi, 1000)) + np.random.normal(0, 0.1, 1000)
        >>> R = rp(time_series, m=2, tau=50, e=.05, threshold_by="frr")
        >>> plt.imshow(R)
    """

    if normed:
        x = normalize(x)
    z = embed(x, m, tau)
    D = squareform(pdist(z, metric=norm))
    R = np.zeros(D.shape, dtype="int")
    if threshold_by == "distance":
        i = np.where(D <= e)
        R[i] = 1
    elif threshold_by == "fan":
        nk = np.ceil(e * R.shape[0]).astype("int")
        i = (np.arange(R.shape[0]), np.argsort(D, axis=0)[:nk])
        R[i] = 1
    elif threshold_by == "frr":
        e = np.percentile(D, e * 100.)
        i = np.where(D <= e)
        R[i] = 1

    return R


def rn(x, m, tau, e, norm="euclidean", threshold_by="distance", normed=True):
    """
    Computes the recurrence network adjacency matrix of a given time series.

    Args:
        x (array-like): 
            The input time series data.
        m (int): 
            The embedding dimension.
        tau (int): 
            The time delay for the embedding.
        e (float or int): 
            The threshold value used for determining recurrence, which can vary based on `threshold_by` method.
        norm (str, optional): 
            The distance metric to use for the recurrence plot (default is "euclidean").
        threshold_by (str, optional): 
            The method for thresholding the distance matrix. 
            The available options are:
                - "distance": Applies a threshold directly to the distance matrix.
                - "fan": Uses a fixed number of nearest neighbors, specified as a fraction of the total number of points.
                - "frr": Automatically selects the threshold to achieve a preselected recurrence rate.
            The default option is "distance".
        normed (bool, optional): 
            If True, normalizes the time series before embedding (default is True).

    Returns:
        ndarray: 
            A 2D binary matrix representing the recurrence network adjacency matrix, 
                where 1 indicates an edge (recurrence) and 0 indicates no edge.
    """

    z = embed(x, m, tau)
    D = squareform(pdist(z, metric=norm))
    np.fill_diagonal(D, np.inf)
    A = np.zeros(D.shape, dtype="int")
    if threshold_by == "distance":
        i = np.where(D <= e)
        A[i] = 1
    elif threshold_by == "fan":
        nk = np.ceil(e * A.shape[0]).astype("int")
        i = (np.arange(A.shape[0]), np.argsort(D, axis=0)[:nk])
        A[i] = 1
    elif threshold_by == "frr":
        e = np.percentile(D, e * 100.)
        i = np.where(D <= e)
        A[i] = 1

    return A


def normalize(x):
    """
    Returns the Z-score normalization of the input data.

    Parameters:
        x (array-like): 
            The input data to be normalized.

    Returns:
        ndarray: 
            The Z-score normalized version of the input data, where the mean is 0 and the standard deviation is 1.

    Example:
        >>> data = np.array([10, 15, 20, 25, 30])
        >>> normalized_data = normalize(data)
        >>> print(f'Mean: {mean(normalized_data)}, stddev: {std(normalized_data)}')
        Mean: 0.0, stddev: 0.9999999999999999
    """

    return (x - x.mean()) / x.std()

def surrogates(x, ns, method, params=None, verbose=False):
    """
    Generates random surrogates for a given time series using different methods.

    Args:
        x (array-like): 
            The input time series data.
        ns (int): 
            The number of surrogates to generate.
        method (str): 
            The method used for generating surrogates. Options are "iaaft", "twins", and "shuffle".
        params (dict, optional): 
            Additional parameters for surrogate generation, required for some methods:
                - For "iaaft": No additional parameters required.
                - For "twins": Requires the following keys:
                - "m": Embedding dimension.
                - "tau": Time delay.
                - "eps": Recurrence threshold.
                - "norm": Distance metric for recurrence plot.
                - "thr_by": Method for thresholding the distance matrix.
                - "tol": Tolerance for twin identification.
        verbose (bool, optional): 
            If True, enables verbose output for progress tracking.

    Returns:
        ndarray: 
            A 2D array where each row is a surrogate time series generated by the specified method.
    """

    nx = len(x)
    xs = np.zeros((ns, nx))
    if method == "iaaft":               # iAAFT
        # as per the steps given in Lancaster et al., Phys. Rep (2018)
        fft, ifft = np.fft.fft, np.fft.ifft
        TOL = 1E-6
        MSE_0 = 100
        MSE_K = 1000
        MAX_ITER = 10000
        ii = np.arange(nx)
        x_amp = np.abs(fft(x))
        x_srt = np.sort(x)

        for k in range(ns):
            # 1) Generate random shuffle of the data
            count = 0
            ri = np.random.permutation(ii)
            r_prev = x[ri]
            MSE_prev = MSE_0
            # while not np.all(rank_prev == rank_curr) and (count < MAX_ITER):
            while (np.abs(MSE_K - MSE_prev) > TOL) * (count < MAX_ITER):
                MSE_prev = MSE_K
                # 2) FFT current iteration yk, and then invert it but while
                # replacing the amplitudes with the original amplitudes but
                # keeping the angles from the FFT-ed version of the random
                phi_r_prev = np.angle(fft(r_prev))
                r = ifft(x_amp * np.exp(phi_r_prev * 1j), nx)
                # 3) rescale zk to the original distribution of x
                # rank_prev = rank_curr
                ind = np.argsort(r)
                r[ind] = x_srt
                MSE_K = (np.abs(x_amp - np.abs(fft(r)))).mean()
                r_prev = r
                # repeat until rank(z_k+1) = rank(z_k)
                count += 1
            if count >= MAX_ITER:
                print("maximum number of iterations reached!")
            xs[k] = np.real(r)
    elif method == "twins":              # twin surrogates
        # 1. Estimate RP according to given parameters
        R = rp(x, m=params["m"], tau=params["tau"], e=params["eps"],
               norm=params["norm"], threshold_by=params["thr_by"])

        # 2. Get embedded vectors
        xe = embed(x, params["m"], params["tau"])
        ne = len(xe)
        assert ne == len(R), "Something is wrong!"

        # 2. Identify twins
        _printmsg("identify twins ...", verbose)
        is_twin = []
        twins = []
        TOL = np.floor((params["tol"] * float(nx)) / 100.).astype("int")
        R_ = R.T
        for i in range(ne):
            diff = R_ ==  R_[i]
            j = np.sum(diff, axis=1) >= (ne - TOL)
            j = np.where(j)[0].tolist()
            j.remove(i)
            if len(j) > 0:
                is_twin.append(i)
                twins.append(j)

        # 3. Generate surrogates
        all_idx = range(ne)
        start_idx = np.random.choice(np.arange(ne), size=ns)
        xs[:, 0] = xe[start_idx, 0]

        for i in range(ns):
            j = 1
            k = start_idx[i]
            while j < nx:
                if k not in is_twin:
                    k += 1
                else:
                    twins_k = twins[is_twin.index(k)]
                    others = list(set(all_idx).difference(set(twins_k)))
                    l = np.random.choice(others)
                    k = np.random.choice(np.r_[l, twins_k])
                if k >= ne:
                    k = np.random.choice(np.arange(ne), size=1)
                xs[i, j] = xe[k, 0]
                j += 1

    elif method == "shuffle":               # simple random shuffling
        k = np.arange(nx)
        for i in range(ns):
            j = np.random.permutation(k)
            xs[i] = x[j]


    return xs

def det(R, lmin=None, hist=None, verb=False):
    r"""
    Calculates the determinism (DET) measure for a given recurrence matrix.

    Args:
        R (ndarray): 
            The recurrence matrix.
        lmin (int, optional): 
            The minimum line length for the calculation. If not specified, defaults to 10% of the matrix size.
        hist (tuple, optional): 
            Precomputed histogram of diagonal lines. If not provided, it will be computed.
            Should be in the form (nlines, bins, ll), where:
                - nlines: The number of diagonal lines for each length.
                - bins: The bin edges for line lengths.
                - ll: The actual line lengths.
        verb (bool, optional): 
            If True, enables verbose output for progress tracking.

    Returns:
        float: 
            The determinism (DET) measure, which quantifies the predictability of the system based on the recurrence plot.
            
        If the calculation cannot be performed, returns NaN.

    Example:
        >>> R = np.array([
        ...     [1, 1, 1, 0, 0], 
        ...     [1, 1, 1, 0, 1], 
        ...     [1, 1, 1, 1, 0],
        ...     [0, 1, 1, 1, 0],
        ...     [0, 1, 0, 0, 1]
        ... ])
        >>> det(R, lmin=2)
        0.75
    """

    if not lmin:
        lmin = int(0.1 * len(R))
    if not hist:
        if verb: print("estimating line length histogram...")
        nlines, bins, ll = diagonal_lines_hist(R, verb=verb)
    else:
        nlines, bins, ll = hist[0], hist[1], hist[2]
    if verb: print("estimating DET...")
    Pl = nlines.astype('float')
    l = (0.5 * (bins[:-1] + bins[1:])).astype('int')
    idx = l >= lmin
    num = l[idx] * Pl[idx]
    den = l * Pl
    if den.sum():
        DET = num.sum() / den.sum()
    else:
        DET = np.nan
    return DET

def diagonal_lines_hist(R, verb=False):
    """
    Computes the histogram of diagonal line lengths in a recurrence matrix.

    Args:
        R (ndarray): 
            The recurrence matrix.
        verb (bool, optional): 
            If True, enables verbose output for progress tracking.

    Returns:
        tuple: A tuple containing:
            - num_lines (ndarray): The count of diagonal lines for each line length.
            - bins (ndarray): The bin edges for line lengths.
            - line_lengths (list): A list of individual diagonal line lengths.
    """

    num_lines = 0
    bins = []
    line_lengths = []
    
    if verb:
        print("diagonal lines histogram...")
    line_lengths = []
    for i in range(1, len(R)):
        d = np.diag(R, k=i)
        ll = _count_num_lines(d)
        line_lengths.append(ll)

    if len(line_lengths):
        line_lengths = np.array(list(chain.from_iterable(line_lengths)))
        bins = np.arange(0.5, len(R)+1, 1.)
        num_lines, _ = np.histogram(line_lengths, bins=bins)
    return num_lines, bins, line_lengths


def _count_num_lines(arr):
    """
    Helpfer function that counts the lengths of consecutive ones in a binary array, representing diagonal lines in a recurrence plot.

    Args:
        arr (ndarray): 
            A binary array where 1 indicates part of a diagonal line and 0 indicates no line.

    Returns:
        list: 
            A list containing the lengths of consecutive ones (diagonal lines) in the array.
    """

    line_lens = []
    counting = False
    l = 0
    for i in range(len(arr)):
        if counting:
            if arr[i] == 0:
                l += 1
                line_lens.append(l)
                l = 0
                counting = False
            elif arr[i] == 1:
                l += 1
                if i == len(arr) - 1:
                    l += 1
                    line_lens.append(l)
        elif not counting:
            if arr[i] == 1:
                counting = True
    return line_lens

def entr(R, lmin=None, hist=None, verb=False):
    r"""
    Computes the entropy (ENTR) of diagonal line lengths in a recurrence matrix.

    The entropy is calculated based on the distribution of diagonal line lengths 
    in the recurrence matrix `R`, which provides a measure of complexity in the system.

    Args:
        R (numpy.ndarray): 
            The recurrence matrix, typically a binary matrix where `1` indicates recurrence.
        lmin (int, optional): 
            Minimum diagonal line length to be considered in the entropy calculation. 
            Defaults to 10% of the size of `R` if not specified.
        hist (tuple, optional): 
            Precomputed line length histogram as a tuple of three elements:
            (number of lines, bin edges, line lengths). If not provided, it will be estimated.
        verb (bool, optional): 
            If True, provides verbose output during computation. Default is True.

    Returns:
        float: 
            The computed entropy (ENTR) of the diagonal line length distribution.

    Notes:
        - Entropy is calculated as:
          :math:`- \sum p_l \log(p_l)`,
          where :math:`p_l` is the probability distribution of diagonal line lengths.

    Example:
        >>> R = np.array([
        ...     [1, 1, 0, 1, 0], 
        ...     [1, 1, 1, 0, 1], 
        ...     [0, 1, 1, 1, 0],
        ...     [1, 1, 1, 1, 0],
        ...     [0, 1, 0, 0, 1]
        ... ])
        >>> entr(R, lmin=2)
        0.6931
    """

    if not lmin:
        lmin = int(0.1 * len(R))
    if not hist:
        if verb: print("estimating line length histogram...")
        nlines, bins, ll = diagonal_lines_hist(R, verb=verb)
    else:
        nlines, bins, ll = hist[0], hist[1], hist[2]
    if verb: print("estimating ENTR...")

    pl = nlines.astype('float') / float(len(ll))
    l = (0.5 * (bins[:-1] + bins[1:])).astype('int')
    idx1 = l >= lmin
    pl = pl[idx1]
    idx = pl > 0.
    ENTR = (-pl[idx] * np.log(pl[idx])).sum()
    return ENTR


def tau_recurrence(R):
    r"""
    Computes the recurrence rate as a function of lag (tau) from the recurrence matrix.

    The recurrence rate for each lag is calculated as the mean of the elements along 
    the diagonals of the recurrence matrix `R` for each lag value (tau). This provides 
    insight into how recurrence behavior varies with increasing time lags.

    Args:
        R (numpy.ndarray): 
            The recurrence matrix, typically a binary square matrix where `1` indicates 
            recurrence and `0` indicates no recurrence.

    Returns:
        numpy.ndarray: 
            An array of recurrence rates for each lag value (tau), with length equal 
            to the size of the matrix `R`.

    Notes:
        - The recurrence rate :math:`q(\tau)` is computed for each lag as:
          :math:`q(\tau) = \frac{\text{sum of elements along the k-th diagonal}}{\text{number of elements in the diagonal}}`
          
        - The lag :math:`\tau` corresponds to the offset of the diagonal from the main diagonal.

    Example:
        >>> R = np.array([
        ...     [1, 0, 1, 0],
        ...     [0, 1, 0, 0],
        ...     [1, 0, 1, 0],
        ...     [0, 0, 0, 1]
        ... ])
        >>> tau_recurrence(R)
        array([1., 0., 0.5, 1])
    """
    
    N=R.shape[0] # R is the Recurrence matrix
    q=np.zeros(N)
    for tau in range(N):
        q[tau]=np.diag(R,k=tau).mean()
    return q


def cpr(Rx, Ry):
    r"""
    Computes the Correlation of Probabilities of Recurrence (CPR) between two recurrence matrices.

    CPR quantifies the correlation between the probabilities of recurrence 
    derived from two recurrence matrices as a function of lag (tau). It provides 
    a measure of dynamical similarity between the two systems represented by 
    their respective recurrence plots.

    Args:
        Rx (numpy.ndarray): 
            The recurrence matrix for the first time series. Must be a square matrix.
        Ry (numpy.ndarray): 
            The recurrence matrix for the second time series. Must have the same 
            shape as `Rx`.

    Returns:
        float: 
            The correlation of probabilities of recurrence (CPR) value. If the 
            recurrence probabilities do not decorrelate within the matrix size, 
            returns `np.nan`.

    Raises:
        AssertionError: 
            If `Rx` and `Ry` have different shapes.

    Notes:
        - The CPR calculation involves the following steps:
            1. Compute recurrence probabilities :math:`q_x(\tau)` and :math:`q_y(\tau)` for each lag (tau).
            2. Determine the decorrelation time, i.e., the lag after which :math:`q_x` and :math:`q_y` fall below :math:`1/e`.
            3. Normalize the probabilities beyond the decorrelation time to have zero mean and unit standard deviation.
            4. Compute CPR as the mean dot product of the normalized series.

        - If the recurrence probabilities do not decorrelate (i.e., never fall below :math:`1/e`), 
          the CPR value is set to `np.nan`.
    
    Example:
        >>> Rx = np.array([
        ...     [1, 0, 1, 1],
        ...     [0, 1, 1, 0],
        ...     [1, 1, 1, 0],
        ...     [1, 0, 0, 1]
        ... ])
        >>> Ry = np.array([
        ...     [1, 1, 0, 1],
        ...     [1, 1, 0, 0],
        ...     [0, 0, 1, 0],
        ...     [1, 0, 0, 1]
        ... ])
        >>> cpr(Rx, Ry)
        0.8386
    """

    assert Rx.shape == Ry.shape, "RPs are of different sizes!"
    N = Rx.shape[0]

    qx = tau_recurrence(Rx)
    qy = tau_recurrence(Ry)

    # obtain indices after taking into account decorrelation time
    e = np.exp(1.)
    try:
        ix = np.where(qx < 1. / e)[0][0]
        iy = np.where(qy < 1. / e)[0][0]
        i = max(ix, iy)
    except IndexError:
        i = N

    # final estimate
    if i < N:
        # normalised data series to mean zero and standard deviation one after
        # removing entries before decorrelation time
        qx_ = qx[i:]
        qx_ = (qx_ - np.nanmean(qx_)) / np.nanstd(qx_)
        qy_ = qy[i:]
        qy_ = (qy_ - np.nanmean(qy_)) / np.nanstd(qy_)

        # estimate CPR as the dot product of normalised series
        C = (qx_ * qy_).mean()
    else:
        C = np.nan

    return C