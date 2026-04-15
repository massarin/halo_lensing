"""
FFTLog for integrals with 1 Bessel function.
-- fftlog: spherical Bessel transforms
-- hankel: cylindrical Bessel (Hankel) transforms
-- optional c_window smoothing of Fourier coefficients

By Xiao Fang (Apr 2019); edited by Sunao Sugiyama (bin-averaged version).
"""

import logging

import numpy as np
from numpy.fft import irfft, rfft
from scipy.special import gamma


class fftlog:
    def __init__(
        self,
        x,
        fx,
        nu=1.1,
        N_extrap_low=0,
        N_extrap_high=0,
        c_window_width=0.25,
        N_pad=0,
        xy=1,
    ):
        self.x_origin = x
        self.dlnx = np.log(x[1] / x[0])
        self.fx_origin = fx
        self.nu = nu
        self.N_extrap_low = N_extrap_low
        self.N_extrap_high = N_extrap_high
        self.c_window_width = c_window_width
        self.xy = xy

        self.x = log_extrap(x, N_extrap_low, N_extrap_high)
        self.fx = log_extrap(fx, N_extrap_low, N_extrap_high)
        self.N = self.x.size

        self.N_pad = N_pad
        if N_pad:
            pad = np.zeros(N_pad)
            self.x = log_extrap(self.x, N_pad, N_pad)
            self.fx = np.hstack((pad, self.fx, pad))
            self.N += 2 * N_pad
            self.N_extrap_high += N_pad
            self.N_extrap_low += N_pad

        if self.N % 2 == 1:
            self.x = self.x[:-1]
            self.fx = self.fx[:-1]
            self.N -= 1
            if N_pad:
                self.N_extrap_high -= 1

        self.m, self.c_m = self._get_c_m()
        self.eta_m = 2 * np.pi / (float(self.N) * self.dlnx) * self.m

    def _get_c_m(self):
        f_b = self.fx * self.x ** (-self.nu)
        c_m = rfft(f_b)
        m = np.arange(0, self.N // 2 + 1)
        c_m = c_m * c_window(m, int(self.c_window_width * self.N // 2.0))
        return m, c_m

    def _y(self, ell):
        if isinstance(self.xy, (int, float)):
            return self.xy / self.x[::-1]
        return (ell + 1) / self.x[::-1]

    def _trim(self, y, Fy):
        return (
            y[self.N_extrap_high : self.N - self.N_extrap_low],
            Fy[self.N_extrap_high : self.N - self.N_extrap_low],
        )

    def fftlog(self, ell):
        """F(y) = int dx/x f(x) j_l(xy), j_l = spherical Bessel of order l."""
        y = self._y(ell)
        z_ar = self.nu + 1j * self.eta_m
        h_m = self.c_m * (self.x[0] * y[0]) ** (-1j * self.eta_m) * g_l(ell, z_ar)
        Fy = irfft(np.conj(h_m)) * y ** (-self.nu) * np.sqrt(np.pi) / 4.0
        return self._trim(y, Fy)

    def fftlog_binave(self, ell, bandwidth_dlny, D, alpha_pow):
        """Bin-averaged fftlog; ref: 2DFFTLog by Xiao Fang et al."""
        y = self._y(ell)
        z_ar = self.nu + 1j * self.eta_m
        gl = g_l_smooth(ell, z_ar, bandwidth_dlny, alpha_pow)
        s_d_lambda = (np.exp(D * bandwidth_dlny) - 1.0) / D
        h_m = self.c_m * (self.x[0] * y[0]) ** (-1j * self.eta_m) * gl / s_d_lambda
        Fy = irfft(np.conj(h_m)) * y ** (-self.nu) * np.sqrt(np.pi) / 4.0
        return self._trim(y, Fy)

    def fftlog_dj(self, ell):
        """F(y) = int dx/x f(x) j'_l(xy)."""
        y = self._y(ell)
        z_ar = self.nu + 1j * self.eta_m
        h_m = self.c_m * (self.x[0] * y[0]) ** (-1j * self.eta_m) * g_l_1(ell, z_ar)
        Fy = irfft(np.conj(h_m)) * y ** (-self.nu) * np.sqrt(np.pi) / 4.0
        return self._trim(y, Fy)

    def fftlog_ddj(self, ell):
        """F(y) = int dx/x f(x) j''_l(xy)."""
        y = self._y(ell)
        z_ar = self.nu + 1j * self.eta_m
        h_m = self.c_m * (self.x[0] * y[0]) ** (-1j * self.eta_m) * g_l_2(ell, z_ar)
        Fy = irfft(np.conj(h_m)) * y ** (-self.nu) * np.sqrt(np.pi) / 4.0
        return self._trim(y, Fy)


class hankel:
    """Hankel transform: F(y) = int dx f(x) J_n(xy)."""

    def __init__(
        self,
        x,
        fx,
        nu,
        N_extrap_low=0,
        N_extrap_high=0,
        c_window_width=0.25,
        N_pad=0,
        xy=1,
    ):
        self.myfftlog = fftlog(
            x,
            np.sqrt(x) * fx,
            nu,
            N_extrap_low,
            N_extrap_high,
            c_window_width,
            N_pad,
            xy=xy,
        )

    def hankel(self, n):
        y, Fy = self.myfftlog.fftlog(n - 0.5)
        return y, Fy * np.sqrt(2 * y / np.pi)

    def hankel_binave(self, n, bandwidth_dlny, D):
        y, Fy = self.myfftlog.fftlog_binave(n - 0.5, bandwidth_dlny, D, D + 0.5)
        return y, Fy * np.sqrt(2 * y / np.pi)


# ── utility functions ─────────────────────────────────────────────────────────


def log_extrap(x, N_extrap_low, N_extrap_high):
    low_x = high_x = []
    if N_extrap_low:
        dlnx_low = np.log(x[1] / x[0])
        low_x = x[0] * np.exp(dlnx_low * np.arange(-N_extrap_low, 0))
    if N_extrap_high:
        dlnx_high = np.log(x[-1] / x[-2])
        high_x = x[-1] * np.exp(dlnx_high * np.arange(1, N_extrap_high + 1))
    return np.hstack((low_x, x, high_x))


def c_window(n, n_cut):
    """One-sided window on c_m; Eq. C1 of McEwen et al. 2016 (arXiv:1603.04826)."""
    n_right = n[-1] - n_cut
    n_r = n[n > n_right]
    theta = (n[-1] - n_r) / float(n[-1] - n_right - 1)
    W = np.ones(n.size)
    W[n > n_right] = theta - np.sin(2 * np.pi * theta) / (2 * np.pi)
    return W


def g_m_vals(mu, q):
    """g(mu,q) = Gamma((mu+1+q)/2) / Gamma((mu+1-q)/2).
    Asymptotic Stirling form used when |Im(q)| + |mu| > 200.
    Adapted from FAST-PT."""
    if mu + 1 + q.real[0] == 0:
        logging.info("gamma(0) encountered. Please change nu value (try nu=1.1).")
        exit()
    imag_q = np.imag(q)
    g_m = np.zeros(q.size, dtype=complex)
    cut = 200
    mask_asym = np.abs(imag_q) + np.abs(mu) > cut
    mask_good = (~mask_asym) & (q != mu + 1 + 0j)

    ap = (mu + 1 + q[mask_good]) / 2.0
    am = (mu + 1 - q[mask_good]) / 2.0
    g_m[mask_good] = gamma(ap) / gamma(am)

    ap = (mu + 1 + q[mask_asym]) / 2.0
    am = (mu + 1 - q[mask_asym]) / 2.0
    g_m[mask_asym] = np.exp(
        (ap - 0.5) * np.log(ap)
        - (am - 0.5) * np.log(am)
        - q[mask_asym]
        + (1.0 / 12) * (1.0 / ap - 1.0 / am)
        + (1.0 / 360) * (1.0 / am**3 - 1.0 / ap**3)
        + (1.0 / 1260) * (1.0 / ap**5 - 1.0 / am**5)
    )
    g_m[q == mu + 1 + 0j] = 0.0
    return g_m


def g_l(l, z_array):
    """2^z * Gamma((l+z)/2) / Gamma((3+l-z)/2)."""
    return 2.0**z_array * g_m_vals(l + 0.5, z_array - 1.5)


def g_l_1(l, z_array):
    """Kernel for first-derivative spherical Bessel integrals."""
    return -(2.0 ** (z_array - 1)) * (z_array - 1) * g_m_vals(l + 0.5, z_array - 2.5)


def g_l_2(l, z_array):
    """Kernel for second-derivative spherical Bessel integrals."""
    return (
        2.0 ** (z_array - 2)
        * (z_array - 1)
        * (z_array - 2)
        * g_m_vals(l + 0.5, z_array - 3.5)
    )


def g_l_smooth(l, z_array, binwidth_dlny, alpha_pow):
    """Smoothed kernel for bin-averaged transforms."""
    gl = 2.0**z_array * g_m_vals(l + 0.5, z_array - 1.5)
    gl *= (np.exp((alpha_pow - z_array) * binwidth_dlny) - 1.0) / (alpha_pow - z_array)
    return gl


# ── top-level convenience transforms ─────────────────────────────────────────


def pk2wp(
    k,
    pk,
    nu=1.01,
    N_extrap_low=0,
    N_extrap_high=0,
    c_window_width=0.25,
    N_pad=0,
    kr=1,
    dlnrp=0.0,
    D=2,
):
    h = hankel(
        k, pk * k**2, nu, N_extrap_low, N_extrap_high, c_window_width, N_pad, xy=kr
    )
    if dlnrp == 0.0:
        rp, wp = h.hankel(0)
    else:
        rp, wp = h.hankel_binave(0, dlnrp, D)
    return rp, wp / (2 * np.pi)


def pk2dwp(
    k,
    pk,
    nu=1.01,
    N_extrap_low=0,
    N_extrap_high=0,
    c_window_width=0.25,
    N_pad=0,
    kr=1,
    dlnrp=0.0,
    D=2,
):
    h = hankel(
        k, pk * k**2, nu, N_extrap_low, N_extrap_high, c_window_width, N_pad, xy=kr
    )
    if dlnrp == 0.0:
        rp, dwp = h.hankel(2)
    else:
        rp, dwp = h.hankel_binave(2, dlnrp, D)
    return rp, dwp / (2 * np.pi)


def pk2xi(
    k,
    pk,
    nu=1.01,
    N_extrap_low=0,
    N_extrap_high=0,
    c_window_width=0.25,
    N_pad=0,
    kr=1,
    l=0,
):
    f = fftlog(
        k, pk * k**3, nu, N_extrap_low, N_extrap_high, c_window_width, N_pad, xy=kr
    )
    r, xi = f.fftlog(l)
    return r, xi / (2 * np.pi**2)


def xi2pk(
    r, xi, nu=1.01, N_extrap_low=0, N_extrap_high=0, c_window_width=0.25, N_pad=0, kr=1
):
    f = fftlog(
        r, xi * r**3, nu, N_extrap_low, N_extrap_high, c_window_width, N_pad, xy=kr
    )
    k, pk = f.fftlog(0)
    return k, pk * 4 * np.pi
