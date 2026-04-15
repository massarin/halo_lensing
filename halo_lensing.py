#!/usr/bin/env python
import sys

import numpy as np
import scipy.special as sc
from astropy import constants as const
from colossus.cosmology import cosmology
from colossus.halo import concentration, profile_nfw
from colossus.lss import bias
from scipy.interpolate import CubicSpline
from scipy.special import j0

import fftlog

################################################################
# utilities
################################################################


def sigma_crit(z, zs, cosmo, comoving=False):
    """Critical surface mass density [h M_sun Mpc^-2]."""
    if (zs <= z) or (zs <= 0.0) or (z <= 0.0):
        sys.exit("ERROR: wrong redshift")

    rl = cosmo.comovingDistance(0.0, z, transverse=True)
    rs = cosmo.comovingDistance(0.0, zs, transverse=True)
    rls = cosmo.comovingDistance(z, zs, transverse=True)

    dol = rl / (1.0 + z)
    dos = rs / (1.0 + zs)
    dls = rls / (1.0 + zs)

    surf_crit = (
        1.0e6
        * (const.c**2 / (4.0 * np.pi * const.G) / (const.M_sun / const.pc))
        * (dos / (dol * dls))
    )

    if comoving:
        surf_crit /= (1.0 + z) ** 2

    return surf_crit


def inv_sigma_crit(z, zs, cosmo, comoving=False):
    """1/sigma_crit [h^-1 M_sun^-1 Mpc^2]; returns 0 when z >= zs."""
    if zs <= z:
        return 0.0
    return 1.0 / sigma_crit(z, zs, cosmo, comoving)


def concent_m(m, z, cosmo, mdef="vir"):
    cosmology.setCurrent(cosmo)
    return concentration.concentration(m, mdef, z, model="diemer19")


def calc_rvir(m, z, cosmo, mdef="vir", comoving=False):
    cosmology.setCurrent(cosmo)
    _, rvir = profile_nfw.NFWProfile.nativeParameters(m, 1.0, z, mdef)
    rvir_out = rvir * 1e-3
    if comoving:
        rvir_out *= 1.0 + z
    return rvir_out


################################################################
# NFW profile  —  projected quantities via colossus
################################################################
# colossus NFWProfile.surfaceDensity / deltaSigma:
#   input r in physical kpc/h → output in M_sun h/kpc^2
# Unit conversion to h^-1 Mpc system (*1e6) and comoving (*1/(1+z)^2)


def _nfw_profile(m, c, z, cosmo, mdef):
    cosmology.setCurrent(cosmo)
    return profile_nfw.NFWProfile(M=m, c=c, z=z, mdef=mdef)


def _r_to_phys_kpc(r, z, comoving):
    """Convert r [h^-1 Mpc, physical or comoving] to physical kpc/h."""
    return r * 1e3 if not comoving else r * 1e3 / (1.0 + z)


def nfw_sigma(r, m, c, z, cosmo, mdef="vir", comoving=False):
    p = _nfw_profile(m, c, z, cosmo, mdef)
    sigma = p.surfaceDensity(_r_to_phys_kpc(r, z, comoving)) * 1e6
    if comoving:
        sigma /= (1.0 + z) ** 2
    return sigma


def nfw_dsigma(r, m, c, z, cosmo, mdef="vir", comoving=False):
    p = _nfw_profile(m, c, z, cosmo, mdef)
    ds = p.deltaSigma(_r_to_phys_kpc(r, z, comoving)) * 1e6
    if comoving:
        ds /= (1.0 + z) ** 2
    return ds


def nfw_bsigma(r, m, c, z, cosmo, mdef="vir", comoving=False):
    return nfw_sigma(r, m, c, z, cosmo, mdef, comoving) + nfw_dsigma(
        r, m, c, z, cosmo, mdef, comoving
    )


def nfw_kappa(r, m, c, z, zs, cosmo, mdef="vir", comoving=False):
    return nfw_sigma(r, m, c, z, cosmo, mdef, comoving) * inv_sigma_crit(
        z, zs, cosmo, comoving
    )


def nfw_kappa_ave(r, m, c, z, zs, cosmo, mdef="vir", comoving=False):
    return nfw_bsigma(r, m, c, z, cosmo, mdef, comoving) * inv_sigma_crit(
        z, zs, cosmo, comoving
    )


def nfw_gamma(r, m, c, z, zs, cosmo, mdef="vir", comoving=False):
    return nfw_dsigma(r, m, c, z, cosmo, mdef, comoving) * inv_sigma_crit(
        z, zs, cosmo, comoving
    )


################################################################
# Takada-Jain (sharply truncated NFW) profile
################################################################


def _calc_crhos_crs(m, c, z, cosmo, mdef, comoving):
    cosmology.setCurrent(cosmo)
    rhos, rs = profile_nfw.NFWProfile.nativeParameters(m, c, z, mdef)
    rs_out = rs * 1e-3
    rhos_out = rhos * 1e9
    if comoving:
        rs_out *= 1.0 + z
        rhos_out /= (1.0 + z) ** 3
    return rhos_out, rs_out


def m_nfw(x):
    return np.log(1.0 + x) - x / (1.0 + x)


def tj_sigma_dl(x, c):
    f = np.zeros_like(x)

    m = x >= c
    f[m] = 0.0

    m = (x > 1.0 + 1e-4) & (x < c)
    f[m] = (-1.0) * np.sqrt(c * c - x[m] ** 2) / (
        (1.0 - x[m] ** 2) * (1.0 + c)
    ) - np.arccos((x[m] ** 2 + c) / (x[m] * (1.0 + c))) / (
        (x[m] ** 2 - 1.0) * np.sqrt(x[m] ** 2 - 1.0)
    )

    m = (x >= 1.0 - 1e-4) & (x <= 1.0 + 1e-4)
    f[m] = np.sqrt(c * c - 1.0) * (1.0 + 1.0 / (1.0 + c)) / (3.0 * (1.0 + c))

    m = x < 1.0 - 1e-4
    f[m] = (-1.0) * np.sqrt(c * c - x[m] ** 2) / (
        (1.0 - x[m] ** 2) * (1.0 + c)
    ) + np.arccosh((x[m] ** 2 + c) / (x[m] * (1.0 + c))) / (
        (1.0 - x[m] ** 2) * np.sqrt(1.0 - x[m] ** 2)
    )

    return 0.5 * f


def tj_bsigma_dl(x, c):
    f = np.zeros_like(x)

    m = x >= c
    f[m] = m_nfw(c) / x[m] ** 2

    m = (x > 1.0 + 1e-4) & (x < c)
    f[m] = (
        (np.sqrt(c * c - x[m] ** 2) - c) / (x[m] ** 2 * (1.0 + c))
        + np.log(x[m] * (1.0 + c) / (c + np.sqrt(c * c - x[m] ** 2))) / x[m] ** 2
        + np.arccos((x[m] ** 2 + c) / (x[m] * (1.0 + c)))
        / (x[m] ** 2 * np.sqrt(x[m] ** 2 - 1.0))
    )

    m = (x >= 1.0 - 1e-4) & (x <= 1.0 + 1e-4)
    f[m] = (2.0 * np.sqrt(c * c - 1.0) - c) / (1.0 + c) + np.log(
        (1.0 + c) / (c + np.sqrt(c * c - 1.0))
    )

    m = x < 1.0 - 1e-4
    f[m] = (
        (np.sqrt(c * c - x[m] ** 2) - c) / (x[m] ** 2 * (1.0 + c))
        + np.log(x[m] * (1.0 + c) / (c + np.sqrt(c * c - x[m] ** 2))) / x[m] ** 2
        + np.arccosh((x[m] ** 2 + c) / (x[m] * (1.0 + c)))
        / (x[m] ** 2 * np.sqrt(1.0 - x[m] ** 2))
    )

    return f


def tj_dsigma_dl(x, c):
    return tj_bsigma_dl(x, c) - tj_sigma_dl(x, c)


def tj_sigma(r, m, c, z, cosmo, mdef="vir", comoving=False):
    _, crs = _calc_crhos_crs(m, c, z, cosmo, mdef, comoving)
    return (m / (np.pi * crs**2)) * tj_sigma_dl(r / crs, c)


def tj_bsigma(r, m, c, z, cosmo, mdef="vir", comoving=False):
    _, crs = _calc_crhos_crs(m, c, z, cosmo, mdef, comoving)
    return (m / (np.pi * crs**2)) * tj_bsigma_dl(r / crs, c)


def tj_dsigma(r, m, c, z, cosmo, mdef="vir", comoving=False):
    _, crs = _calc_crhos_crs(m, c, z, cosmo, mdef, comoving)
    return (m / (np.pi * crs**2)) * tj_dsigma_dl(r / crs, c)


def tj_kappa(r, m, c, z, zs, cosmo, mdef="vir", comoving=False):
    return tj_sigma(r, m, c, z, cosmo, mdef, comoving) * inv_sigma_crit(
        z, zs, cosmo, comoving
    )


def tj_kappa_ave(r, m, c, z, zs, cosmo, mdef="vir", comoving=False):
    return tj_bsigma(r, m, c, z, cosmo, mdef, comoving) * inv_sigma_crit(
        z, zs, cosmo, comoving
    )


def tj_gamma(r, m, c, z, zs, cosmo, mdef="vir", comoving=False):
    return tj_dsigma(r, m, c, z, cosmo, mdef, comoving) * inv_sigma_crit(
        z, zs, cosmo, comoving
    )


def tj_sigma_f(k, m, c, z, cosmo, mdef="vir", comoving=False):
    _, crs = _calc_crhos_crs(m, c, z, cosmo, mdef, comoving)
    return m * y_tj(k * crs, c)


def tj_kappa_f(k, m, c, z, zs, cosmo, mdef="vir", comoving=False):
    return tj_sigma_f(k, m, c, z, cosmo, mdef, comoving) * inv_sigma_crit(
        z, zs, cosmo, comoving
    )


def y_tj(x, c):
    si1, ci1 = sc.sici(x)
    si2, ci2 = sc.sici((1.0 + c) * x)
    f = (
        np.cos(x) * (ci2 - ci1)
        + np.sin(x) * (si2 - si1)
        - np.sin(c * x) / ((1.0 + c) * x)
    )
    return f / m_nfw(c)


################################################################
# Baltz-Marshall-Oguri (smoothly truncated NFW) profile
################################################################


def _f_dl_bmo(x):
    """Returns (F(x)-1)/(1-x^2), F(x) where F(x) = 2*nfw_sigma_dl*(1-x^2)+1."""
    # Use colossus NFW surface density for the dimensionless f1 term
    # f1 = 2*nfw_sigma_dl(x),  f2 = f1*(1-x^2) + 1
    # We need nfw_sigma_dl — compute it directly without colossus overhead
    x = np.asarray(x, dtype=float)
    f1 = np.zeros_like(x)
    m = x > 1.0 + 1e-4
    a = np.sqrt((x[m] - 1.0) / (x[m] + 1.0))
    f1[m] = (1.0 - 2.0 * np.arctan(a) / np.sqrt(x[m] ** 2 - 1.0)) / (x[m] ** 2 - 1.0)
    m2 = x < 1.0 - 1e-4
    a2 = np.sqrt((1.0 - x[m2]) / (x[m2] + 1.0))
    f1[m2] = (2.0 * np.arctanh(a2) / np.sqrt(1.0 - x[m2] ** 2) - 1.0) / (
        1.0 - x[m2] ** 2
    )
    m3 = (x >= 1.0 - 1e-4) & (x <= 1.0 + 1e-4)
    f1[m3] = 11.0 / 15.0 - 0.4 * x[m3]
    f2 = f1 * (1.0 - x**2) + 1.0
    return f1, f2


def _l_dl_bmo(x, t):
    return np.log(x / (np.sqrt(x**2 + t**2) + t))


def bmo_sigma_dl(x, t):
    ff1, ff2 = _f_dl_bmo(x)
    f1 = t**4 / (4.0 * (t**2 + 1.0) ** 3)
    f2 = (
        2.0 * (t**2 + 1.0) * ff1
        + 8.0 * ff2
        + (t**4 - 1.0) / (t**2 * (t**2 + x**2))
        - np.pi
        * (4.0 * (t**2 + x**2) + t**2 + 1.0)
        / ((t**2 + x**2) * np.sqrt(t**2 + x**2))
        + (t**2 * (t**4 - 1.0) + (t**2 + x**2) * (3.0 * t**4 - 6.0 * t**2 - 1.0))
        * _l_dl_bmo(x, t)
        / (t**3 * (t**2 + x**2) * np.sqrt(t**2 + x**2))
    )
    return f1 * f2


def bmo_bsigma_dl(x, t):
    ff1, ff2 = _f_dl_bmo(x)
    f1 = t**4 / (2.0 * (t**2 + 1.0) ** 3 * x**2)
    f2 = (
        2.0 * (t**2 + 4.0 * x**2 - 3.0) * ff2
        + (1.0 / t) * (np.pi * (3.0 * t**2 - 1.0) + 2.0 * t * (t**2 - 3.0) * np.log(t))
        + (1.0 / (t**3 * np.sqrt(t**2 + x**2)))
        * (
            (-1.0) * t**3 * np.pi * (4.0 * x**2 + 3.0 * t**2 - 1.0)
            + (2.0 * t**4 * (t**2 - 3.0) + x**2 * (3.0 * t**4 - 6.0 * t**2 - 1.0))
            * _l_dl_bmo(x, t)
        )
    )
    return f1 * f2


def bmo_dsigma_dl(x, t):
    return bmo_bsigma_dl(x, t) - bmo_sigma_dl(x, t)


def m_bmo(x, t):
    f1 = t**2 / (2.0 * (t**2 + 1.0) ** 3 * (1.0 + x) * (t**2 + x**2))
    f2 = (t**2 + 1.0) * x * (
        x * (x + 1.0) - t**2 * (x - 1.0) * (2.0 + 3.0 * x) - 2.0 * t**4
    ) + t * (x + 1.0) * (t**2 + x**2) * (
        2.0 * (3.0 * t**2 - 1.0) * np.arctan(x / t)
        + t * (t**2 - 3.0) * np.log(t**2 * (1.0 + x) ** 2 / (t**2 + x**2))
    )
    return f1 * f2


def m_bmo_tot(t):
    f1 = t**2 / (2.0 * (t**2 + 1.0) ** 3)
    f2 = (3.0 * t**2 - 1.0) * (np.pi * t - t**2 - 1.0) + 2.0 * t**2 * (
        t**2 - 3.0
    ) * np.log(t)
    return f1 * f2


def bmo_sigma(r, m, c, tv, z, cosmo, mdef="vir", comoving=False):
    crhos, crs = _calc_crhos_crs(m, c, z, cosmo, mdef, comoving)
    return 4.0 * crhos * crs * bmo_sigma_dl(r / crs, tv * c)


def bmo_bsigma(r, m, c, tv, z, cosmo, mdef="vir", comoving=False):
    crhos, crs = _calc_crhos_crs(m, c, z, cosmo, mdef, comoving)
    return 4.0 * crhos * crs * bmo_bsigma_dl(r / crs, tv * c)


def bmo_dsigma(r, m, c, tv, z, cosmo, mdef="vir", comoving=False):
    crhos, crs = _calc_crhos_crs(m, c, z, cosmo, mdef, comoving)
    return 4.0 * crhos * crs * bmo_dsigma_dl(r / crs, tv * c)


def bmo_kappa(r, m, c, tv, z, zs, cosmo, mdef="vir", comoving=False):
    return bmo_sigma(r, m, c, tv, z, cosmo, mdef, comoving) * inv_sigma_crit(
        z, zs, cosmo, comoving
    )


def bmo_kappa_ave(r, m, c, tv, z, zs, cosmo, mdef="vir", comoving=False):
    return bmo_bsigma(r, m, c, tv, z, cosmo, mdef, comoving) * inv_sigma_crit(
        z, zs, cosmo, comoving
    )


def bmo_gamma(r, m, c, tv, z, zs, cosmo, mdef="vir", comoving=False):
    return bmo_dsigma(r, m, c, tv, z, cosmo, mdef, comoving) * inv_sigma_crit(
        z, zs, cosmo, comoving
    )


def bmo_sigma_f(k, m, c, tv, z, cosmo, mdef="vir", comoving=False):
    _, crs = _calc_crhos_crs(m, c, z, cosmo, mdef, comoving)
    return m * y_bmo(k * crs, c, tv * c)


def bmo_kappa_f(k, m, c, tv, z, zs, cosmo, mdef="vir", comoving=False):
    return bmo_sigma_f(k, m, c, tv, z, cosmo, mdef, comoving) * inv_sigma_crit(
        z, zs, cosmo, comoving
    )


def y_bmo(x, c, t):
    si, ci = sc.sici(x)
    p, q = _y_bmo_pq(t * x)
    sx, cx = np.sin(x), np.cos(x)
    f1 = t / (4.0 * m_nfw(c) * (1.0 + t**2) ** 3 * x)
    f2 = (
        2.0 * (3.0 * t**4 - 6.0 * t**2 - 1.0) * p
        - 2.0 * t * (t**4 - 1.0) * x * q
        - 2.0 * t**2 * np.pi * np.exp(-t * x) * ((t**2 + 1.0) * x + 4.0 * t)
        + 2.0 * t**3 * (np.pi - 2.0 * si) * (4.0 * cx + (t**2 + 1.0) * x * sx)
        + 4.0 * t**3 * ci * (4.0 * sx - (t**2 + 1.0) * x * cx)
    )
    return f1 * f2


def _y_bmo_pq(x):
    p = np.zeros_like(x)
    q = np.zeros_like(x)
    m = x < 14.0
    shi, chi = sc.shichi(x[m])
    p[m] = np.sinh(x[m]) * chi - np.cosh(x[m]) * shi
    q[m] = np.cosh(x[m]) * chi - np.sinh(x[m]) * shi
    m = ~m
    p[m] = -1.0 / x[m]
    q[m] = 1.0 / x[m] ** 2
    return p, q


################################################################
# Hernquist profile
################################################################


def _conv_re_to_rb(re, z, comoving):
    return 0.551 * re * (1.0 + z) if comoving else 0.551 * re


def hern_sigma_dl(x):
    x = np.asarray(x, dtype=float)
    f = np.zeros_like(x)
    m = x > 1.0 + 1e-4
    a = np.sqrt(x[m] ** 2 - 1.0)
    f[m] = ((2.0 + x[m] ** 2) * np.arctan(a) / a - 3.0) / (x[m] ** 2 - 1.0) ** 2
    m2 = x < 1.0 - 1e-4
    a2 = np.sqrt(1.0 - x[m2] ** 2)
    f[m2] = ((2.0 + x[m2] ** 2) * np.arctanh(a2) / a2 - 3.0) / (x[m2] ** 2 - 1.0) ** 2
    m3 = (x >= 1.0 - 1e-4) & (x <= 1.0 + 1e-4)
    f[m3] = 4.0 / 15.0 - 16.0 * (x[m3] - 1.0) / 35.0
    return f


def hern_bsigma_dl(x):
    x = np.asarray(x, dtype=float)
    f = np.zeros_like(x)
    m = x > 1.0 + 1e-4
    a = np.sqrt(x[m] ** 2 - 1.0)
    f[m] = 2.0 * (1.0 - np.arctan(a) / a) / (x[m] ** 2 - 1.0)
    m2 = x < 1.0 - 1e-4
    a2 = np.sqrt(1.0 - x[m2] ** 2)
    f[m2] = 2.0 * (1.0 - np.arctanh(a2) / a2) / (x[m2] ** 2 - 1.0)
    m3 = (x >= 1.0 - 1e-4) & (x <= 1.0 + 1e-4)
    f[m3] = 2.0 / 3.0 - 4.0 * (x[m3] - 1.0) / 5.0
    return f


def hern_dsigma_dl(x):
    return hern_bsigma_dl(x) - hern_sigma_dl(x)


def hern_sigma(r, m, re, z, comoving=False):
    rb = _conv_re_to_rb(re, z, comoving)
    return m * hern_sigma_dl(r / rb) / (2.0 * np.pi * rb**2)


def hern_bsigma(r, m, re, z, comoving=False):
    rb = _conv_re_to_rb(re, z, comoving)
    return m * hern_bsigma_dl(r / rb) / (2.0 * np.pi * rb**2)


def hern_dsigma(r, m, re, z, comoving=False):
    rb = _conv_re_to_rb(re, z, comoving)
    return m * hern_dsigma_dl(r / rb) / (2.0 * np.pi * rb**2)


def hern_kappa(r, m, re, z, zs, cosmo, comoving=False):
    return hern_sigma(r, m, re, z, comoving) * inv_sigma_crit(z, zs, cosmo, comoving)


def hern_kappa_ave(r, m, re, z, zs, cosmo, comoving=False):
    return hern_bsigma(r, m, re, z, comoving) * inv_sigma_crit(z, zs, cosmo, comoving)


def hern_gamma(r, m, re, z, zs, cosmo, comoving=False):
    return hern_dsigma(r, m, re, z, comoving) * inv_sigma_crit(z, zs, cosmo, comoving)


################################################################
# 2-halo term
################################################################


def sigma_2h_nob_f(k, z, cosmo, comoving=False):
    rhom = cosmo.rho_m(0.0) * 1e9
    growth = cosmo.growthFactor(z)
    kk = k if comoving else k / (1.0 + z)
    pk = cosmo.matterPowerSpectrum(kk)
    return rhom * growth**2 * pk


def sigma_2h_f(k, m, z, cosmo, mdef="vir", comoving=False):
    bh = bias.haloBias(m, z, mdef, model="tinker10")
    return bh * sigma_2h_nob_f(k, z, cosmo, comoving)


def kappa_2h_nob_f(k, z, zs, cosmo, comoving=False):
    return sigma_2h_nob_f(k, z, cosmo, comoving) * inv_sigma_crit(
        z, zs, cosmo, comoving
    )


def kappa_2h_f(k, m, z, zs, cosmo, mdef="vir", comoving=False):
    return sigma_2h_f(k, m, z, cosmo, mdef, comoving) * inv_sigma_crit(
        z, zs, cosmo, comoving
    )


################################################################
# Off-centering via FFT
# flag_d = 0 -> sigma,  flag_d = 1 -> dsigma
################################################################


def _calc_fft(k, f, r_bin, flag_d):
    if flag_d == 0:
        rr, ff = fftlog.pk2wp(k, f, 1.01, N_extrap_low=2048)
    else:
        rr, ff = fftlog.pk2dwp(k, f, 1.01, N_extrap_low=2048)
    return CubicSpline(rr, ff)(r_bin)


def bmo_sigma_off_fft(r, m, c, tv, z, roff, flag_d, cosmo, mdef="vir", comoving=False):
    k_bin = 10 ** np.arange(-3.0, 3.5, 0.05)
    pk = bmo_sigma_f(k_bin, m, c, tv, z, cosmo, mdef, comoving) * j0(k_bin * roff)
    return _calc_fft(k_bin, pk, r, flag_d)


def bmo_kappa_off_fft(
    r, m, c, tv, z, roff, zs, flag_d, cosmo, mdef="vir", comoving=False
):
    return bmo_sigma_off_fft(
        r, m, c, tv, z, roff, flag_d, cosmo, mdef, comoving
    ) * inv_sigma_crit(z, zs, cosmo, comoving)


def sigma_off_2h_fft(
    r, m, c, f_cen, sig_off, tv, z, flag_d, flag_out, cosmo, mdef="vir", comoving=False
):
    k_bin = 10 ** np.arange(-2.5, 3.5, 0.05)
    pk1 = sigma_2h_f(k_bin, m, z, cosmo, mdef, comoving)
    pk2 = bmo_sigma_f(k_bin, m, c, tv, z, cosmo, mdef, comoving)

    if flag_out == 0:
        pk = pk1 + (1.0 - f_cen) * pk2 * np.exp(-0.5 * k_bin**2 * sig_off**2)
    elif flag_out == 1:
        return 0.0 * r
    elif flag_out == 2:
        pk = pk1
    elif flag_out == 3:
        pk = (1.0 - f_cen) * pk2 * np.exp(-0.5 * k_bin**2 * sig_off**2)

    return _calc_fft(k_bin, pk, r, flag_d)


def kappa_off_2h_fft(
    r,
    m,
    c,
    f_cen,
    sig_off,
    tv,
    z,
    zs,
    flag_d,
    flag_out,
    cosmo,
    mdef="vir",
    comoving=False,
):
    return sigma_off_2h_fft(
        r, m, c, f_cen, sig_off, tv, z, flag_d, flag_out, cosmo, mdef, comoving
    ) * inv_sigma_crit(z, zs, cosmo, comoving)


def sigma_off(
    r, m, z, f_cen, sig_off, flag_d, flag_out, cosmo, mdef="vir", comoving=False
):
    c = concent_m(m, z, cosmo, mdef)
    tv = 2.5

    if flag_out <= 1:
        s1h = (
            bmo_sigma(r, m, c, tv, z, cosmo, mdef, comoving)
            if flag_d == 0
            else bmo_dsigma(r, m, c, tv, z, cosmo, mdef, comoving)
        )
    else:
        s1h = 0.0

    soff2h = sigma_off_2h_fft(
        r, m, c, f_cen, sig_off, tv, z, flag_d, flag_out, cosmo, mdef, comoving
    )
    return f_cen * s1h + soff2h


def kappa_off(
    r, m, z, f_cen, sig_off, zs, flag_d, flag_out, cosmo, mdef="vir", comoving=False
):
    return sigma_off(
        r, m, z, f_cen, sig_off, flag_d, flag_out, cosmo, mdef, comoving
    ) * inv_sigma_crit(z, zs, cosmo, comoving)


################################################################
# Fixed mis-centering (delta-function PDF)
################################################################


def tj_sigma_fixroff_fft(r, m, c, z, roff, flag_d, cosmo, mdef="vir", comoving=False):
    k_bin = 10 ** np.arange(-3.0, 3.5, 0.01)
    pk = tj_sigma_f(k_bin, m, c, z, cosmo, mdef, comoving) * j0(k_bin * roff)
    return _calc_fft(k_bin, pk, r, flag_d)


def tj_kappa_fixroff_fft(
    r, m, c, z, roff, zs, flag_d, cosmo, mdef="vir", comoving=False
):
    return tj_sigma_fixroff_fft(
        r, m, c, z, roff, flag_d, cosmo, mdef, comoving
    ) * inv_sigma_crit(z, zs, cosmo, comoving)


def bmo_sigma_fixroff_fft(
    r, m, c, tv, z, roff, flag_d, cosmo, mdef="vir", comoving=False
):
    k_bin = 10 ** np.arange(-3.0, 3.5, 0.01)
    pk = bmo_sigma_f(k_bin, m, c, tv, z, cosmo, mdef, comoving) * j0(k_bin * roff)
    return _calc_fft(k_bin, pk, r, flag_d)


def bmo_kappa_fixroff_fft(
    r, m, c, tv, z, roff, zs, flag_d, cosmo, mdef="vir", comoving=False
):
    return bmo_sigma_fixroff_fft(
        r, m, c, tv, z, roff, flag_d, cosmo, mdef, comoving
    ) * inv_sigma_crit(z, zs, cosmo, comoving)


################################################################
# main
################################################################

if __name__ == "__main__":
    my_cosmo = {
        "flat": True,
        "H0": 70.0,
        "Om0": 0.3,
        "Ob0": 0.05,
        "sigma8": 0.81,
        "ns": 0.96,
    }
    cosmo = cosmology.setCosmology("my_cosmo", my_cosmo)

    r_bin = 10 ** np.arange(np.log10(0.01), np.log10(100.0), 0.02)
    m, z = 1.0e14, 0.5
    f_cen, sig_off = 0.6, 0.3
    mdef, comoving = "200m", False

    f1 = sigma_off(r_bin, m, z, f_cen, sig_off, 0, 0, cosmo, mdef, comoving)
    f2 = sigma_off(r_bin, m, z, f_cen, sig_off, 0, 1, cosmo, mdef, comoving)
    f3 = sigma_off(r_bin, m, z, f_cen, sig_off, 0, 2, cosmo, mdef, comoving)
    f4 = sigma_off(r_bin, m, z, f_cen, sig_off, 0, 3, cosmo, mdef, comoving)
    f5 = sigma_off(r_bin, m, z, f_cen, sig_off, 1, 0, cosmo, mdef, comoving)
    f6 = sigma_off(r_bin, m, z, f_cen, sig_off, 1, 1, cosmo, mdef, comoving)
    f7 = sigma_off(r_bin, m, z, f_cen, sig_off, 1, 2, cosmo, mdef, comoving)
    f8 = sigma_off(r_bin, m, z, f_cen, sig_off, 1, 3, cosmo, mdef, comoving)

    for i in range(len(r_bin)):
        print(
            "%e %e %e %e %e %e %e %e %e"
            % (r_bin[i], f1[i], f2[i], f3[i], f4[i], f5[i], f6[i], f7[i], f8[i])
        )
