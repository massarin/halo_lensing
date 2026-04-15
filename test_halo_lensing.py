"""
Regression tests: compare refactored fftlog.py / halo_lensing.py
against the originals (fftlog_orig.py / halo_lensing_orig.py).

Run:  python test_halo_lensing.py
Out:  test_plots.pdf
"""

import importlib
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from colossus.cosmology import cosmology
from matplotlib.backends.backend_pdf import PdfPages

import fftlog
import halo_lensing as hl
import original.fftlog_orig as fftlog_orig

# halo_lensing_orig uses `import fftlog` — point it at fftlog_orig
sys.modules["fftlog"] = fftlog_orig
import original.halo_lensing_orig as hl_orig

sys.modules["fftlog"] = fftlog  # restore

COSMO_PARAMS = {
    "flat": True,
    "H0": 70.0,
    "Om0": 0.3,
    "Ob0": 0.05,
    "sigma8": 0.81,
    "ns": 0.96,
}


def setup_cosmo():
    return cosmology.setCosmology("test_cosmo", COSMO_PARAMS)


def check(new, old, rtol, label):
    """Assert max relative error < rtol; return the error.
    Normalise by the peak |old| to handle zero-crossings gracefully."""
    scale = np.max(np.abs(old))
    err = np.max(np.abs(new - old)) / scale
    assert err < rtol, f"{label}: max rel err {err:.2e} >= {rtol}"
    return err


# ── fftlog tests ──────────────────────────────────────────────────────────────


def test_fftlog_fftlog():
    """fftlog.fftlog(ell) matches original."""
    k = np.logspace(-2, 2, 512)
    fx = k**1.5 * np.exp(-(k**2))

    new = fftlog.fftlog(k, fx)
    old = fftlog_orig.fftlog(k, fx)

    y_new, F_new = new.fftlog(0)
    y_old, F_old = old.fftlog(0)

    err_y = check(y_new, y_old, 1e-12, "fftlog y-grid")
    err_F = check(F_new, F_old, 1e-10, "fftlog F(y)")
    return y_new, F_new, y_old, F_old, err_F


def test_fftlog_hankel():
    """hankel.hankel(n) matches original."""
    k = np.logspace(-2, 2, 512)
    fx = k**0.5 * np.exp(-(k**2) / 4)

    new = fftlog.hankel(k, fx, nu=1.01)
    old = fftlog_orig.hankel(k, fx, nu=1.01)

    y_new, H_new = new.hankel(0)
    y_old, H_old = old.hankel(0)

    err = check(H_new, H_old, 1e-10, "hankel H(y)")
    return y_new, H_new, y_old, H_old, err


def test_pk2xi():
    # The original pk2xi had a bug: passed N_pad in the c_window_width slot.
    # The fix changes c_window smoothing, producing an O(1e-5) difference.
    k = np.logspace(-2, 2, 512)
    pk = k**1.5 * np.exp(-(k**2))

    r_new, xi_new = fftlog.pk2xi(k, pk)
    r_old, xi_old = fftlog_orig.pk2xi(k, pk)

    err = check(xi_new, xi_old, 1e-4, "pk2xi")
    return r_new, xi_new, r_old, xi_old, err


def test_pk2wp():
    # Same bug fix as pk2xi affects c_window_width.
    k = np.logspace(-2, 2, 512)
    pk = k**0.5 * np.exp(-(k**2) / 4)

    rp_new, wp_new = fftlog.pk2wp(k, pk)
    rp_old, wp_old = fftlog_orig.pk2wp(k, pk)

    err = check(wp_new, wp_old, 5e-4, "pk2wp")
    return rp_new, wp_new, rp_old, wp_old, err


def test_pk2dwp():
    k = np.logspace(-2, 2, 512)
    pk = k**0.5 * np.exp(-(k**2) / 4)

    rp_new, dwp_new = fftlog.pk2dwp(k, pk)
    rp_old, dwp_old = fftlog_orig.pk2dwp(k, pk)

    err = check(dwp_new, dwp_old, 5e-4, "pk2dwp")
    return rp_new, dwp_new, rp_old, dwp_old, err


# ── halo_lensing tests ────────────────────────────────────────────────────────


def test_nfw_sigma():
    cosmo = setup_cosmo()
    m, c, z = 1e14, 5.0, 0.3
    r = np.logspace(-1, 1.5, 100)

    new = hl.nfw_sigma(r, m, c, z, cosmo)
    old = hl_orig.nfw_sigma(r, m, c, z, cosmo)

    err = check(new, old, 1e-6, "nfw_sigma")
    return r, new, old, err


def test_nfw_dsigma():
    cosmo = setup_cosmo()
    m, c, z = 1e14, 5.0, 0.3
    r = np.logspace(-1, 1.5, 100)

    new = hl.nfw_dsigma(r, m, c, z, cosmo)
    old = hl_orig.nfw_dsigma(r, m, c, z, cosmo)

    err = check(new, old, 1e-5, "nfw_dsigma")
    return r, new, old, err


def test_tj_sigma():
    cosmo = setup_cosmo()
    m, c, z = 1e14, 5.0, 0.3
    r = np.logspace(-1, 0.5, 60)

    new = hl.tj_sigma(r, m, c, z, cosmo)
    old = hl_orig.tj_sigma(r, m, c, z, cosmo)

    err = check(new, old, 1e-10, "tj_sigma")
    return r, new, old, err


def test_bmo_sigma():
    cosmo = setup_cosmo()
    m, c, tv, z = 1e14, 5.0, 2.5, 0.3
    r = np.logspace(-1, 1.5, 100)

    new = hl.bmo_sigma(r, m, c, tv, z, cosmo)
    old = hl_orig.bmo_sigma(r, m, c, tv, z, cosmo)

    err = check(new, old, 1e-10, "bmo_sigma")
    return r, new, old, err


def test_bmo_dsigma():
    cosmo = setup_cosmo()
    m, c, tv, z = 1e14, 5.0, 2.5, 0.3
    r = np.logspace(-1, 1.5, 100)

    new = hl.bmo_dsigma(r, m, c, tv, z, cosmo)
    old = hl_orig.bmo_dsigma(r, m, c, tv, z, cosmo)

    err = check(new, old, 1e-10, "bmo_dsigma")
    return r, new, old, err


def test_hern_sigma():
    r = np.logspace(-1, 1.5, 80)
    new = hl.hern_sigma(r, 1e12, 0.05, 0.3)
    old = hl_orig.hern_sigma(r, 1e12, 0.05, 0.3)
    err = check(new, old, 1e-10, "hern_sigma")
    return r, new, old, err


def test_sigma_crit():
    cosmo = setup_cosmo()
    new = hl.sigma_crit(0.3, 1.0, cosmo)
    old = hl_orig.sigma_crit(0.3, 1.0, cosmo)
    err = abs(new - old) / abs(old)
    assert err < 1e-6, f"sigma_crit rel err {err:.2e}"
    return new, old, err


# ── plotting ──────────────────────────────────────────────────────────────────


def _diff_panel(ax, x, new, old, title, xlabel):
    ax.semilogx(x, (new - old) / np.max(np.abs(old)))
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("(new - old) / max|old|")


def make_plots(pdf, results):
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    axes = axes.ravel()

    r_nfw, new_nfw, old_nfw, _ = results["nfw_sigma"]
    r_ds, new_ds, old_ds, _ = results["nfw_dsigma"]
    r_tj, new_tj, old_tj, _ = results["tj_sigma"]
    r_bmo, new_bmo, old_bmo, _ = results["bmo_sigma"]
    r_bds, new_bds, old_bds, _ = results["bmo_dsigma"]
    r_h, new_h, old_h, _ = results["hern_sigma"]
    r_xi, xi_new, _, xi_old, _ = results["pk2xi"]
    rp_wp, wp_new, _, wp_old, _ = results["pk2wp"]

    _diff_panel(axes[0], r_nfw, new_nfw, old_nfw, "nfw_sigma", "r")
    _diff_panel(axes[1], r_ds, new_ds, old_ds, "nfw_dsigma", "r")
    _diff_panel(axes[2], r_tj, new_tj, old_tj, "tj_sigma", "r")
    _diff_panel(axes[3], r_bmo, new_bmo, old_bmo, "bmo_sigma", "r")
    _diff_panel(axes[4], r_bds, new_bds, old_bds, "bmo_dsigma", "r")
    _diff_panel(axes[5], r_h, new_h, old_h, "hern_sigma", "r")

    # FFTLog xi
    axes[6].semilogx(r_xi, (xi_new - xi_old) / np.max(np.abs(xi_old)))
    axes[6].axhline(0, color="k", lw=0.8, ls="--")
    axes[6].set_title("pk2xi xi(r)")
    axes[6].set_xlabel("r")
    axes[6].set_ylabel("(new - old) / max|old|")

    # Hankel wp
    axes[7].semilogx(rp_wp, (wp_new - wp_old) / np.max(np.abs(wp_old)))
    axes[7].axhline(0, color="k", lw=0.8, ls="--")
    axes[7].set_title("pk2wp w_p(r_p)")
    axes[7].set_xlabel("r_p")
    axes[7].set_ylabel("(new - old) / max|old|")

    fig.suptitle("(new - original) / original relative difference", fontsize=12)
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

    print("Plots saved to test_plots.pdf")


# ── runner ────────────────────────────────────────────────────────────────────


def run_all():
    tests = [
        ("fftlog.fftlog(0)", test_fftlog_fftlog),
        ("hankel.hankel(0)", test_fftlog_hankel),
        ("pk2xi", test_pk2xi),
        ("pk2wp", test_pk2wp),
        ("pk2dwp", test_pk2dwp),
        ("nfw_sigma", test_nfw_sigma),
        ("nfw_dsigma", test_nfw_dsigma),
        ("tj_sigma", test_tj_sigma),
        ("bmo_sigma", test_bmo_sigma),
        ("bmo_dsigma", test_bmo_dsigma),
        ("hern_sigma", test_hern_sigma),
        ("sigma_crit", test_sigma_crit),
    ]

    results = {}
    for name, fn in tests:
        print(f"  {name:<25}", end=" ", flush=True)
        out = fn()
        # last element is always the error
        err = out[-1]
        print(f"OK  (max rel err = {err:.2e})")
        results[name.replace(".", "_").replace("(", "").replace(")", "")] = out

    print("\nAll tests passed.")

    # remap keys to shorter names for plotting
    plot_data = {
        "nfw_sigma": results["nfw_sigma"],
        "nfw_dsigma": results["nfw_dsigma"],
        "tj_sigma": results["tj_sigma"],
        "bmo_sigma": results["bmo_sigma"],
        "bmo_dsigma": results["bmo_dsigma"],
        "hern_sigma": results["hern_sigma"],
        "pk2xi": results["pk2xi"],
        "pk2wp": results["pk2wp"],
    }

    with PdfPages("test_plots.pdf") as pdf:
        make_plots(pdf, plot_data)


if __name__ == "__main__":
    run_all()
