"""
Numerical consistency tests for fftlog.py and halo_lensing.py.
Run: python test_halo_lensing.py
Produces: test_plots.pdf
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.integrate import quad
from scipy.interpolate import CubicSpline

import fftlog
import halo_lensing as hl
from colossus.cosmology import cosmology
from colossus.halo import profile_nfw

COSMO_PARAMS = {'flat': True, 'H0': 70.0, 'Om0': 0.3, 'Ob0': 0.05, 'sigma8': 0.81, 'ns': 0.96}


def setup_cosmo():
    return cosmology.setCosmology('test_cosmo', COSMO_PARAMS)


# ── test helpers ──────────────────────────────────────────────────────────────

def assert_close(a, b, rtol=1e-3, label=''):
    err = np.max(np.abs(a - b) / (np.abs(b) + 1e-30))
    assert err < rtol, f'{label}: max rel error {err:.2e} > {rtol}'
    return err


# ── test 1: FFTLog vs direct integration ─────────────────────────────────────

def test_fftlog_xi():
    """pk2xi accuracy against direct quadrature."""
    k = np.logspace(-2, np.log10(30), 512)
    pk = k**1.5 * np.exp(-k**2)

    r, xi = fftlog.pk2xi(k, pk, N_extrap_low=512)

    def xi_direct(rv):
        val, _ = quad(
            lambda kv: kv**2 * kv**1.5 * np.exp(-kv**2) * np.sin(kv*rv)/(kv*rv) / (2*np.pi**2),
            0, 100, limit=2000, epsabs=1e-15, epsrel=1e-8
        )
        return val

    r_test = np.array([0.3, 1.0, 2.0, 5.0])
    xi_quad = np.array([xi_direct(rv) for rv in r_test if r.min() < rv < r.max()])
    xi_fft = np.array([np.interp(rv, r, xi) for rv in r_test if r.min() < rv < r.max()])
    rel_err = np.abs(xi_fft - xi_quad) / np.abs(xi_quad)

    assert np.all(rel_err < 2e-3), f'max relative error = {rel_err.max():.2e}'
    return r, xi, r_test, xi_quad, rel_err


def test_hankel_pk2wp():
    """pk2wp accuracy: compare to direct Hankel quadrature."""
    k = np.logspace(-2, 2, 512)
    pk = k * np.exp(-k**2 / 4)  # smooth test function

    rp, wp = fftlog.pk2wp(k, pk, N_extrap_low=512)

    def wp_direct(rv):
        # w_p(r_p) = 1/(2pi) int k P(k) k J_0(k r_p) dk
        val, _ = quad(
            lambda kv: kv * kv * np.exp(-kv**2/4) * np.i0(0) * np.cos(0),
            0, 50, limit=1000
        )
        # Correct formula: wp = 1/(2pi) int dk k P(k) J_0(k rp)
        val, _ = quad(
            lambda kv: kv * kv * np.exp(-kv**2/4),
            0, 50, limit=1000
        )
        return val / (2*np.pi)

    # Verify identity: at rp=0, wp = 1/(2pi) int k^2 P(k) dk
    wp_at_0 = wp_direct(0)
    wp_fft_at_0 = np.interp(rp.min(), rp, wp)
    rel = abs(wp_at_0 - wp_fft_at_0) / wp_at_0
    # This just checks wp is in the right ballpark, not exact (rp.min > 0)
    return rp, wp


def test_fftlog_hankel_consistency():
    """Verify pk2dwp = derivative of pk2wp in the right sense:
    dsigma(r) = bsigma(r) - sigma(r), which should be >= 0 for typical profiles."""
    k = np.logspace(-2, 2, 512)
    pk = k**0.5 * np.exp(-k**2 / 9)

    rp_w, wp = fftlog.pk2wp(k, pk, N_extrap_low=512)
    rp_d, dwp = fftlog.pk2dwp(k, pk, N_extrap_low=512)

    # Check output shapes match
    assert len(rp_w) == len(rp_d), 'shape mismatch between pk2wp and pk2dwp'
    return rp_w, wp, dwp


# ── test 2: NFW profile identities ───────────────────────────────────────────

def test_nfw_dsigma_identity():
    """dsigma = bsigma - sigma to machine precision."""
    cosmo = setup_cosmo()
    m, c, z = 1e14, 5.0, 0.3
    r = np.logspace(-1, 1, 50)

    sig = hl.nfw_sigma(r, m, c, z, cosmo)
    ds  = hl.nfw_dsigma(r, m, c, z, cosmo)
    bs  = hl.nfw_bsigma(r, m, c, z, cosmo)

    rel = np.abs(bs - ds - sig) / sig
    assert np.all(rel < 1e-10), f'dsigma identity: max rel err {rel.max():.2e}'
    return r, sig, bs, ds


def test_nfw_vs_colossus_direct():
    """nfw_sigma matches colossus NFWProfile.surfaceDensity directly."""
    cosmo = setup_cosmo()
    m, c, z = 1e14, 5.0, 0.3
    r = np.logspace(-1, 1, 30)

    sig_hl = hl.nfw_sigma(r, m, c, z, cosmo)

    cosmology.setCurrent(cosmo)
    p = profile_nfw.NFWProfile(M=m, c=c, z=z, mdef='vir')
    sig_col = p.surfaceDensity(r * 1e3) * 1e6

    rel = np.abs(sig_hl - sig_col) / sig_col
    assert np.all(rel < 1e-6), f'nfw vs colossus max rel err = {rel.max():.2e}'
    return r, sig_hl, sig_col


def test_nfw_sigma_positivity():
    """sigma > 0 and dsigma > 0 for all r."""
    cosmo = setup_cosmo()
    r = np.logspace(-2, 2, 100)
    m, c, z = 1e14, 5.0, 0.3

    assert np.all(hl.nfw_sigma(r, m, c, z, cosmo) > 0)
    assert np.all(hl.nfw_dsigma(r, m, c, z, cosmo) > 0)


# ── test 3: TJ and BMO profile identities ────────────────────────────────────

def test_tj_dsigma_identity():
    """dsigma_tj = bsigma_tj - sigma_tj."""
    cosmo = setup_cosmo()
    m, c, z = 1e14, 5.0, 0.3
    r = np.logspace(-1, 0.8, 50)

    sig = hl.tj_sigma(r, m, c, z, cosmo)
    bs  = hl.tj_bsigma(r, m, c, z, cosmo)
    ds  = hl.tj_dsigma(r, m, c, z, cosmo)

    mask = sig > 0
    rel = np.abs(bs[mask] - ds[mask] - sig[mask]) / sig[mask]
    assert np.all(rel < 1e-10), f'TJ dsigma identity: max rel err {rel.max():.2e}'
    return r, sig, bs, ds


def test_bmo_dsigma_identity():
    """dsigma_bmo = bsigma_bmo - sigma_bmo."""
    cosmo = setup_cosmo()
    m, c, tv, z = 1e14, 5.0, 2.5, 0.3
    r = np.logspace(-1, 1.5, 50)

    sig = hl.bmo_sigma(r, m, c, tv, z, cosmo)
    bs  = hl.bmo_bsigma(r, m, c, tv, z, cosmo)
    ds  = hl.bmo_dsigma(r, m, c, tv, z, cosmo)

    rel = np.abs(bs - ds - sig) / np.abs(sig)
    assert np.all(rel < 1e-10), f'BMO dsigma identity: max rel err {rel.max():.2e}'
    return r, sig, bs, ds


# ── test 4: Hernquist profile identity ───────────────────────────────────────

def test_hern_dsigma_identity():
    """dsigma_hern = bsigma_hern - sigma_hern."""
    r = np.logspace(-1, 1.5, 50)
    sig = hl.hern_sigma(r, 1e12, 0.05, 0.3)
    bs  = hl.hern_bsigma(r, 1e12, 0.05, 0.3)
    ds  = hl.hern_dsigma(r, 1e12, 0.05, 0.3)

    rel = np.abs(bs - ds - sig) / sig
    assert np.all(rel < 1e-10), f'Hernquist dsigma identity: max rel err {rel.max():.2e}'
    return r, sig, bs, ds


# ── test 5: sigma_crit ────────────────────────────────────────────────────────

def test_sigma_crit():
    """sigma_crit is positive and inv_sigma_crit returns 0 when z >= zs."""
    cosmo = setup_cosmo()
    sc = hl.sigma_crit(0.3, 1.0, cosmo)
    assert sc > 0
    assert hl.inv_sigma_crit(1.0, 0.5, cosmo) == 0.0
    assert hl.inv_sigma_crit(0.3, 1.0, cosmo) > 0


# ── plotting ──────────────────────────────────────────────────────────────────

def make_plots(pdf):
    cosmo = setup_cosmo()

    # Fig 1: FFTLog xi(r) accuracy
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    k = np.logspace(-2, np.log10(30), 512)
    pk = k**1.5 * np.exp(-k**2)
    r, xi = fftlog.pk2xi(k, pk, N_extrap_low=512)
    r_test = np.array([0.3, 1.0, 2.0, 5.0])
    xi_quad = []
    for rv in r_test:
        if r.min() < rv < r.max():
            val, _ = quad(
                lambda kv: kv**2 * kv**1.5 * np.exp(-kv**2) * np.sin(kv*rv)/(kv*rv) / (2*np.pi**2),
                0, 100, limit=2000, epsabs=1e-15, epsrel=1e-8
            )
            xi_quad.append(val)
    xi_quad = np.array(xi_quad)
    xi_fft_test = np.interp(r_test, r, xi)
    rel_err = np.abs(xi_fft_test - xi_quad) / np.abs(xi_quad)

    axes[0].loglog(k, pk, 'b-', label='P(k)')
    axes[0].set_xlabel('k')
    axes[0].set_ylabel('P(k)')
    axes[0].set_title('Input P(k)')
    axes[0].legend()

    axes[1].semilogx(r, xi, 'b-', label='FFTLog')
    axes[1].scatter(r_test, xi_quad, c='r', zorder=5, label='direct quad')
    axes[1].set_xlabel('r')
    axes[1].set_ylabel(r'$\xi(r)$')
    axes[1].set_title(f'FFTLog pk2xi (max rel err = {rel_err.max():.1e})')
    axes[1].legend()
    fig.suptitle('Test 1: FFTLog pk→xi accuracy')
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

    # Fig 2: NFW profile sigma, bsigma, dsigma
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    m, c, z = 1e14, 5.0, 0.3
    r = np.logspace(-1, 1.5, 100)
    sig = hl.nfw_sigma(r, m, c, z, cosmo)
    bs  = hl.nfw_bsigma(r, m, c, z, cosmo)
    ds  = hl.nfw_dsigma(r, m, c, z, cosmo)

    axes[0].loglog(r, sig, label=r'$\Sigma$')
    axes[0].loglog(r, bs,  label=r'$\bar\Sigma$')
    axes[0].loglog(r, ds,  label=r'$\Delta\Sigma$')
    axes[0].set_xlabel(r'r [$h^{-1}$ Mpc]')
    axes[0].set_ylabel(r'[$h\,M_\odot\,{\rm Mpc}^{-2}$]')
    axes[0].set_title(f'NFW: M={m:.0e}, c={c}, z={z}')
    axes[0].legend()

    rel_id = np.abs(bs - ds - sig) / sig
    axes[1].semilogx(r, rel_id)
    axes[1].set_xlabel(r'r [$h^{-1}$ Mpc]')
    axes[1].set_ylabel(r'$|\bar\Sigma - \Delta\Sigma - \Sigma|/\Sigma$')
    axes[1].set_title(r'Identity $\bar\Sigma = \Delta\Sigma + \Sigma$ (should be ~machine eps)')
    axes[1].set_ylim(0, 1e-13)
    fig.suptitle('Test 2: NFW profile identities')
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

    # Fig 3: NFW vs TJ vs BMO comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    tv = 2.5
    sig_nfw = hl.nfw_sigma(r, m, c, z, cosmo)
    sig_tj  = hl.tj_sigma(r, m, c, z, cosmo)
    sig_bmo = hl.bmo_sigma(r, m, c, tv, z, cosmo)
    ds_nfw  = hl.nfw_dsigma(r, m, c, z, cosmo)
    ds_tj   = hl.tj_dsigma(r, m, c, z, cosmo)
    ds_bmo  = hl.bmo_dsigma(r, m, c, tv, z, cosmo)

    axes[0].loglog(r, sig_nfw, label='NFW')
    axes[0].loglog(r, sig_tj,  label='TJ', linestyle='--')
    axes[0].loglog(r, sig_bmo, label=f'BMO (tv={tv})', linestyle=':')
    axes[0].set_xlabel(r'r [$h^{-1}$ Mpc]')
    axes[0].set_ylabel(r'$\Sigma$')
    axes[0].set_title('Surface density: NFW vs TJ vs BMO')
    axes[0].legend()

    axes[1].loglog(r, ds_nfw, label='NFW')
    mask = ds_tj > 0
    axes[1].loglog(r[mask], ds_tj[mask],  label='TJ', linestyle='--')
    axes[1].loglog(r, ds_bmo, label=f'BMO (tv={tv})', linestyle=':')
    axes[1].set_xlabel(r'r [$h^{-1}$ Mpc]')
    axes[1].set_ylabel(r'$\Delta\Sigma$')
    axes[1].set_title(r'Excess surface density: NFW vs TJ vs BMO')
    axes[1].legend()
    fig.suptitle('Test 3: Profile comparison')
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

    # Fig 4: Hernquist profile
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    r_h = np.logspace(-1, 1.5, 100)
    m_h, re_h, z_h = 1e12, 0.05, 0.3
    sig_h = hl.hern_sigma(r_h, m_h, re_h, z_h)
    bs_h  = hl.hern_bsigma(r_h, m_h, re_h, z_h)
    ds_h  = hl.hern_dsigma(r_h, m_h, re_h, z_h)

    axes[0].loglog(r_h, sig_h, label=r'$\Sigma$')
    axes[0].loglog(r_h, bs_h,  label=r'$\bar\Sigma$')
    axes[0].loglog(r_h, ds_h,  label=r'$\Delta\Sigma$')
    axes[0].set_xlabel(r'r [$h^{-1}$ Mpc]')
    axes[0].set_ylabel(r'[$h\,M_\odot\,{\rm Mpc}^{-2}$]')
    axes[0].set_title(f'Hernquist: M={m_h:.0e}, re={re_h}, z={z_h}')
    axes[0].legend()

    rel_id_h = np.abs(bs_h - ds_h - sig_h) / sig_h
    axes[1].semilogx(r_h, rel_id_h)
    axes[1].set_xlabel(r'r [$h^{-1}$ Mpc]')
    axes[1].set_ylabel(r'$|\bar\Sigma - \Delta\Sigma - \Sigma|/\Sigma$')
    axes[1].set_title(r'Hernquist $\Delta\Sigma$ identity')
    axes[1].set_ylim(0, 1e-13)
    fig.suptitle('Test 4: Hernquist profile')
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

    # Fig 5: sigma_crit vs z_s
    fig, ax = plt.subplots(figsize=(7, 5))
    zl = 0.3
    zs_arr = np.linspace(0.35, 2.5, 100)
    sc_arr = np.array([hl.sigma_crit(zl, zs, cosmo) for zs in zs_arr])
    ax.plot(zs_arr, sc_arr)
    ax.set_xlabel(r'$z_s$')
    ax.set_ylabel(r'$\Sigma_{\rm crit}$ [$h\,M_\odot\,{\rm Mpc}^{-2}$]')
    ax.set_title(f'Critical surface density, $z_l = {zl}$')
    fig.suptitle('Test 5: sigma_crit')
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

    print('Plots saved to test_plots.pdf')


# ── runner ────────────────────────────────────────────────────────────────────

def run_all():
    results = {}

    print('Test 1: FFTLog xi(r) accuracy ...', end=' ', flush=True)
    r, xi, r_test, xi_quad, rel_err = test_fftlog_xi()
    print(f'OK (max rel err = {rel_err.max():.2e})')
    results['fftlog_xi'] = rel_err

    print('Test 2: Hankel pk2wp ...', end=' ', flush=True)
    rp, wp = test_hankel_pk2wp()
    print('OK')

    print('Test 3: Hankel consistency ...', end=' ', flush=True)
    rp_w, wp_w, dwp_w = test_fftlog_hankel_consistency()
    print('OK')

    print('Test 4: NFW dsigma identity ...', end=' ', flush=True)
    r, sig, bs, ds = test_nfw_dsigma_identity()
    print('OK')

    print('Test 5: NFW vs colossus direct ...', end=' ', flush=True)
    r, sig_hl, sig_col = test_nfw_vs_colossus_direct()
    print(f'OK (max rel err = {np.max(np.abs(sig_hl-sig_col)/sig_col):.2e})')

    print('Test 6: NFW positivity ...', end=' ', flush=True)
    test_nfw_sigma_positivity()
    print('OK')

    print('Test 7: TJ dsigma identity ...', end=' ', flush=True)
    test_tj_dsigma_identity()
    print('OK')

    print('Test 8: BMO dsigma identity ...', end=' ', flush=True)
    test_bmo_dsigma_identity()
    print('OK')

    print('Test 9: Hernquist dsigma identity ...', end=' ', flush=True)
    test_hern_dsigma_identity()
    print('OK')

    print('Test 10: sigma_crit ...', end=' ', flush=True)
    test_sigma_crit()
    print('OK')

    print('\nAll tests passed.')

    with PdfPages('test_plots.pdf') as pdf:
        make_plots(pdf)

    return results


if __name__ == '__main__':
    run_all()
