# halo_lensing

Weak lensing profiles for NFW, Takada-Jain (TJ), Baltz-Marshall-Oguri (BMO), and Hernquist halos, plus a 2-halo term. Includes FFT-based off-centering via a bundled FFTLog implementation.

Uses [colossus](https://bitbucket.org/bdiemer/colossus) for cosmology, distances, power spectra, and NFW projected profiles.

If you use this code, please cite:

- [M. Oguri et al., PASJ, 78, 416 (2026)](https://ui.adsabs.harvard.edu/abs/2026PASJ...78..416O)

---

## Install

```bash
pip install colossus astropy scipy numpy
```

---

## Quick start

```python
import numpy as np
from colossus.cosmology import cosmology
import halo_lensing as hl

cosmo = cosmology.setCosmology('planck18')

r = np.logspace(-1, 1, 50)   # projected radius [h^-1 Mpc]
m, z = 1e14, 0.3             # halo mass [h^-1 M_sun], redshift
c = hl.concent_m(m, z, cosmo)

sigma  = hl.nfw_sigma(r, m, c, z, cosmo)   # surface mass density
dsigma = hl.nfw_dsigma(r, m, c, z, cosmo)  # excess surface density ΔΣ
```

---

## API reference

All length units are **h⁻¹ Mpc** (physical unless `comoving=True`).  
All mass units are **h⁻¹ M☉**.  
Surface density units are **h M☉ Mpc⁻²**.

### Cosmology helpers

| Function | Returns |
|---|---|
| `sigma_crit(z, zs, cosmo)` | critical surface density Σ_crit [h M☉ Mpc⁻²] |
| `inv_sigma_crit(z, zs, cosmo)` | 1/Σ_crit; 0 if z ≥ zs |
| `concent_m(m, z, cosmo)` | NFW concentration (Diemer+19 model) |
| `calc_rvir(m, z, cosmo)` | virial radius |

### NFW profile (via colossus)

| Function | Description |
|---|---|
| `nfw_sigma(r, m, c, z, cosmo)` | projected surface density Σ |
| `nfw_bsigma(r, m, c, z, cosmo)` | mean surface density within r |
| `nfw_dsigma(r, m, c, z, cosmo)` | excess surface density ΔΣ = b̄Σ − Σ |
| `nfw_kappa / nfw_kappa_ave / nfw_gamma` | convergence / mean convergence / shear |

All accept `mdef='vir'` and `comoving=False`.

### Takada-Jain (sharply truncated NFW)

Same signature as NFW, plus the concentration `c` sets the truncation radius.

```python
sig  = hl.tj_sigma(r, m, c, z, cosmo)
dsig = hl.tj_dsigma(r, m, c, z, cosmo)
```

Fourier-space form (needed for off-centering):

```python
k = np.logspace(-3, 3, 256)
uk = hl.tj_sigma_f(k, m, c, z, cosmo)   # Σ̃(k) = m · y_TJ(k·rs, c)
```

### Baltz-Marshall-Oguri (smoothly truncated NFW)

Extra parameter `tv`: truncation scale in units of r_vir (`tv=2.5` is typical).

```python
sig  = hl.bmo_sigma(r, m, c, tv, z, cosmo)
dsig = hl.bmo_dsigma(r, m, c, tv, z, cosmo)
uk   = hl.bmo_sigma_f(k, m, c, tv, z, cosmo)
```

### Hernquist profile

No colossus dependency; takes effective radius `re` [h⁻¹ Mpc] directly.

```python
sig  = hl.hern_sigma(r, m, re, z)
dsig = hl.hern_dsigma(r, m, re, z)
```

### 2-halo term

```python
k   = np.logspace(-3, 2, 256)
pk2 = hl.sigma_2h_f(k, m, z, cosmo)          # bias × ρ_m × D²(z) × P(k)
pk2_nob = hl.sigma_2h_nob_f(k, z, cosmo)     # same without halo bias
```

### Off-centering (Gaussian mis-centering PDF)

```python
# Total signal = centered 1-halo + Gaussian off-centered 1-halo + 2-halo
# flag_d: 0 = Σ,  1 = ΔΣ
# flag_out: 0 = total,  1 = centered 1h,  2 = 2h,  3 = off-centered 1h
sig_total = hl.sigma_off(r, m, z, f_cen=0.7, sig_off=0.3,
                         flag_d=0, flag_out=0, cosmo=cosmo, mdef='200m')
```

### Off-centering (fixed offset, delta-function PDF)

```python
sig = hl.bmo_sigma_fixroff_fft(r, m, c, tv, z, roff=0.4, flag_d=0, cosmo=cosmo)
```

---

## FFTLog

`fftlog.py` implements the FFTLog algorithm for integrals with spherical and
cylindrical Bessel functions. Top-level convenience functions:

```python
import fftlog

k  = np.logspace(-3, 2, 512)
pk = cosmo.matterPowerSpectrum(k)

r,  xi  = fftlog.pk2xi(k, pk)    # P(k) → ξ(r)
k2, pk2 = fftlog.xi2pk(r, xi)    # ξ(r) → P(k)
rp, wp  = fftlog.pk2wp(k, pk)    # P(k) → w_p(r_p)  (Hankel J_0)
rp, dwp = fftlog.pk2dwp(k, pk)   # P(k) → Δw_p(r_p) (Hankel J_2)
```

Key parameters shared by all functions:

| Parameter | Default | Effect |
|---|---|---|
| `nu` | 1.01 | power-law bias; avoid half-integers |
| `N_extrap_low` | 0 | points to extrapolate at low x |
| `N_extrap_high` | 0 | points to extrapolate at high x |
| `c_window_width` | 0.25 | fraction of modes to taper |
| `N_pad` | 0 | zero-padding points each side |
| `kr` | 1 | sets the x·y = kr pivot |

---

## Full example: NFW + 2-halo + off-centering

```python
import numpy as np
import matplotlib.pyplot as plt
from colossus.cosmology import cosmology
import halo_lensing as hl

cosmo = cosmology.setCosmology('planck18')

r    = np.logspace(-1, 1.3, 60)
m, z = 5e13, 0.25
mdef = '200m'
f_cen, sig_off = 0.7, 0.2

total  = hl.sigma_off(r, m, z, f_cen, sig_off, flag_d=1, flag_out=0, cosmo=cosmo, mdef=mdef)
cen_1h = hl.sigma_off(r, m, z, f_cen, sig_off, flag_d=1, flag_out=1, cosmo=cosmo, mdef=mdef)
twoh   = hl.sigma_off(r, m, z, f_cen, sig_off, flag_d=1, flag_out=2, cosmo=cosmo, mdef=mdef)
off_1h = hl.sigma_off(r, m, z, f_cen, sig_off, flag_d=1, flag_out=3, cosmo=cosmo, mdef=mdef)

plt.loglog(r, total,  label='total')
plt.loglog(r, cen_1h, label='1h centered', ls='--')
plt.loglog(r, twoh,   label='2h', ls=':')
plt.loglog(r, off_1h, label='1h off-centered', ls='-.')
plt.xlabel(r'$r_p\ [h^{-1}\,\mathrm{Mpc}]$')
plt.ylabel(r'$\Delta\Sigma\ [h\,M_\odot\,\mathrm{Mpc}^{-2}]$')
plt.legend()
plt.show()
```
