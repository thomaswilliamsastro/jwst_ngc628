import copy
import warnings
from pathlib import Path
import os

import numpy as np
import scipy.stats as stats
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.colors as mcolors
from astropy import units as u
from astropy.io import fits
from astropy.wcs import WCS
from astropy.table import QTable
from astropy.nddata import Cutout2D
from astropy.coordinates import SkyCoord
from skimage.transform import warp_polar

from reproject import reproject_interp


def deproject(
        center_coord=None, incl=0 * u.deg, pa=0 * u.deg,
        header=None, wcs=None, naxis=None, ra=None, dec=None,
        return_offset=False):
    """
    Calculate deprojected radii and projected angles in a disk.
    This function deals with projected images of astronomical objects
    with an intrinsic disk geometry. Given sky coordinates of the
    disk center, disk inclination and position angle, this function
    calculates deprojected radii and projected angles based on
    (1) a FITS header (`header`), or
    (2) a WCS object with specified axis sizes (`wcs` + `naxis`), or
    (3) RA and DEC coodinates (`ra` + `dec`).
    Both deprojected radii and projected angles are defined relative
    to the center in the inclined disk frame. For (1) and (2), the
    outputs are 2D images; for (3), the outputs are arrays with shapes
    matching the broadcasted shape of `ra` and `dec`.
    Parameters
    ----------
    center_coord : `~astropy.coordinates.SkyCoord` object or 2-tuple
        Sky coordinates of the disk center
    incl : `~astropy.units.Quantity` object or number, optional
        Inclination angle of the disk (0 degree means face-on)
        Default is 0 degree.
    pa : `~astropy.units.Quantity` object or number, optional
        Position angle of the disk (red/receding side, North->East)
        Default is 0 degree.
    header : `~astropy.io.fits.Header` object, optional
        FITS header specifying the WCS and size of the output 2D maps
    wcs : `~astropy.wcs.WCS` object, optional
        WCS of the output 2D maps
    naxis : array-like (with two elements), optional
        Size of the output 2D maps
    ra : array-like, optional
        RA coordinate of the sky locations of interest
    dec : array-like, optional
        DEC coordinate of the sky locations of interest
    return_offset : bool, optional
        Whether to return the angular offset coordinates together with
        deprojected radii and angles. Default is to not return.
    Returns
    -------
    deprojected coordinates : list of arrays
        If `return_offset` is set to True, the returned arrays include
        deprojected radii, projected angles, as well as angular offset
        coordinates along East-West and North-South direction;
        otherwise only the former two arrays will be returned.
    Notes
    -----
    This is the Python version of an IDL function `deproject` included
    in the `cpropstoo` package. See URL below:
    https://github.com/akleroy/cpropstoo/blob/master/cubes/deproject.pro
    """

    if isinstance(center_coord, SkyCoord):
        x0_deg = center_coord.ra.degree
        y0_deg = center_coord.dec.degree
    else:
        x0_deg, y0_deg = center_coord
        if hasattr(x0_deg, 'unit'):
            x0_deg = x0_deg.to(u.deg).value
            y0_deg = y0_deg.to(u.deg).value
    if hasattr(incl, 'unit'):
        incl_deg = incl.to(u.deg).value
    else:
        incl_deg = incl
    if hasattr(pa, 'unit'):
        pa_deg = pa.to(u.deg).value
    else:
        pa_deg = pa

    if header is not None:
        wcs_cel = WCS(header).celestial
        naxis1 = header['NAXIS1']
        naxis2 = header['NAXIS2']
        # create ra and dec grids
        ix = np.arange(naxis1)
        iy = np.arange(naxis2).reshape(-1, 1)
        ra_deg, dec_deg = wcs_cel.wcs_pix2world(ix, iy, 0)
    elif (wcs is not None) and (naxis is not None):
        wcs_cel = wcs.celestial
        naxis1, naxis2 = naxis
        # create ra and dec grids
        ix = np.arange(naxis1)
        iy = np.arange(naxis2).reshape(-1, 1)
        ra_deg, dec_deg = wcs_cel.wcs_pix2world(ix, iy, 0)
    else:
        ra_deg, dec_deg = np.broadcast_arrays(ra, dec)
        if hasattr(ra_deg, 'unit'):
            ra_deg = ra_deg.to(u.deg).value
            dec_deg = dec_deg.to(u.deg).value

    # recast the ra and dec arrays in terms of the center coordinates
    # arrays are now in degrees from the center
    dx_deg = (ra_deg - x0_deg) * np.cos(np.deg2rad(y0_deg))
    dy_deg = dec_deg - y0_deg

    # rotation angle (rotate x-axis up to the major axis)
    rotangle = np.pi / 2 - np.deg2rad(pa_deg)

    # create deprojected coordinate grids
    deprojdx_deg = (dx_deg * np.cos(rotangle) +
                    dy_deg * np.sin(rotangle))
    deprojdy_deg = (dy_deg * np.cos(rotangle) -
                    dx_deg * np.sin(rotangle))
    deprojdy_deg /= np.cos(np.deg2rad(incl_deg))

    # make map of deprojected distance from the center
    radius_deg = np.sqrt(deprojdx_deg ** 2 + deprojdy_deg ** 2)

    # make map of angle w.r.t. position angle
    projang_deg = np.rad2deg(np.arctan2(deprojdy_deg, deprojdx_deg))

    if return_offset:
        return radius_deg, projang_deg, dx_deg, dy_deg
    else:
        return radius_deg, projang_deg


def show_polar_plot(data, ax, cmap='viridis'):
    vmin, vmax = np.percentile(data[np.isfinite(data)], [1, 99])

    current_cmap = matplotlib.cm.get_cmap(cmap).copy()
    current_cmap.set_bad(color='gray')

    plt.imshow(
        data,
        cmap=current_cmap,
        origin='lower', aspect='auto',
        extent=[-180, 180, 1, 6],
        norm=mcolors.PowerNorm(0.5, vmin=vmin, vmax=vmax))
    # plt.colorbar(pad=0, label=cbar_label)
    # plt.xticks([-180, 0, 180], ['', '', ''])
    plt.ylabel(r'$r$ (kpc)')
    plt.yticks([2, 4, 6], ['2', '4', '6'])

    ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())

    # plt.grid()


matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['font.size'] = 14

os.chdir('/Users/williams/Documents/phangs/jwst_early_science')

data_dir = 'data'
plot_dir = 'plots'

if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

# M74/NGC0628 coordinate and orientation info (from PHANGS sample table v1.6)
center_coord = (24.173855 * u.deg, 15.783643 * u.deg)
incl = 8.9 * u.deg
pa = 20.7 * u.deg
dist = 9.84 * u.Mpc

jwst_file_name = os.path.join('data',
                              'ngc0628_miri_lv3_f2100w_i2d_align.fits',
                              )
alma_file_name = os.path.join('data', 'ngc0628_12m+7m+tp_co21_strict_mom0.fits')
muse_file_name = os.path.join('data',
                              'muse_ha.fits',
                              # 'ngc0628_mips24_image_v5-0.fits'
                              )

environment_file_name = os.path.join('data', 'NGC0628_simple.fits')

spur1_file_name = os.path.join('data',
                               'spur1.fits',
                               )
spur2_file_name = os.path.join('data',
                               'spur2.fits',
                               )

jwst_hdu = fits.open(jwst_file_name)
jwst_data = jwst_hdu['SCI'].data
jwst_hdr = jwst_hdu['SCI'].header

jwst_hdu.close()

alma_hdu = fits.open(alma_file_name)
muse_hdu = fits.open(muse_file_name)
spur1_hdu = fits.open(spur1_file_name)
spur2_hdu = fits.open(spur2_file_name)
environment_hdu = fits.open(environment_file_name)

co_data = reproject_interp(alma_hdu[0],
                           jwst_hdr,
                           return_footprint=False)
ha_data = reproject_interp(muse_hdu[1],
                           jwst_hdr,
                           return_footprint=False)

env_data = reproject_interp(environment_hdu[0],
                            jwst_hdr,
                            return_footprint=False,
                            order='nearest-neighbor')

spur1_reproj = reproject_interp(spur1_hdu[0],
                                jwst_hdr,
                                return_footprint=False)

spur2_reproj = reproject_interp(spur2_hdu[0],
                                jwst_hdr,
                                return_footprint=False)

alma_hdu.close()
muse_hdu.close()
environment_hdu.close()
spur1_hdu.close()
spur2_hdu.close()

co_data_masked = copy.deepcopy(co_data)
co_data_masked[env_data == 1] = np.nan

ha_data_masked = copy.deepcopy(ha_data)
ha_data_masked[env_data == 1] = np.nan

jwst_data_masked = copy.deepcopy(jwst_data)
jwst_data_masked[env_data == 1] = np.nan

jwst_data_masked[jwst_data_masked == 0] = np.nan

nan_mask = np.where((np.isnan(co_data_masked)) | (np.isnan(ha_data_masked)) | np.isnan(jwst_data_masked))

# co_data_masked[nan_mask] = np.nan
ha_data_masked[nan_mask] = np.nan
jwst_data_masked[nan_mask] = np.nan

r_deg, phi_deg = deproject(
    center_coord=center_coord, incl=incl, pa=pa,
    header=jwst_hdr)
r_kpc = (np.deg2rad(r_deg) * dist).to('kpc').value

r_kpc_lim = (1, 6)
r_kpc_levels = (3, 6)

r_kpc_bins = np.linspace(
    r_kpc_lim[0], r_kpc_lim[1],
    (r_kpc_lim[1] - r_kpc_lim[0]) * 50 + 1)
r_kpc_mid = (r_kpc_bins[:-1] + r_kpc_bins[1:]) / 2
r_kpc_centres = r_kpc_bins + np.diff(r_kpc_bins)[0] / 2
r_kpc_centres = r_kpc_centres[:-1]

phi_deg_bins = np.linspace(-180, 180, 181)
phi_deg_centres = phi_deg_bins + np.diff(phi_deg_bins)[0] / 2
phi_deg_centres = phi_deg_centres[:-1]

mask = np.isfinite(co_data_masked)
co_data_polar, _, _, _ = stats.binned_statistic_2d(
    r_kpc[mask], phi_deg[mask], co_data_masked[mask],
    statistic=np.nanmean, bins=(r_kpc_bins, phi_deg_bins))
co_data_polar_medsub = \
    co_data_polar - np.nanmedian(co_data_polar, axis=1, keepdims=True)

mask = np.isfinite(jwst_data_masked)
jwst_data_polar, _, _, _ = stats.binned_statistic_2d(
    r_kpc[mask], phi_deg[mask], jwst_data_masked[mask],
    statistic=np.nanmean, bins=(r_kpc_bins, phi_deg_bins))
jwst_data_polar_medsub = \
    jwst_data_polar - np.nanmedian(jwst_data_polar, axis=1, keepdims=True)

mask = np.isfinite(ha_data_masked)
ha_data_polar, _, _, _ = stats.binned_statistic_2d(
    r_kpc[mask], phi_deg[mask], ha_data_masked[mask],
    statistic=np.nanmean, bins=(r_kpc_bins, phi_deg_bins))
ha_data_polar_medsub = \
    ha_data_polar - np.nanmedian(ha_data_polar, axis=1, keepdims=True)

mask = np.isfinite(spur1_reproj)
spur1_polar, _, _, _ = stats.binned_statistic_2d(
    r_kpc[mask], phi_deg[mask], spur1_reproj[mask],
    statistic=np.nanmean, bins=(r_kpc_bins, phi_deg_bins))
mask = np.isfinite(spur2_reproj)
spur2_polar, _, _, _ = stats.binned_statistic_2d(
    r_kpc[mask], phi_deg[mask], spur2_reproj[mask],
    statistic=np.nanmean, bins=(r_kpc_bins, phi_deg_bins))

spur1_mask = np.array(spur1_polar > 0.2, dtype=int)
spur2_mask = np.array(spur2_polar > 0.5, dtype=int)

ii, jj = np.meshgrid(phi_deg_centres,
                     r_kpc_centres)

plot_name = os.path.join(plot_dir, 'polar_unwrap')

plt.figure(figsize=(6.5, 10))

ax = plt.subplot(3, 1, 1)
show_polar_plot(co_data_polar_medsub, ax)

plt.text(0.95, 0.95,
         r'ALMA CO',
         ha='right', va='top',
         bbox=dict(facecolor='white', edgecolor='black', alpha=1),
         transform=ax.transAxes,
         )

# Plot on the spiral arms
plt.plot([-180, 180], [5.6, 1], c='r')
plt.plot([-180, 0], [2.8, 1], c='r')

# plt.arrow(63, 3.5, -6, 0,
#           width=0.1,
#           head_length=5,
#           color='red',
#           )
plt.contour(ii, jj, spur1_mask,
            levels=1,
            colors='orange',
            # lw=1,
            )

plt.contour(ii, jj, spur2_mask,
            levels=1,
            colors='cyan',
            # lw=1,
            )

ax = plt.subplot(3, 1, 2)
show_polar_plot(jwst_data_polar_medsub, ax)
# plt.contour(ii, jj,
#             co_data_polar_medsub,
#             colors='white',
#             linewidths=1,
#             # linestyles='--',
#             levels=[0.1, 10],  # 3,
#             )

plt.plot([-180, 180], [5.6, 1], c='r')
plt.plot([-180, 0], [2.8, 1], c='r')

# plt.arrow(83, 3.5, -6, 0,
#           width=0.1,
#           head_length=5,
#           color='red',
#           )
plt.contour(ii, jj, spur1_mask,
            levels=1,
            colors='orange',
            # lw=1,
            )

plt.contour(ii, jj, spur2_mask,
            levels=1,
            colors='cyan',
            # lw=1,
            )

plt.text(0.95, 0.95,
         r'JWST 21$\mu$m',
         ha='right', va='top',
         bbox=dict(facecolor='white', edgecolor='black', alpha=1),
         transform=ax.transAxes,
         )

ax = plt.subplot(3, 1, 3)
show_polar_plot(ha_data_polar_medsub, ax)
# plt.contour(ii, jj,
#             co_data_polar_medsub,
#             colors='white',
#             linewidths=1,
#             # linestyles='--',
#             levels=[0.1, 10],  # 3,
#             )

plt.plot([-180, 180], [5.6, 1], c='r')
plt.plot([-180, 0], [2.8, 1], c='r')

# plt.arrow(83, 3.5, -6, 0,
#           width=0.1,
#           head_length=5,
#           color='red',
#           )
plt.contour(ii, jj, spur1_mask,
            levels=1,
            colors='orange',
            # lw=1,
            )

plt.contour(ii, jj, spur2_mask,
            levels=1,
            colors='cyan',
            # lw=1,
            )

plt.text(0.95, 0.95,
         r'MUSE H$\alpha$',
         ha='right', va='top',
         bbox=dict(facecolor='white', edgecolor='black', alpha=1),
         transform=ax.transAxes,
         )

plt.xlabel(r'$\theta$ (deg)')
plt.xticks([-180, 0, 180], ['+/-180', '0', '+/-180'])

plt.subplots_adjust(hspace=0, wspace=0)

# plt.show()

plt.savefig(plot_name + '.png', bbox_inches='tight')
plt.savefig(plot_name + '.pdf', bbox_inches='tight')
plt.close()

print('Complete!')
