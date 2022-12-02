import os

import astropy.units as u
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from reproject import reproject_interp

PIXEL_SCALE_NAMES = ['XPIXSIZE', 'CDELT1', 'CD1_1']


def get_pix_size(hdu):
    """Get pixel scale from header.

    Checks HDU header for pixel scale keywords, and returns a pixel scale in arcsec. If no suitable keyword is found,
    will throw up an error.

    Args:
        hdu (astropy.fits.PrimaryHDU): HDU to get pixel scale for.

    Returns:
        pix_scale (float): Pixel scale in arcsec.

    Raises:
        Warning: If no suitable pixel scale keyword is found in header.

    """
    for pixel_keyword in PIXEL_SCALE_NAMES:
        try:
            try:
                pix_scale = np.abs(float(hdu.header[pixel_keyword]))
            except ValueError:
                continue
            if pixel_keyword in ['CDELT1', 'CD1_1']:
                pix_scale *= 3600
            return pix_scale
        except KeyError:
            pass

    raise Warning('No pixel scale found')


def project(x, y, pa, inc):
    """General rotation/projection routine.

    Given coordinates (x, y), will rotate and project given position angle (counter-clockwise from N), and
    inclination. Assumes centre is at (0, 0).

    Args:
        x (float or numpy.ndarray): x-coordinate(s)
        y (float or numpy.ndarray): y-coordinates(s)
        pa (float): Position angle (degrees)
        inc (float): Inclination (degrees)

    Returns:
        x_proj, y_proj: The rotated, projected (x, y) coordinates.

    """

    angle = np.radians(pa)
    cos_a, sin_a = np.cos(angle), np.sin(angle)

    x_proj = x * cos_a + y * sin_a
    y_proj = - x * sin_a + y * cos_a

    # Account for inclination

    x_proj /= np.cos(np.radians(inc))

    return x_proj, y_proj


def normalise(array, percentile=99):

    norm_val = np.nanpercentile(array, percentile)

    return array / norm_val, norm_val


def rolling_average(x_data, y_data, window=50, percentiles=None):
    if percentiles is None:
        percentiles = [16, 50, 84]
    if type(percentiles) != list:
        percentiles = [percentiles]

    # Filter NaNs
    idx = np.where((~np.isnan(x_data)) & (~np.isnan(y_data)))
    x_data, y_data = x_data[idx], y_data[idx]

    # Sort
    idx = np.argsort(x_data)
    x_data, y_data = x_data[idx], y_data[idx]

    rolling_y = np.zeros([len(x_data), len(percentiles)])

    for i in range(len(x_data)):
        start_idx = max(i - int(window / 2), 0)
        stop_idx = min(i + int(window / 2), len(x_data))
        rolling_y[i, :] = np.percentile(y_data[start_idx:stop_idx], percentiles)

    return x_data, rolling_y


matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['font.size'] = 14

os.chdir('/Users/thomaswilliams/Documents/phangs/jwst_early_science')

# M74/NGC0628 coordinate and orientation info (from PHANGS sample table v1.6)
center_coord = (24.173855 * u.deg, 15.783643 * u.deg)
incl = 8.9
pa = 20.7
dist = 9.84

env = fits.open('data/NGC0628_simple.fits')
env_hdu = env[0]

jwst = fits.open('data/ngc0628_miri_f2100w_anchored.fits')
jwst_hdu = jwst[0]

ha = fits.open('data/NGC0628_sfr.fits')
ha_hdu = ha['SFR_HA']

co = fits.open('data/ngc0628_12m+7m+tp_co21_strict_mom0.fits')
co_hdu = co[0]

# Make SFE HDU
ha_reproj = reproject_interp(ha_hdu, co_hdu.header,
                             return_footprint=False)
sfe = ha_reproj / (co_hdu.data * 6.25 * 1e3)

# plt.figure()
# plt.scatter(co_hdu.data * 6.25, ha_reproj)
# plt.xscale('log')
# plt.yscale('log')
# plt.show()

# vmin, vmax = np.nanpercentile(sfe, [1, 99])
# plt.figure()
# plt.imshow(sfe, origin='lower', vmin=vmin, vmax=vmax)
# plt.show()

sfe_hdu = fits.PrimaryHDU(sfe, header=co_hdu.header)

spurs = {'co_poor': {'filename': 'data/spur1.fits', 'threshold': 0.2},
         'co_rich': {'filename': 'data/spur2.fits', 'threshold': 0.5},
         }

hdu_names = {jwst_hdu: 'jwst',
             ha_hdu: 'ha',
             co_hdu: 'co',
             sfe_hdu: 'sfe'}

colours = {'jwst': 'r',
           'ha': 'g',
           'co': 'b',
           'sfe': 'darkorange'
           }

labels = {'jwst': r'21$\mu$m',
          'ha': r'H$\alpha$ or SFR',
          'co': 'CO',
          'sfe': 'SFE'
          }

r_dict = {}
flux_dict = {}
env_reprojs = {}

# We'll base the coords on the JWST file

wcs = WCS(jwst_hdu)
pix_size = get_pix_size(jwst_hdu)

x_cen, y_cen = wcs.all_world2pix(center_coord[0], center_coord[1], 1)

# Pull out pixel coordinates
ii, jj = np.meshgrid(np.arange(jwst_hdu.data.shape[1]),
                     np.arange(jwst_hdu.data.shape[0]))

# Calculate projected x/y coordinates

x_region = ii - x_cen
y_region = jj - y_cen

x_region *= pix_size / 3600 * np.pi / 180 * dist * 1e3
y_region *= pix_size / 3600 * np.pi / 180 * dist * 1e3

# Put x_region_pix and y_region_pix into the frame of the original input HDU.

x_region_phys, y_region_phys = project(x_region, y_region, pa, incl)
r_phys = np.sqrt(x_region_phys ** 2 + y_region_phys ** 2)

for key in spurs.keys():

    spur = fits.open(spurs[key]['filename'])
    spur_hdu = spur[0]
    spur_mask = np.array(spur_hdu.data > spurs[key]['threshold'], dtype=int)

    env_reproj = reproject_interp(env_hdu,
                                  spur_hdu.header,
                                  return_footprint=False,
                                  order='nearest-neighbor')
    env_reproj[env_reproj != 6] = np.nan
    env_reproj[env_reproj == 6] = 1
    env_reproj[spur_mask == 0] = np.nan

    r_phys_reproj = reproject_interp((r_phys, jwst_hdu.header),
                                     spur_hdu.header,
                                     return_footprint=False)
    r_phys_vals = r_phys_reproj[env_reproj == 1]
    env_reprojs[key] = [np.nanmin(r_phys_vals), np.nanmax(r_phys_vals)]

    for data_hdu in [jwst_hdu, ha_hdu, co_hdu, sfe_hdu]:

        key_out = '%s_%s' % (key, hdu_names[data_hdu])

        filename_out = 'data/%s.txt' % key_out

        if not os.path.exists(filename_out):

            spur_mask_reproj = reproject_interp((spur_mask, spur_hdu.header),
                                                data_hdu.header,
                                                order='nearest-neighbor',
                                                return_footprint=False
                                                )
            spur_mask_reproj[np.isnan(spur_mask_reproj)] = 0
            spur_mask_reproj = spur_mask_reproj.astype(bool)

            r_phys_reproj = reproject_interp((r_phys, jwst_hdu.header),
                                             data_hdu.header,
                                             return_footprint=False)

            flux = data_hdu.data[spur_mask_reproj]
            r = r_phys_reproj[spur_mask_reproj]

            non_nans = ~np.isnan(flux) & ~np.isnan(r) & np.isfinite(flux)

            np.savetxt(filename_out, np.c_[r[non_nans], flux[non_nans]])

        else:

            r, flux = np.loadtxt(filename_out, unpack=True)

        r_dict[key_out] = r
        flux_dict[key_out] = flux

    spur.close()

# Plot everything up

plot_name = os.path.join('plots', 'radial_sfr_sfe')

plt.figure(figsize=(8, 4))

for i, spur_type in enumerate(['co_rich', 'co_poor']):

    print(spur_type)

    ax = plt.subplot(1, 2, i + 1)

    plt.axvspan(env_reprojs[spur_type][0], env_reprojs[spur_type][1], facecolor='grey', alpha=0.5)

    for j, flux_type in enumerate(['jwst', 'ha', 'co', 'sfe']):
        c = colours[flux_type]
        label = labels[flux_type]

        key = '%s_%s' % (spur_type, flux_type)

        r = r_dict[key]
        flux = flux_dict[key]
        flux[flux == 0] = np.nan
        flux, norm_value = normalise(flux, percentile=50)

        print(flux_type, norm_value)

        rolling_r, rolling_flux = rolling_average(r, flux, window=int(len(r) * 0.01))

        rolling_flux = np.log10(rolling_flux)

        offset = j + 1

        # plt.scatter(r, flux,
        #             marker='.', c=c, alpha=0.2, rasterized=True)
        plt.plot(rolling_r, rolling_flux[:, 1] + offset, c=c, alpha=0.5, label=label)
        plt.fill_between(rolling_r, rolling_flux[:, 2] + offset, rolling_flux[:, 0] + offset, color=c, alpha=0.15)

    # plt.yscale('log')

    plt.xlabel('$r$ (kpc)')
    plt.ylabel(r'$\log_{10}$(Normalized Value) + Offset')

    if i == 0:
        # plt.legend(loc='upper left', fancybox=False, edgecolor='k', framealpha=1)
        plt.text(0.95, 0.95, 'CO-rich',
                 ha='right', va='top',
                 transform=ax.transAxes,
                 bbox=dict(facecolor='white', edgecolor='k'))

    if i == 1:
        handles, labels = ax.get_legend_handles_labels()
        # ax.legend(handles[::-1], labels[::-1], title='Line', loc='upper left')
        ax.legend(handles[::-1], labels[::-1],
                  loc='center left', bbox_to_anchor=(1.15, 0.5),
                  fancybox=False, edgecolor='k', framealpha=1)
        plt.text(0.05, 0.95, 'CO-poor',
                 ha='left', va='top',
                 transform=ax.transAxes,
                 bbox=dict(facecolor='white', edgecolor='k'))

        ax.yaxis.set_label_position('right')
        ax.yaxis.tick_right()

    plt.grid()

plt.subplots_adjust(hspace=0, wspace=0)

# plt.show()
plt.savefig(plot_name + '.pdf', bbox_inches='tight')
plt.savefig(plot_name + '.png', bbox_inches='tight')

env.close()
jwst.close()
ha.close()
co.close()

print('Complete!')
