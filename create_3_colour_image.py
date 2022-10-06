import os

import astropy.units as u
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from reproject import reproject_interp

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['font.size'] = 14

os.chdir('/Users/williams/Documents/phangs/jwst_early_science')

high_pass_jwst = True

data_dir = 'data'
plot_dir = 'plots'

pmin, pmax = 1, 99

# hdu = fits.open('/Users/williams/Documents/phangs/muse/DR2.1/NGC0628_MAPS.fits')
# hdu['HA6562_FLUX'].writeto(os.path.join(data_dir, 'muse_ha.fits'), overwrite=True)

if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

jwst_file_name = os.path.join('data',
                              'ngc0628_miri_f2100w_anchored.fits',
                              )
alma_file_name = os.path.join('data', 'ngc0628_12m+7m+tp_co21_strict_mom0.fits')
muse_file_name = os.path.join('data',
                              'muse_ha.fits',
                              # 'ngc0628_mips24_image_v5-0.fits'
                              )

environment_file_name = os.path.join('data', 'NGC0628_simple.fits')

centre = SkyCoord(24.173855, 15.783643, unit=(u.deg, u.deg))
pa = 20.
size = (0.04 * u.deg, 0.08 * u.deg)
# size = (0.5 * u.deg, 0.5 * u.deg)

w = WCS(jwst_file_name)
x_cen, y_cen = w.all_world2pix(centre.ra, centre.dec, 1)

# Rotate (-5000, 0), (5000, 0) by the PA to plot on later
x_pa = np.array([-5000, 5000], dtype=float)
y_pa = np.array([np.abs(x_pa[0] / np.tan(np.radians(pa))), - x_pa[1] / np.tan(np.radians(pa))])

x_pa += x_cen
y_pa += y_cen

# We'll regrid everything to the JWST data
jwst_hdu = fits.open(jwst_file_name)
jwst_sci = jwst_hdu[0]
jwst_sci.data[jwst_sci.data == 0] = np.nan
jwst_w = WCS(jwst_sci)

alma_hdu = fits.open(alma_file_name)
muse_hdu = fits.open(muse_file_name)

environment_hdu = fits.open(environment_file_name)

alma_reproj = reproject_interp(alma_hdu,
                               jwst_sci.header,
                               return_footprint=False)

muse_reproj = reproject_interp(muse_hdu[1],
                               jwst_sci.header,
                               return_footprint=False)

env_reproj = reproject_interp(environment_hdu[0],
                              jwst_sci.header,
                              return_footprint=False,
                              order='nearest-neighbor')
env_reproj[env_reproj != 6] = np.nan
env_reproj[env_reproj == 6] = 1

ii, jj = np.meshgrid(np.arange(env_reproj.shape[1]), np.arange(env_reproj.shape[0]))

# Create some masks to show observation extent
jwst_fov = ~np.isnan(jwst_sci.data)
muse_fov = ~np.isnan(muse_reproj)
alma_fov = ~np.isnan(alma_reproj)

vmin_jwst, vmax_jwst = 0, 7
jwst_rescale = jwst_sci.data - vmin_jwst
jwst_rescale /= (vmax_jwst - vmin_jwst)

vmin_muse, vmax_muse = 0, 5000
muse_reproj -= vmin_muse
muse_reproj /= (vmax_muse - vmin_muse)

vmin_alma, vmax_alma = 0, 6

alma_reproj -= vmin_alma
alma_reproj /= (vmax_alma - vmin_alma)

# nan_idx = np.where((np.isnan(jwst_rescale)) | (np.isnan(alma_reproj)) | (np.isnan(muse_reproj)))

# jwst_rescale[nan_idx] = np.nan
# muse_reproj[nan_idx] = np.nan
# alma_reproj[nan_idx] = np.nan

rgb = np.dstack((jwst_rescale,
                 muse_reproj,
                 alma_reproj))

plot_name = os.path.join(plot_dir, 'jwst_muse_alma')

plt.figure(figsize=(10, 10))
ax = plt.subplot(projection=jwst_w)
plt.imshow(rgb,
           origin='lower')

xlim = plt.xlim()
ylim = plt.ylim()

plt.contourf(ii, jj, env_reproj,
             levels=1,
             colors='white',
             alpha=0.2,
             )

plt.contour(ii, jj, jwst_fov,
            levels=1,
            colors='red',
            alpha=0.5,
            )
plt.contour(ii, jj, muse_fov,
            levels=1,
            colors='green',
            alpha=0.5,
            )
plt.contour(ii, jj, alma_fov,
            levels=1,
            colors='blue',
            alpha=0.5,
            )

# Plot on the position angle line
plt.plot(x_pa, y_pa, c='white', lw=2)

plt.xlim(xlim)
plt.ylim(ylim)

# TODO Plot on the pointers to the spurs

lon = ax.coords[0]
lat = ax.coords[1]

lon.display_minor_ticks(True)
lat.display_minor_ticks(True)

plt.grid(zorder=99, alpha=0.5)

# plt.text(0.05, 0.95,
#          r'\textcolor{red}{MIRI 21$\mu$m}' + '\n' +
#          r'\textcolor{green}{MUSE H$\alpha$}' + '\n' +
#          r'\textcolor{blue}{ALMA CO}',
#          ha='left', va='top',
#          bbox=dict(facecolor='black', edgecolor='white', alpha=1),
#          transform=ax.transAxes)

plt.text(0.05, 0.95,
         r'MIRI 21$\mu$m',
         c='r',
         ha='left', va='top',
         bbox=dict(facecolor='black', edgecolor='black', alpha=1),
         transform=ax.transAxes)
plt.text(0.05, 0.92,
         r'MUSE H$\alpha$',
         c='g',
         ha='left', va='top',
         bbox=dict(facecolor='black', edgecolor='black', alpha=1),
         transform=ax.transAxes)
plt.text(0.05, 0.89,
         r'ALMA CO',
         c='b',
         ha='left', va='top',
         bbox=dict(facecolor='black', edgecolor='black', alpha=1),
         transform=ax.transAxes)

plt.text(0.95, 0.95,
         'NGC 0628',
         c='white',
         ha='right', va='top',
         bbox=dict(facecolor='black', edgecolor='white', alpha=0.75),
         transform=ax.transAxes
         )

pix_scale = np.abs(jwst_hdu[0].header['CDELT1'])
kpc_scalebar = np.degrees(1e3 / 10e6) / pix_scale

# Add on scalebar
scalebar = AnchoredSizeBar(ax.transData,
                           kpc_scalebar, '1 kpc', 'lower left',
                           pad=0.5,
                           borderpad=1,
                           sep=4,
                           # color='white',
                           frameon=True,
                           size_vertical=1,
                           )

ax.add_artist(scalebar)
# ax.arrow(24.18, 15.751, kpc_scalebar, 0,
#          head_width=0, head_length=0,
#          fc='white', ec='white', width=0.0005,
#          transform=ax.get_transform('icrs'),
#          zorder=1000)
# plt.text(24.196, 15.75, '1 kpc',
#          color='white', rotation=0,
#          transform=ax.get_transform('icrs'),
#          zorder=1001)

plt.xlabel('RA (J2000)')
plt.ylabel('Dec (J2000)')

plt.tight_layout()

# plt.show()

plt.savefig(plot_name + '.png', bbox_inches='tight')
plt.savefig(plot_name + '.pdf', bbox_inches='tight')
plt.close()

# plt.show()

plot_name = os.path.join(plot_dir, 'jwst_alma_zoom')

# jwst_cutout = Cutout2D(jwst_sci.data,
#                        position=centre,
#                        size=size,
#                        wcs=jwst_w)
#
# zero_idx = np.where(jwst_cutout.data == 0)
#
# if high_pass_jwst:
#     jwst_cutout.data = butterworth(jwst_cutout.data)
#
# jwst_cutout.data[zero_idx] = np.nan
#
# vmin_jwst, vmax_jwst = np.nanpercentile(jwst_cutout.data, [pmin, pmax])
#
# # jwst_norm = matplotlib.colors.Normalize(vmin=vmin_jwst,
# #                                         vmax=vmax_jwst,
# #                                         )
# jwst_norm = matplotlib.colors.PowerNorm(1, vmin=vmin_jwst, vmax=vmax_jwst)
#
# alma_reproj = reproject_interp(alma_hdu[0],
#                                jwst_cutout.wcs.to_header(),
#                                shape_out=jwst_cutout.data.shape,
#                                return_footprint=False)
# alma_reproj[alma_reproj == 0] = np.nan
# # alma_norm = matplotlib.colors.Normalize(vmin=np.nanpercentile(alma_reproj, 3),
# #                                         vmax=np.nanpercentile(alma_reproj, 97),
# #                                         )
# vmin_alma, vmax_alma = np.nanpercentile(alma_reproj, [pmin, pmax])
#
# alma_norm = matplotlib.colors.PowerNorm(0.5, vmin=vmin_alma, vmax=vmax_alma)
#
# muse_reproj = reproject_interp(muse_hdu[1],  # [1] for real MUSE
#                                jwst_cutout.wcs.to_header(),
#                                shape_out=jwst_cutout.data.shape,
#                                return_footprint=False)
# muse_reproj[muse_reproj == 0] = np.nan
#
# vmin_muse, vmax_muse = np.nanpercentile(muse_reproj, [pmin, pmax])
#
# muse_norm = matplotlib.colors.PowerNorm(0.5, vmin=vmin_muse, vmax=vmax_muse)
#
# # muse_norm = matplotlib.colors.Normalize(vmin=np.nanpercentile(muse_reproj, 1),
# #                                         vmax=np.nanpercentile(muse_reproj, 99),
# #                                         )
#
# env_reproj = reproject_interp(environment_hdu[0],
#                               jwst_cutout.wcs.to_header(),
#                               shape_out=jwst_cutout.data.shape,
#                               return_footprint=False,
#                               order='nearest-neighbor')
# env_reproj[env_reproj != 6] = np.nan
# env_reproj[env_reproj == 6] = 1
#
# ii, jj = np.meshgrid(np.arange(env_reproj.shape[1]), np.arange(env_reproj.shape[0]))
#
# # rgb = make_lupton_rgb(jwst_cutout.data,
# #                       alma_reproj,
# #                       alma_reproj,
# #                       # Q=10,
# #                       stretch=0.5,
# #                       )
# rgb = np.dstack((jwst_norm(jwst_cutout.data),
#                  muse_norm(muse_reproj),
#                  # alma_norm(alma_reproj),
#                  alma_norm(alma_reproj)))
#
# plt.figure(figsize=(8, 4))
#
# ax = plt.subplot(projection=jwst_cutout.wcs)
# plt.imshow(rgb,
#            origin='lower')
#
# plt.contourf(ii, jj, env_reproj,
#              levels=1,
#              colors='white',
#              alpha=0.25,
#              )
#
# plt.xlabel('RA (J2000)')
# plt.ylabel('Dec (J2000)')
#
# # plt.text(0.95, 0.95,
# #          'Test',
# #          ha='right', va='top',
# #          bbox=dict(facecolor='white', edgecolor='black', alpha=1),
# #          transform=ax.transAxes)
#
# plt.tight_layout()
#
# plt.savefig(plot_name + '.pdf', bbox_inches='tight')
# plt.savefig(plot_name + '.png', bbox_inches='tight')
#
# # plt.show()
# plt.close()
#
jwst_hdu.close()
alma_hdu.close()
muse_hdu.close()
environment_hdu.close()

print('Complete!')
