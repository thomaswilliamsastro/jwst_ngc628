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

environment_file_name = os.path.join('data', 'NGC0628_simple.fits')

# Regrid environment to JWST
jwst_hdu = fits.open(jwst_file_name)
jwst_sci = jwst_hdu[0]
jwst_sci.data[jwst_sci.data == 0] = np.nan

environment_hdu = fits.open(environment_file_name)
environment_data = environment_hdu[0]

environment_regrid = reproject_interp(environment_hdu,
                                      jwst_sci.header,
                                      order='nearest-neighbor',
                                      return_footprint=False)

total_flux = np.nansum(jwst_sci.data)
flux_in_arms = np.nansum(jwst_sci.data[environment_regrid == 6])

print(total_flux)
print(flux_in_arms)

print(flux_in_arms / total_flux * 100)

# plt.figure()
# plt.imshow(environment_regrid, origin='lower')
# plt.show()

jwst_hdu.close()
environment_hdu.close()
