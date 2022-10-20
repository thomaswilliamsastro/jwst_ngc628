import os

import astropy.units as u
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import uncertainties.unumpy as unp
from astropy.table import Table
from uncertainties import ufloat

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['font.size'] = 14

os.chdir('/Users/williams/Documents/phangs/jwst_early_science')

plot_dir = 'plots'

if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

# Between 3 and 4 kpc we have a constant offset of 20 degrees. Get angular rotations and see the scatter

dist = 9.84 * u.Mpc

tab = Table.read('data/PHANGS_RC_NGC0628_arcsec.ecsv')
r_kpc = np.radians(tab['radius'] / 3600) * dist.to(u.kpc)

v_err = (tab['VrotErr+'] + tab['VrotErr-']) / 2

idx = np.where(np.logical_and(r_kpc.value >= 3, r_kpc.value <= 4))

# Neglect uncertainties here since it's mostly from inclination which divides out
om = unp.uarray(tab['Vrot'].value[idx] / r_kpc.value[idx],
                0,
                # v_err[idx] / r_kpc.value[idx],
                )

om_p = ufloat(31.1, 3)

max_thetas = [40,
              55,
              ]

labels = ['CO-rich', 'CO-poor']
cs = ['b', 'r']

plot_name = os.path.join(plot_dir, 'timescales')

plt.figure(figsize=(5, 4))
ax = plt.subplot(1, 1, 1)

for i, max_theta in enumerate(max_thetas):

    theta = max_theta * (r_kpc.value[idx] - 3)
    theta = unp.uarray(theta, 5)

    t = theta / (0.576 * (om - om_p)) * 10

    plt.errorbar(r_kpc.value[idx], unp.nominal_values(t), yerr=unp.std_devs(t),
                 c=cs[i], marker='o', ls='none', label=labels[i])

plt.ylim([0, 300])

plt.legend(loc='upper left', fancybox=False)

ax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())

plt.xlabel(r'$r$ (kpc)')
plt.ylabel(r'$t$ (Myr)')

plt.grid()

plt.tight_layout()

# plt.show()

plt.savefig(plot_name + '.png', bbox_inches='tight')
plt.savefig(plot_name + '.pdf', bbox_inches='tight')
plt.close()

print('Complete!')
