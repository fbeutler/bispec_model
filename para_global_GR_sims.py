import nbodykit.lab
import nbodykit

red_tag = 'z1'
tag = "GR_sims"
combine_bins = 1
kmin = 0.001
kmax = 2.
debug = False

zmin = 0.05
zmax = 0.465
redshift = 1.0 #0.341

biases = [1.3]
b2 = [-0.74]
#biases = [1.08, 1.22, 1.42, 1.69, 2.07, 2.59]

inpath = '/Users/xflorian/GR_sims'
outpath = '/Users/xflorian/GR_sims/products'

if red_tag == 'z1':
	redshift = 0.38 #0.51 # 0.61 
elif red_tag == 'z2':
	redshift = 0.51 #0.51 # 0.61 
elif red_tag == 'z3':
	redshift = 0.61 #0.51 # 0.61 
else:
	print("ERROR: invalid red_tag")
	sys.exit()

Nmocks = 2048

patchy_cosmology = {
	'omega_m': 0.307115,
	'omega_b': 0.048206,
	'sigma_8': 0.8288,
	'omega_nu': 0.,
	'ns': 0.9611,
	'h': 0.6777
}

GR_sim_cosmology = {
	'omega_m': 0.25733000,
	'omega_b': 0.043557099,
	'sigma_8': 0.80100775,
	'omega_nu': 0.,
	'ns': 0.963,
	'h': 0.72
}

Clarkson_cosmology = {
	'omega_m': 0.308,
	'omega_b': 0.05236,
	'omega_nu': 0.,
	'ns': 0.968,
	'h': 0.67
}

used_cos = Clarkson_cosmology

om_0 = used_cos['omega_m'] - used_cos['omega_nu'] - used_cos['omega_b']
cosmo = nbodykit.lab.cosmology.Cosmology(
	Omega0_cdm=om_0, 
	N_ncdm=1,
    Omega0_k=0.,
    Omega0_b=used_cos['omega_b'],
    h=used_cos['h'],
    n_s=used_cos['ns'],
    T0_cmb=2.7255
)
fgrowth = cosmo.scale_independent_growth_rate(redshift)
Pk_class = nbodykit.lab.cosmology.power.linear.LinearPower(cosmo, redshift=redshift, transfer='CLASS')
Tk_class = nbodykit.lab.cosmology.power.transfers.CLASS(cosmo, redshift=redshift)
