from mpi4py import MPI
import sys, os
import numpy as np
import random
import math
import pickle
import itertools
import time
import json

import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import scipy.integrate as integrate
from scipy import special as sp
from sympy.physics.wigner import wigner_3j

import nbodykit.lab
import nbodykit

from helper_functions import read_files
from helper_functions import para_global_GR_sims as para

ep = 0.0001
grid = 512


def chi(z=para.redshift):
	''' 
	Return comoving radial distance to redshift z 
	(in Mpc/h)
	'''
	return para.cosmo.comoving_distance(z)

def Hz(z=para.redshift):
	''' 
	Return the Hubble parameter at redshift z
	(in km/s/(Mpc/h))
	'''
	return para.cosmo.H0*para.cosmo.efunc(z)

def aHz_in_invdist(z=para.redshift): 
	''' Return aH(z) in usits of (Mpc/h)^-1 '''
	Hz_in_invdist = Hz(z)/(para.cosmo.C)
	return Hz_in_invdist/(1. + z)

def fg(z=para.redshift):
	''' Return the linear growth rate '''
	return para.cosmo.scale_independent_growth_rate(z)
	#return para.fgrowth

def Ocdm(z=para.redshift):
	''' Return \Omega_cdm(z) '''
	return para.cosmo.Omega_cdm(z)

def E(z=para.redshift): 
	''' Return the dimensionless Hubble parameter H(z)/H0 '''
	return para.cosmo.efunc(z)

#Obs systematics

def bevo(z=para.redshift):
	''' Evolution bias '''
	return -4.

def dbevodz(z=para.redshift):
	''' 
	Redshift derivative of the evolution bias
	(see eq. 4.13)
	'''
	return 0.

def dB10dL(z=para.redshift):
	''' ? '''
	return 0.

def LumS(z=para.redshift):
	return -0.95

def B10(z=para.redshift):
	''' Linear bias parameter b1 '''
	return para.biases[0]

def B20(z=para.redshift):
	''' Second order bias parameter b2 '''
	return para.b2[0]

def Tidalbias(z=para.redshift):
	''' 
	Tidal bias 2(b1 - 1)/7 
	(see eq. 4.7)
	'''
	return -4.*(B10(z) - 1.)/7.

def dB10dz(z):
    return -0.00016

def A(z=para.redshift):
	''' 
	See eq. 4.6 in notes
	In the last term chi(z)*aHz_in_invdist(z) is dimensionless since:
	- aHz_in_invdist has unit (Mpc/h)^-1
	- chi has unit (Mpc/h)
	'''
	return bevo(z) + 1.5*Ocdm(z) - 3. + (2. - 5.*LumS(z))*(1. - 1./(chi(z)*aHz_in_invdist(z)))

def C(z=para.redshift):
	''' 
	In the last term chi(z)*aHz_in_invdist(z) is dimensionless since:
	- aHz_in_invdist has unit (Mpc/h)^-1
	- chi has unit (Mpc/h)
	(see eq. 4.13)
	'''
	return B10(z)*(A(z) + fg(z)) + dB10dz(z)/aHz_in_invdist(z) + 2.*(1. - 1./(chi(z)*aHz_in_invdist(z)))*dB10dL(z)

def EA(z=para.redshift):
	''' see eq. 4.14 '''
	return 4. - 2.*A(z) - 1.5*Ocdm(z)

def F2(k1, k2, k3):
	''' 
	F2 kernel
	(see eq. 4.9)
	'''
	k1dk2 = (k3**2 - k1**2 - k2**2)/2.
	mu = k1dk2/(k1*k2)
	F2 = 5./7. + 0.5*k1dk2*(1./(k1**2) + 1./(k2**2)) + 2.*mu**2/7.
	return 2.*F2

def G2(k1, k2, k3):
	''' 
	G2 kernel
	(see eq. 4.11)
	'''
	k1dk2 = (k3**2 - k1**2 - k2**2)/2.
	mu = k1dk2/(k1*k2)
	G2 = 3./7. + 0.5*k1dk2*(1./k1**2 + 1./k2**2) + 4.*mu**2/7.
	return 2.*G2

def S2(k1, k2, k3):
	'''
	S2 kernel
	(see eq. 4.8)
	'''
	k1dk2 = (k3**2 - k1**2 - k2**2)/2.
	mu = k1dk2/(k1*k2)
	return mu**2 - 1./3.

def Z1(z, mu):    
	'''
	Kaiser factor b1 + f\mu**2
	(see eq. 4.4)
	'''                     
	return B10(z) + fg(z)*mu**2

def Z2(z, k1, k2, k3, mu1, mu2, mu3):
	'''
	Z2 kernel... 
	(see eq. 4.10)
	'''                   
	Line1 = B10(z)*F2(k1,k2,k3) + B20(z) 										# ???
	Line2 = fg(z)**2*(2.*mu1**2*mu2**2 + mu1*mu2*(mu1**2*k1/k2 + mu2**2*k2/k1)) # first term in eq. 4.10
	Line3 = fg(z)*B10(z)*(mu1**2 + mu2**2 + mu1*mu2*(k1/k2 + k2/k1)) 			# second term in eq. 4.10
	Line4 = fg(z)*mu3**2*G2(k1,k2,k3) + Tidalbias(z)*S2(k1,k2,k3) 				# ???
	return Line1 + Line2 + Line3 + Line4 

def KD1(z, k, mu):  
	''' 
	KD1 kernel
	(see eq. 4.5)
	'''    
	return A(z)*fg(z)*mu*aHz_in_invdist(z)/k

def KD2(z, k1, k2, k3, mu1, mu2, mu3): 
	'''
	KD2 kernel
	(see eq. 4.12)
	'''
	mu12 = -(k1**2. + k2**2. -(k3**2.))/(2.*k1*k2)
	line1 = -1.5*(mu1*k1/k2**2 + mu2*k2/k1**2)*Ocdm(z)*B10(z) + 2.*mu12*(mu1/k2 + mu2/k1)*fg(z)**2 + (mu1/k1 + mu2/k2)*C(z)*fg(z)
	line2 = -1.5*(mu1**3*k1/k2**2 + mu2**3*k2/k1**2)*Ocdm(z)*fg(z) + mu1*mu2*(mu1/k2 + mu2/k1)*(1.5*Ocdm(z) - EA(z)*fg(z))*fg(z)
	line3 = (mu3/k3)*G2(k1,k2,k3)*A(z)*fg(z)
	return aHz_in_invdist(z)*(line1 + line2 + line3)

def NewtonainBispec(z, k1, k2, k3, mu1, mu2, mu3):
	'''
	Newtonian limit of the bispectrum
	(see eq. 5.3)
	'''
	return Z1(z, mu1)*Z1(z, mu2)*Z2(z, k1, k2, k3, mu1, mu2, mu3)

def DipoleBispec(z, k1, k2, k3, mu1, mu2, mu3):
	''' 
	Full relativistic bispectrum 
	(see eq. 5.4)
	'''
	tmp1 = Z1(z, mu1)*Z1(z, mu2)*KD2(z, k1, k2, k3, mu1, mu2, mu3)
	tmp2 = (Z1(z, mu1)*KD1(z, k2, mu2) + KD1(z, k1, mu1)*Z1(z, mu2))*Z2(z, k1, k2, k3, mu1, mu2, mu3)
	return tmp1 + tmp2

# def B_sq(z, k1, k2, Theta_12, Omega, Phi): 
# 	''' Calculate the bispectrum for a given k1 and k2 '''
# 	# K3=np.sqrt((K2*np.sin(Theta_12))**2. + (K1 + K2*np.cos(Theta_12))**2. )
# 	k3 = np.sqrt(k1**2 + k2**2 + 2.*k1*k2*np.cos(Theta_12))
# 	mu_1 = np.cos(Omega)
# 	mu_2 = np.sin(Theta_12)*np.sin(Omega)*np.cos(Phi) + np.cos(Theta_12)*np.cos(Omega)
# 	mu_3 = (-k1/k3)*mu_1 + (-k2/k3)*mu_2
	  
# 	BgN = np.zeros((len(Theta_12)))
# 	BgD = np.zeros((len(Theta_12)))
# 	for i in range(len(Theta_12)): 
# 	    BgN[i] = (NewtonainBispec(z, k1, k2, k3[i], mu_1, mu_2[i], mu_3[i])*(para.Pk_class(k1)*para.Pk_class(k2))
# 	            + NewtonainBispec(z, k2, k3[i], k1, mu_2[i], mu_3[i], mu_1)*para.Pk_class(k2)*para.Pk_class(k3[i])
# 	            + NewtonainBispec(z, k3[i], k1, k2, mu_3[i], mu_1, mu_2[i])*(para.Pk_class(k3[i])*para.Pk_class(k1)))
	 
# 	    BgD[i] = (DipoleBispec(z, k1, k2, k3[i], mu_1, mu_2[i], mu_3[i])*(para.Pk_class(k1)*para.Pk_class(k2))
# 	            + DipoleBispec(z, k2, k3[i], k1, mu_2[i], mu_3[i], mu_1)*para.Pk_class(k2)*para.Pk_class(k3[i])
# 	            + DipoleBispec(z, k3[i], k1, k2, mu_3[i], mu_1, mu_2[i])*(para.Pk_class(k3[i])*para.Pk_class(k1)))                                                       
# 	return k3, BgN, BgD

def BNgggintegrand(Omega, Phi, z, k1, k2, k3, P1, P2, P3, L, M):
	''' Integrant of eq. 5.13 in the Newtonian limit given k1, k2, k3, L=0 and M=0 '''
	#print("BNggg Omega = ", Omega)
	#print("BNggg Phi = ", Phi)
	costheta12 = -(k1**2. + k2**2. -(k3**2.))/(2.*k1*k2)
	#theta12 = np.arccos(costheta12)
	mu_1 = np.cos(Omega)
	#mu_2 = mu_1*costheta12 - math.sqrt(1. - mu_1**2)*np.cos(Phi)*math.sqrt(abs(1. - costheta12**2))
	#mu_2 = mu_1*costheta12 + np.sqrt(1. - mu_1**2)*np.sin(theta12)*np.sin(Phi)
	mu_2 = mu_1*costheta12 - np.sqrt(1. - mu_1**2)*np.sqrt(1. - costheta12**2)*np.sin(Phi)
	mu_3 = (-k1/k3)*mu_1 + (-k2/k3)*mu_2
	#print("BNggg mu_1 = ", mu_1)
	#print("BNggg mu_2 = ", mu_2)
	#print("BNggg mu_3 = ", mu_3)
	    
	BgN = (NewtonainBispec(z, k1, k2, k3, mu_1, mu_2, mu_3)*P1*P2 
		 + NewtonainBispec(z, k2, k3, k1, mu_2, mu_3, mu_1)*P2*P3
	     + NewtonainBispec(z, k3, k1, k2, mu_3, mu_1, mu_2)*P3*P1)
	# The first argument of sph_harm goes from [0, 2\pi] the second goes from [0, \pi]
	return np.real(BgN*sp.sph_harm(M, L, Phi, Omega))/np.sqrt(4.*np.pi*(2.*L + 1.))

def BDgggintegrand(Omega, Phi, z, k1, k2, k3, P1, P2, P3, L, M):
	''' Integrant of eq. 5.13 in the Newtonian limit given k1, k2, k3 '''
	#print("BDggg Omega = ", Omega)
	#print("BDggg Phi = ", Phi)
	costheta12 = -(k1**2 + k2**2 -(k3**2))/(2.*k1*k2)
	#theta12 = np.arccos(costheta12)
	mu_1 = np.cos(Omega)
	#mu_2 = mu_1*costheta12 - math.sqrt(1. - mu_1**2)*np.cos(Phi)*math.sqrt(abs(1. - costheta12**2))
	#mu_2 = mu_1*costheta12 + np.sqrt(1. - mu_1**2)*np.sin(theta12)*np.sin(Phi)
	mu_2 = mu_1*costheta12 - np.sqrt(1. - mu_1**2)*np.sqrt(1. - costheta12**2)*np.sin(Phi)
	mu_3 = -(k1/k3)*mu_1 - (k2/k3)*mu_2
	#print("BDggg mu_1 = ", mu_1)
	#print("BDggg mu_2 = ", mu_2)
	#print("BDggg mu_3 = ", mu_3)
	    
	BgD = (DipoleBispec(z, k1, k2, k3, mu_1, mu_2, mu_3)*P1*P2
		 + DipoleBispec(z, k2, k3, k1, mu_2, mu_3, mu_1)*P2*P3
	     + DipoleBispec(z, k3, k1, k2, mu_3, mu_1, mu_2)*P3*P1)
	if M == 0:
		return np.real(BgD*sp.sph_harm(M, L, Phi, Omega))/np.sqrt(4.*np.pi*(2.*L + 1.))
	else:
		return np.imag(BgD*sp.sph_harm(M, L, Phi, Omega))/np.sqrt(4.*np.pi*(2.*L + 1.))

def BNggg(z, k1, k2, k3, P1, P2, P3, L, M):
	''' 
	Calculate the bispectrum multipoles according to eq. 5.13 in the Newtonian limit
	'''
	time0 = time.time()
	result = integrate.dblquad(BNgggintegrand,       # func
							   ep, 	             # \Phi lower limit
							   2.*np.pi-ep, 	     # \Phi upper limit
							   lambda Phi: ep,    # \Omega lower limit
							   lambda Phi: np.pi-ep, # \Omega upper limit
							   args=(z, k1, k2, k3, P1, P2, P3, L, M), 
							   epsabs=1.49e-8, epsrel=1.49e-8)
	print("Calculated BNggg after %0.8f" % (time.time()-time0))
	return result

def BDggg(z, k1, k2, k3, P1, P2, P3, L, M):
	''' 
	Calculate the bispectrum multipoles B(k_1,k_2,k_3)
	according to eq. 5.13
	'''
	time0 = time.time()
	# result = integrate.nquad(BDgggintegrand, 
	# 						 [[0.002, 2.*np.pi-ep], [0.002, np.pi-ep]], 
	# 						 args=(z, k1, k2, k3, P1, P2, P3, L, M), 
	# 						 full_output=True)
	result = integrate.dblquad(BDgggintegrand,       # func
							   ep, 	             # \Phi lower limit
							   2.*np.pi-ep, 	     # \Phi upper limit
							   lambda Phi: ep,    # \Omega lower limit
							   lambda Phi: np.pi-ep, # \Omega upper limit
							   args=(z, k1, k2, k3, P1, P2, P3, L, M), 
							   epsabs=1.49e-8, epsrel=1.49e-8)
	print("Calculated BDggg after %0.8f" % (time.time()-time0))
	return result

def Bg(z, k1, k2, Theta_12):
	'''
	Calculate the bispectrum multipoles B(k_1,k_2,\theta_12)
	for different angles \theta_12
	(see eq. 5.13)
	'''
	print("Start Bg calculation for (k1=%0.4f,k2=%0.4f)" % (k1,k2)) 
	time0 = time.time() 
	#K3=np.sqrt((K2*np.sin(Theta_12))**2. + (K1 + K2*np.cos(Theta_12))**2. )
	k3 = np.sqrt(k1**2 + k2**2 + 2.*k1*k2*np.cos(Theta_12))
	pk_k1 = para.Pk_class(k1)
	pk_k2 = para.Pk_class(k2)

	BgN = np.zeros((len(Theta_12)))
	BgDm0 = np.zeros((len(Theta_12)))
	BgDm1 = np.zeros((len(Theta_12)))
	for i in range(len(Theta_12)): 
		print("Theta_12 = %d out of %d with theta_12 = %0.4f" % (i, len(Theta_12), Theta_12[i]))
		pk_k3 = para.Pk_class(k3[i])
		BgN[i] = BNggg(z, k1, k2, k3[i], pk_k1, pk_k2, pk_k3, 0, 0)[0]
		BgDm0[i] = BDggg(z, k1, k2, k3[i], pk_k1, pk_k2, pk_k3, 1, 0)[0]
		BgDm1[i] = BDggg(z, k1, k2, k3[i], pk_k1, pk_k2, pk_k3, 1, 1)[0]
		print("BgDm0[i] = ", BgDm0[i])
	print("Finished Bg calculation after %0.8fsec" % (time.time()-time0))	                                                                              
	return Theta_12, BgN, BgDm0, BgDm1  

def process_arrays(x):
	''' 
	Process arrays gathered with MPI
	- Combine the list of lists into one list
	- Turn the list into a numpy array
	'''
	x = list(itertools.chain.from_iterable(x))
	return np.array(x)

def switch_basis(theta_12, BDm0, BDm1):
	'''
	Switch the bispectrum basis from Scocimarro to tripolar
	spherical harmonics by integrating over theta_12 (k1 and k2 are fixed)
	'''
	ell1 = 1
	ell2 = 0
	L = 1
	#{'k1': [0.002, 0.005, 0.01, 0.02], 'k2': [0.002, 0.005, 0.01, 0.02], 'b1': 1.08, 'b2': 0.0, 'be': 0.0, 
	#'B101': [48662961.978645*sqrt(3), 78399057.5970337*sqrt(3), 76977056.1599017*sqrt(3), 37667729.8442636*sqrt(3)]}

	# B101_dict =  {'k1': [0.002, 0.005, 0.01, 0.02], 'k2': [0.002, 0.005, 0.01, 0.02], 'b1': 1.08, 'b2': 0.0, 'be': 0.0, 
	#'B101': [145988885.935935*sqrt(3), 235197172.791101*sqrt(3), 230931168.479705*sqrt(3), 113003189.532791*sqrt(3)]}
	print("switch basis to (%d,%d,%d)" % (ell1,ell2,L))
	BDm0_interp = interp1d(theta_12, BDm0)
	BDm1_interp = interp1d(theta_12, BDm1) # B_{11} = -B_{1-1}
	sp_factor = np.sqrt(4.*np.pi/(2.*ell2 + 1.))
	int1 = lambda costh_12: wigner_3j(ell1,ell2,L,0,0,0)*sp_factor*sp.sph_harm(0, ell2, costh_12, 0.)*BDm0_interp(np.arccos(costh_12))   # M = 0
	int2 = lambda costh_12: wigner_3j(ell1,ell2,L,0,-1,1)*sp_factor*sp.sph_harm(-1, ell2, costh_12, 0.)*BDm1_interp(np.arccos(costh_12)) # M = 1
	int3 = lambda costh_12: wigner_3j(ell1,ell2,L,0,1,-1)*sp_factor*sp.sph_harm(1, ell2, costh_12, 0.)*BDm1_interp(np.arccos(costh_12))  # M = -1
	tmp = integrate.nquad(int1, [[-1+ep, 1-ep]], full_output=True)
	print("tmp = ", tmp)
	result = tmp[0]
	print("result = ", result)
	if ell2 > 0:
		###### is there a symmetry which makes the M=1 and M=-1 contributions to cancel out?
		tmp = integrate.nquad(int2, [[-1+ep, 1-ep]], full_output=True)
		print("tmp = ", tmp)
		result += tmp[0]
		print("result = ", result)
		tmp = integrate.nquad(int3, [[-1+ep, 1-ep]], full_output=True)
		print("tmp = ", tmp)
		result += tmp[0]
		print("result = ", result)

	H = wigner_3j(ell1,ell2,L,0,0,0)
	N = (2.*ell1 + 1.)*(2.*ell2 + 1)*(2.*L + 1.)
	return -0.5*N*H*result/np.sqrt(4.*np.pi*(2.*L + 1.))

def plot_bi(theta_12, Bmono_N, BDm0, BDm1):
	''' Plot the bispectrum in the Scoccimarro basis '''
	# Divide by h to get to Mpc^6
	plt.plot(theta_12, Bmono_N/para.cosmo.h**6, color='k', label=r'$l = 0$, Newtonian')
	plt.plot(theta_12, -1.*Bmono_N/para.cosmo.h**6, color='k', linestyle='--')
	plt.plot(theta_12, BDm0/para.cosmo.h**6, color='red', label=r'$l = 1$, $m = 0$')
	# Plot negative bispectrum as dashed line
	plt.plot(theta_12, -1.*BDm0/para.cosmo.h**6, linestyle='--', color='red')
	plt.plot(theta_12, BDm1/para.cosmo.h**6, color='blue', label=r'$l = 1$, $m = 1$')
	# Plot negative bispectrum as dashed line
	plt.plot(theta_12, -1.*BDm1/para.cosmo.h**6, linestyle='--', color='blue')
	plt.yscale('log')
	plt.legend(loc='best')
	plt.title(r'Monopole and Dipole')
	plt.xlabel(r'$\theta_{12}$')
	plt.ylabel(r'$B({\bf k}_1, {\bf k}_2, {\bf k}_3)$ [Mpc$^6$]')
	plt.savefig('%s/bi_model_theta12.png' % para.outpath, dpi=500)
	plt.show()
	return 

def plot_bispectrum_diff(bk_dict_list1, bk_dict_list2, B101_dict_list, tag):
	fig = plt.figure()
	ax = plt.gca()
	for i, bk_dict in enumerate(bk_dict_list1):
		if bk_dict:
			plt.plot(bk_dict['k1'], bk_dict['k1']*(bk_dict_list2[i]['bk_img']-bk_dict['bk_img']), label=bk_dict['label'].replace('_', ''))
	for kn_dict in B101_dict_list:
		if bk_dict:
			plt.plot(kn_dict['k1'], kn_dict['k1']*kn_dict['B101'], 'o', label=bk_dict['label'])
	plt.xlabel(r'$k_{1}$ [h/Mpc]')
	plt.ylabel(r'$k_1\Delta[iB(k_1,k_2,k_1=k_2)]$')
	plt.xscale('log')
	plt.axhline(y=0., color='k', linestyle='--')
	plt.text(0.5, 0.9, tag, fontsize=15, transform=ax.transAxes)
	#plt.legend(loc=0)
	fig.savefig("/Users/xflorian/GR_sims/outputs/bispectrum/bi_%s_diff.pdf" % tag)
	plt.show()
	return

def read_b101(k_array, theta_res):
	B101_dict = {}
	B101_dict['k1'] = []
	B101_dict['k2'] = []
	B101_dict['B101'] = []
	for k in k_array:
		filename = r"%s/B101_%0.4f_%0.4f_%d.dat" % (para.outpath, k, k, theta_res)
		if os.path.isfile(filename):
			B101_dict['k1'].append(k)
			B101_dict['k2'].append(k)
			with open( filename, "rb" ) as f:
				tmp = pickle.load(f)
				B101_dict['B101'].append(tmp)
	B101_dict['k1'] = np.array(B101_dict['k1'])
	B101_dict['k2'] = np.array(B101_dict['k2'])
	B101_dict['B101'] = np.array(B101_dict['B101'])
	B101_dict['label'] = "theta res = %d" % theta_res
	print("B101_dict = ", B101_dict)
	return B101_dict

def main():
	'''
	Bispectrum model calculator
	- We first integrate over the orientation angles of the triangle \phi and \theta
	resulting in B_{LM}(k_1,k_2,\theta_{12})
	- We than integrate over \theta_{12} and change the base to get B_{\ell_1\ell_2L}(k_1,k_2)
	'''
	comm = MPI.COMM_WORLD
	rank = comm.Get_rank()
	size = comm.Get_size()
	print('Rank %d out of %d hast started' % (rank, size))
	time0 = time.time()

	# Flag which allows to recalculate bispectra, even if it is already on disk
	recalc = True

	B101_dict = {}
	B101_dict['k1'] = []
	B101_dict['k2'] = []
	B101_dict['b1'] = B10(para.redshift)
	B101_dict['b2'] = B20(para.redshift)
	B101_dict['be'] = bevo(para.redshift)
	B101_dict['B101'] = []

	theta_resolution = 20
	# Divide the theta array according to the number of core available
	index_low = int(rank*theta_resolution/size)
	index_high = int((rank+1)*theta_resolution/size)
	# Loop over k1 and k2
	#for (k1, k2) in [(0.002,0.002), (0.005,0.005), (0.01,0.01), (0.02,0.02), (0.03,0.03)]:
	for (k1, k2) in [(0.01*para.cosmo.h,0.01*para.cosmo.h)]: # multiply by h to to get 0.01Mpc^{-1}
		#for (k1, k2) in [(0.1,0.1)]: # multiply by h to to get 0.01Mpc^{-1}
		theta_12 = np.linspace(0.002, np.pi-0.002, theta_resolution)
		# Calculate B(k1,k2,\theta_12) if needed
		filename = r"%s/theta_12_%0.4f_%0.4f_%d.dat" % (para.outpath, k1, k2, theta_resolution)
		if not os.path.isfile(filename) or recalc:
			theta_12, Bmono_N, BDm0, BDm1 = Bg(para.redshift, k1, k2, theta_12[index_low:index_high])
			# Collect jobs from different cores... at root
			theta_12 = comm.gather(theta_12, root=0)
			Bmono_N = comm.gather(Bmono_N, root=0)
			BDm0 = comm.gather(BDm0, root=0)
			BDm1 = comm.gather(BDm1, root=0)
			comm.Barrier()

			if rank == 0:
				# Store results on disk
				theta_12 = process_arrays(theta_12)
				Bmono_N = process_arrays(Bmono_N)
				BDm0 = process_arrays(BDm0)
				BDm1 = process_arrays(BDm1)

				pickle.dump(theta_12, open( filename, "wb" ))
				pickle.dump(Bmono_N, open(r"%s/Bmono_N_%0.4f_%0.4f_%d.dat" % (para.outpath, k1, k2, theta_resolution), "wb" ))
				pickle.dump(BDm0, open(r"%s/BDm0_%0.4f_%0.4f_%d.dat" % (para.outpath, k1, k2, theta_resolution), "wb" ))
				pickle.dump(BDm1, open(r"%s/BDm1_%0.4f_%0.4f_%d.dat" % (para.outpath, k1, k2, theta_resolution), "wb" ))
		else:
			# Load from disk if files already exist
			theta_12 = pickle.load(open( filename, "rb" ))
			Bmono_N = pickle.load(open(r"%s/Bmono_N_%0.4f_%0.4f_%d.dat" % (para.outpath, k1, k2, theta_resolution), "rb" ))
			BDm0 = pickle.load(open(r"%s/BDm0_%0.4f_%0.4f_%d.dat" % (para.outpath, k1, k2, theta_resolution), "rb" ))
			BDm1 = pickle.load(open(r"%s/BDm1_%0.4f_%0.4f_%d.dat" % (para.outpath, k1, k2, theta_resolution), "rb" ))

		if rank == 0:
			plot_bi(theta_12, Bmono_N, BDm0, BDm1)
			sys.exit()

			# Convert to triangular spherical harmonics basis 
			# Integral over \theta_12
			B101 = switch_basis(theta_12, BDm0, BDm1)

			filename = r"%s/B101_%0.4f_%0.4f_%d.dat" % (para.outpath, k1, k2, theta_resolution)
			pickle.dump(B101, open( filename, "wb" ))

			B101_dict['k1'].append(k1)
			B101_dict['k2'].append(k2)
			B101_dict['B101'].append(B101)
		comm.Barrier()

	B101_dict['k1'] = np.array(B101_dict['k1'])
	B101_dict['k2'] = np.array(B101_dict['k2'])
	B101_dict['B101'] = np.array(B101_dict['B101'])
	B101_dict['label'] = "theta res = %d" % theta_resolution
	print("B101_dict = ", B101_dict)

	if rank == 0:
		filename = "/Users/xflorian/GR_sims/outputs/bispectrum/bk101_%d_%d_%d_100_200_z1_kmax_0_1_40_diag" % (grid, grid, grid)
		bk101_z1 = read_files.read_bispectrum(filename)
		filename = "/Users/xflorian/GR_sims/outputs/bispectrum/bk101_%d_%d_%d_100_200_z2_kmax_0_1_40_diag" % (grid, grid, grid)
		bk101_z2 = read_files.read_bispectrum(filename)

		B101_dict_res20 = read_b101(B101_dict['k1'], 20)

		plot_bispectrum_diff([bk101_z1], [bk101_z2], [B101_dict, B101_dict_res20], "Doppler_term")
	comm.Barrier()
	return 


# to call the main() function to begin the program.
if __name__ == '__main__':
    main()
