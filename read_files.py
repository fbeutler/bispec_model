import os, sys
import numpy as np
from scipy.interpolate import interp1d

# my files


def read_data(tag):
	'''
	Here we real all power spectra for NGC and SGC and return a list of lists
	'''
	# Read the mock catalog power spectra
	list_of_pk_SGC = []
	#for i in range(2049):
	for i in range(1, 2049+1):
		#filename = '/Users/xflorian/clustools/output/dipole/power_V6S/SGC_%s/ps1D_DR12_patchy_SGC_%s_COMPnbar_TSC_V6S_%d_600_600_600_300.dat' % (red_tag, red_tag, i)
		if tag == 'V6C':
			print("read V6C")
			filename = '/Users/xflorian/clustools/output/dipole/power_V6C/SGC_%s/ps1D_DR12_patchy_SGC_%s_COMPnbar_TSC_V6C_%d_600_600_600_300.dat' % (red_tag, red_tag, i)
		else:
			filename = '/Users/xflorian/clustools/output/dipole/power_V6S/SGC_%s/ps1D_DR12_patchy_SGC_%s_COMPnbar_TSC_V6S_%d_600_600_600_300.dat' % (red_tag, red_tag, i)
		#filename = '/Users/xflorian/clustools/output/dipole/ps1D_DR12_patchy_NGC_z1_COMPnbar_TIC_lin_%d_500_500_500_300.dat' % (i)
		if os.path.isfile(filename):
			print("reading %s" % filename)
			list_of_pk_SGC.append(read_power(filename, combine_bins))
		else:
			print("%s not found" % filename)
	print("N SGC = ", len(list_of_pk_SGC))
	if not list_of_pk_SGC:
		print("No SGC data")
		sys.exit()
	
	list_of_pk_NGC = []
	for i in range(1, 2049+1):
		#filename = '/Users/xflorian/clustools/output/dipole/ps1D_DR12_patchy_SGC_z1_COMPnbar_TIC_lin_%d_300_300_300_300.dat' % (i)
		if tag == 'V6C':
			print("read V6C")
			filename = '/Users/xflorian/clustools/output/dipole/power_V6C/NGC_%s/ps1D_DR12_patchy_NGC_%s_COMPnbar_TSC_V6C_%d_600_600_600_300.dat' % (red_tag, red_tag, i)
		else:
			filename = '/Users/xflorian/clustools/output/dipole/power_V6S/NGC_%s/ps1D_DR12_patchy_NGC_%s_COMPnbar_TSC_V6S_%d_600_600_600_300.dat' % (red_tag, red_tag, i)
		if os.path.isfile(filename):
			print("reading %s" % filename)
			list_of_pk_NGC.append(read_power(filename, combine_bins))
		else:
			print("%s not found" % filename)
	print("N NGC = ", len(list_of_pk_NGC))
	if not list_of_pk_NGC:
		print("No NGC data")
		sys.exit()
	return list_of_pk_SGC, list_of_pk_NGC


def read_power_old(filename, para):
	'''
	Read power spectrum and return dictionary
	'''
	if not os.path.isfile(filename):
		print("WARNING: file %s not found" % filename)
		return {}
	else:
		output = {}
		with open(filename, "r") as f:
			bin_counter = 0
			int_k = int_pk0 = int_pk1 = int_pk2 = int_pk3 = int_pk4 = norm = 0.
			pk0 = []
			pk1 = []
			pk2 = []
			pk3 = []
			pk4 = []
			k_ps = []
			for i, line in enumerate(f):
				if i < 31: # Ignore header
					if line[:5] == 'kx_ny':
						dummy = list(map(str, line.split()))
						output['kx_ny'] = float(dummy[2])
						output['ky_ny'] = float(dummy[5])
						output['kz_ny'] = float(dummy[8])
						print("Nyquist frequency has been found to be:") 
						print("kx_ny = %f, kx_ny = %f, kx_ny = %f" % (output['kx_ny'], output['ky_ny'], output['kz_ny']))
				else:
					dummy = list(map(float, line.split()))

					int_k += dummy[1]*dummy[8]
					int_pk0 += dummy[2]*dummy[8]
					int_pk1 += dummy[3]*dummy[8]
					int_pk2 += dummy[4]*dummy[8]
					int_pk3 += dummy[5]*dummy[8]
					int_pk4 += dummy[6]*dummy[8]
					norm += dummy[8]
					bin_counter += 1

					if bin_counter == para.combine_bins:
						if norm == 0:
							# If norm is zero, no modes have been found, so all 
							# power spectra should be zero
							norm = 1.
						keff = int_k/float(norm)
						if para.kmax > keff > para.kmin:
							k_ps.append(keff)
							pk0.append(int_pk0/float(norm))
							pk1.append(int_pk1/float(norm))
							pk2.append(int_pk2/float(norm))
							pk3.append(int_pk3/float(norm))
							pk4.append(int_pk4/float(norm))
						else:
							print("WARNING: Mode %f has been ignored" % int_k)

						int_k = int_pk0 = int_pk1 = int_pk2 = int_pk3 = int_pk4 = norm = 0.
						bin_counter = 0

		print("k_ps = ", k_ps)
		if k_ps and len(k_ps) > 1:
			output['delta_k'] = k_ps[-1] - k_ps[-2]
		output['k'] = np.array(k_ps)
		output['pk0'] = np.array(pk0)
		output['pk1'] = np.array(pk1)
		output['pk2'] = np.array(pk2)
		output['pk3'] = np.array(pk3)
		output['pk4'] = np.array(pk4)
		if 'kx_ny' not in output:
			print("WARNING: No Nyquist frequency found!")
		return output


def read_bispectrum(filename):
	bi_dict = {}
	bi_dict['k1'] = []
	bi_dict['k2'] = []
	bi_dict['bk_real'] = []
	bi_dict['bk_img'] = []
	if not os.path.isfile(filename):
		return []
	else:
		with open(filename, "r") as f:
			for line in f:
				dummy = list(map(float, line.split()))
				bi_dict['k1'].append(dummy[0])
				bi_dict['k2'].append(dummy[1])
				bi_dict['bk_real'].append(dummy[2])
				bi_dict['bk_img'].append(dummy[3])
		bi_dict['k1'] = np.array(bi_dict['k1'])
		bi_dict['k2'] = np.array(bi_dict['k2'])
		bi_dict['bk_real'] = np.array(bi_dict['bk_real'])
		bi_dict['bk_img'] = np.array(bi_dict['bk_img'])
	if 'bk101' in filename:
		bi_dict['label'] = '101'
	elif 'bk011' in filename:
		bi_dict['label'] = '011'
	elif 'bk000' in filename:
		bi_dict['label'] = '000'
	else:
		print("WARNING: New multipole?")
	return bi_dict
	

def read_power(filename, para):
	'''
	Read power spectrum and return dictionary
	'''
	if not os.path.isfile(filename):
		print("WARNING: file %s not found" % filename)
		return {}
	else:
		output = {}
		with open(filename, "r") as f:
			bin_counter = 0
			int_k = int_pk0 = int_pk1 = int_pk2 = int_pk3 = int_pk4 = norm = 0.
			int_sig0 = int_sig1 = int_sig2 = int_sig3 = int_sig4 = 0.
			pk0 = []
			pk1 = []
			pk2 = []
			pk3 = []
			pk4 = []
			sig0 = []
			sig1 = []
			sig2 = []
			sig3 = []
			sig4 = []
			k_ps = []
			for i, line in enumerate(f):
				if i < 31: # Ignore header
					if line[:5] == 'kx_ny':
						dummy = list(map(str, line.split()))
						output['kx_ny'] = float(dummy[2])
						output['ky_ny'] = float(dummy[5])
						output['kz_ny'] = float(dummy[8])
						print("Nyquist frequency has been found to be:") 
						print("kx_ny = %f, kx_ny = %f, kx_ny = %f" % (output['kx_ny'], output['ky_ny'], output['kz_ny']))
				else:
					dummy = list(map(float, line.split()))

					if len(dummy) == 13:
						int_k += dummy[1]*dummy[12]
						int_pk0 += dummy[2]*dummy[12]
						int_pk1 += dummy[4]*dummy[12]
						int_pk2 += dummy[6]*dummy[12]
						int_pk3 += dummy[8]*dummy[12]
						int_pk4 += dummy[10]*dummy[12]
						norm += dummy[12]
						bin_counter += 1

						if dummy[12] > 0:
							int_sig0 += 1./dummy[3]**2
							int_sig1 += 1./dummy[5]**2
							int_sig2 += 1./dummy[7]**2
							int_sig3 += 1./dummy[9]**2
							int_sig4 += 1./dummy[11]**2

						if bin_counter == para.combine_bins:
							if norm == 0:
								# If norm is zero, no modes have been found, so all 
								# power spectra should be zero
								norm = 1.
							keff = int_k/float(norm)
							if para.kmax > keff > para.kmin:
								k_ps.append(keff)
								pk0.append(int_pk0/float(norm))
								sig0.append(1./np.sqrt(int_sig0))
								pk1.append(int_pk1/float(norm))
								sig1.append(1./np.sqrt(int_sig1))
								pk2.append(int_pk2/float(norm))
								sig2.append(1./np.sqrt(int_sig2))
								pk3.append(int_pk3/float(norm))
								sig3.append(1./np.sqrt(int_sig3))
								pk4.append(int_pk4/float(norm))
								sig4.append(1./np.sqrt(int_sig4))
							else:
								print("WARNING: Mode %f has been ignored" % int_k)

							int_k = int_pk0 = int_pk1 = int_pk2 = int_pk3 = int_pk4 = norm = 0.
							int_sig0 = int_sig1 = int_sig2 = int_sig3 = int_sig4 = 0.
							bin_counter = 0
					else:
						int_k += dummy[1]*dummy[8]
						int_pk0 += dummy[2]*dummy[8]
						int_pk1 += dummy[3]*dummy[8]
						int_pk2 += dummy[4]*dummy[8]
						int_pk3 += dummy[5]*dummy[8]
						int_pk4 += dummy[6]*dummy[8]
						norm += dummy[8]
						bin_counter += 1

						if bin_counter == para.combine_bins:
							if norm == 0:
								# If norm is zero, no modes have been found, so all 
								# power spectra should be zero
								norm = 1.
							keff = int_k/float(norm)
							if para.kmax > keff > para.kmin:
								k_ps.append(keff)
								pk0.append(int_pk0/float(norm))
								pk1.append(int_pk1/float(norm))
								pk2.append(int_pk2/float(norm))
								pk3.append(int_pk3/float(norm))
								pk4.append(int_pk4/float(norm))
							else:
								print("WARNING: Mode %f has been ignored" % int_k)

							int_k = int_pk0 = int_pk1 = int_pk2 = int_pk3 = int_pk4 = norm = 0.
							int_sig0 = int_sig1 = int_sig2 = int_sig3 = int_sig4 = 0.
							bin_counter = 0

		print("k_ps = ", k_ps)
		if k_ps and len(k_ps) > 1:
			output['delta_k'] = k_ps[-1] - k_ps[-2]
		output['k'] = np.array(k_ps)
		output['pk0'] = np.array(pk0)
		output['pk1'] = np.array(pk1)
		output['pk2'] = np.array(pk2)
		output['pk3'] = np.array(pk3)
		output['pk4'] = np.array(pk4)
		output['sig0'] = np.array(sig0)
		output['sig1'] = np.array(sig1)
		output['sig2'] = np.array(sig2)
		output['sig3'] = np.array(sig3)
		output['sig4'] = np.array(sig4)
		if 'kx_ny' not in output:
			print("WARNING: No Nyquist frequency found!")
		return output


def read_kwindow(filename, para):
	'''
	Read window function and return dictionary
	'''
	d = para.cosmo.comoving_distance(para.redshift)

	W = {}
	W['k'] = []
	for i in range(0, 5):
		for j in range(0, 5):
			W['W%d%d' % (i,j)] = []

	if not os.path.isfile(filename):
		print("WARNING: file %s not found" % filename)
		return {}
	else:
		with open(filename, 'r') as f:
			norm1 = 0.;
			norm2 = 0.;
			norm3 = 0.;
			for i, line in enumerate(f):
				if i < 31: # Ignore header
					if line[:5] == 'kx_ny':
						dummy = list(map(str, line.split()))
						W['kx_ny'] = float(dummy[2])
						W['ky_ny'] = float(dummy[5])
						W['kz_ny'] = float(dummy[8])
						print("Nyquist frequency has been found to be:") 
						print("kx_ny = %f, kx_ny = %f, kx_ny = %f" % (W['kx_ny'], W['ky_ny'], W['kz_ny']))
				else:
					dummy = list(map(float, line.split()))
					if not norm1:
						norm1 = dummy[2]
					if not norm2:
						norm2 = dummy[7]
					if not norm3:
						norm3 = dummy[12]
					W['k'].append(dummy[0])
					W['W00'].append(dummy[2])
					W['W01'].append(dummy[3])
					W['W02'].append(dummy[4])
					W['W03'].append(dummy[5])
					W['W04'].append(dummy[6])
					W['W10'].append(dummy[7])
					W['W11'].append(dummy[8])
					W['W12'].append(dummy[9])
					W['W13'].append(dummy[10])
					W['W14'].append(dummy[11])
					W['W20'].append(dummy[12])
					W['W21'].append(dummy[13])
					W['W22'].append(dummy[14])
					W['W23'].append(dummy[15])
					W['W24'].append(dummy[16])

	W['k'] = np.array(W['k'])
	W['W00'] = np.array(W['W00'])/norm1
	W['W01'] = np.array(W['W01'])/norm1
	W['W02'] = np.array(W['W02'])/norm1
	W['W03'] = np.array(W['W03'])/norm1
	W['W04'] = np.array(W['W04'])/norm1
	W['W10'] = np.array(W['W10'])/(norm2*d)
	W['W11'] = np.array(W['W11'])/(norm2*d)
	W['W12'] = np.array(W['W12'])/(norm2*d)
	W['W13'] = np.array(W['W13'])/(norm2*d)
	W['W14'] = np.array(W['W14'])/(norm2*d)
	W['W20'] = np.array(W['W20'])/(norm3*d**2)
	W['W21'] = np.array(W['W21'])/(norm3*d**2)
	W['W22'] = np.array(W['W22'])/(norm3*d**2)
	W['W23'] = np.array(W['W23'])/(norm3*d**2)
	W['W24'] = np.array(W['W23'])/(norm3*d**2)
	return W


def read_kwindowFT(inputfile):
	# Read power spectrum data... combine bins with the combine_bins parameter
	kmax = None

	kwin = []
	Wlist = [[[],[],[],[],[],[],[],[],[]] for i in range(3)]

	header = False
	bin_counter = 0
	Nmodescounter = 0
	if os.path.isfile(inputfile):
		with open(inputfile, 'r') as f:
			counter = 0
			for ii, line in enumerate(f):
				dummy = list(map(float, line.split()))

				kwin.append(dummy[0])
				Wlist[0][0].append(dummy[1])
				Wlist[0][1].append(dummy[2])
				Wlist[0][2].append(dummy[3])
				Wlist[0][3].append(dummy[4])
				Wlist[0][4].append(dummy[5])
				Wlist[1][0].append(dummy[10])
				Wlist[1][1].append(dummy[11])
				Wlist[1][2].append(dummy[12])
				Wlist[1][3].append(dummy[13])
				Wlist[1][4].append(dummy[14])
				Wlist[2][0].append(dummy[19])
				Wlist[2][1].append(dummy[20])
				Wlist[2][2].append(dummy[21])
				Wlist[2][3].append(dummy[22])
				Wlist[2][4].append(dummy[23])
	else:
		print("%s not found" % inputfile)
	return np.array(kwin), np.array(Wlist)


def get_TNS_model_nowin(filename, cosmo):
	d = cosmo.comoving_distance(redshift)

	model0_NGC = {}
	model0_NGC['k'] = []
	model0_NGC['pk'] = []
	model0_NGC['linestyle'] = '-'
	model0_NGC['label'] = 'TNS w/o window'
	model1_NGC = {}
	model1_NGC['k'] = []
	model1_NGC['pk'] = []
	model1_NGC['linestyle'] = '-'
	model1_NGC['label'] = 'TNS w/o window'
	model2_NGC = {}
	model2_NGC['k'] = []
	model2_NGC['pk'] = []
	model2_NGC['linestyle'] = '-'
	model2_NGC['label'] = 'TNS w/o window'
	model3_NGC = {}
	model3_NGC['k'] = []
	model3_NGC['pk'] = []
	model3_NGC['linestyle'] = '-'
	model3_NGC['label'] = 'TNS w/o window'
	model4_NGC = {}
	model4_NGC['k'] = []
	model4_NGC['pk'] = []
	model4_NGC['linestyle'] = '-'
	model4_NGC['label'] = 'TNS w/o window'

	model0_SGC = {}
	model0_SGC['k'] = []
	model0_SGC['pk'] = []
	model0_SGC['linestyle'] = '-'
	model0_SGC['label'] = 'TNS w/o window'
	model1_SGC = {}
	model1_SGC['k'] = []
	model1_SGC['pk'] = []
	model1_SGC['linestyle'] = '-'
	model1_SGC['label'] = 'TNS w/o window'
	model2_SGC = {}
	model2_SGC['k'] = []
	model2_SGC['pk'] = []
	model2_SGC['linestyle'] = '-'
	model2_SGC['label'] = 'TNS w/o window'
	model3_SGC = {}
	model3_SGC['k'] = []
	model3_SGC['pk'] = []
	model3_SGC['linestyle'] = '-'
	model3_SGC['label'] = 'TNS w/o window'
	model4_SGC = {}
	model4_SGC['k'] = []
	model4_SGC['pk'] = []
	model4_SGC['linestyle'] = '-'
	model4_SGC['label'] = 'TNS w/o window'
	with open(filename, "r") as f:
		for line in f:
			if line[0] != '#':
				dummy = list(map(float, line.split()))
				model0_NGC['k'].append(dummy[0])
				model1_NGC['k'].append(dummy[0])
				model2_NGC['k'].append(dummy[0])
				model3_NGC['k'].append(dummy[0])
				model4_NGC['k'].append(dummy[0])

				model0_SGC['k'].append(dummy[0])
				model1_SGC['k'].append(dummy[0])
				model2_SGC['k'].append(dummy[0])
				model3_SGC['k'].append(dummy[0])
				model4_SGC['k'].append(dummy[0])

				model0_NGC['pk'].append(dummy[1])
				model1_NGC['pk'].append(dummy[2]/d)
				model2_NGC['pk'].append(dummy[3])
				model3_NGC['pk'].append(dummy[4]/d)
				model4_NGC['pk'].append(dummy[5])

				model0_SGC['pk'].append(dummy[6])
				model1_SGC['pk'].append(dummy[7]/d)
				model2_SGC['pk'].append(dummy[8])
				model3_SGC['pk'].append(dummy[9]/d)
				model4_SGC['pk'].append(dummy[10])

	list_of_models_NGC = []
	list_of_models_NGC.append(model0_NGC)
	list_of_models_NGC.append(model1_NGC)
	list_of_models_NGC.append(model2_NGC)
	list_of_models_NGC.append(model3_NGC)
	list_of_models_NGC.append(model4_NGC)
	list_of_models_SGC = []
	list_of_models_SGC.append(model0_SGC)
	list_of_models_SGC.append(model1_SGC)
	list_of_models_SGC.append(model2_SGC)
	list_of_models_SGC.append(model3_SGC)
	list_of_models_SGC.append(model4_SGC)
	return list_of_models_NGC, list_of_models_SGC


def get_TNS_model(filename, tag):

	kmax = 0.3

	model0 = {}
	model0['k'] = []
	model0['pk'] = []
	model0['linestyle'] = '-'
	model1 = {}
	model1['k'] = []
	model1['pk'] = []
	model1['linestyle'] = '-'
	model2 = {}
	model2['k'] = []
	model2['pk'] = []
	model2['linestyle'] = '-'
	model3 = {}
	model3['k'] = []
	model3['pk'] = []
	model3['linestyle'] = '-'
	model4 = {}
	model4['k'] = []
	model4['pk'] = []
	model4['linestyle'] = '-'
	dummy01 = []
	dummy02 = []
	dummy03 = []
	dummy11 = []
	dummy12 = []
	dummy13 = []
	dummy21 = []
	dummy22 = []
	dummy23 = []
	dummy31 = []
	dummy32 = []
	dummy33 = []
	dummy41 = []
	dummy42 = []
	dummy43 = []
	with open(filename, "r") as f:
		for line in f:
			if line[0] != '#':
				dummy = list(map(float, line.split()))
				model0['k'].append(dummy[0])
				model1['k'].append(dummy[0])
				model2['k'].append(dummy[0])
				model3['k'].append(dummy[0])
				model4['k'].append(dummy[0])
				dummy01.append(dummy[2])
				dummy02.append(dummy[3])
				dummy03.append(dummy[4])
				dummy11.append(dummy[6])
				dummy12.append(dummy[7])
				dummy13.append(dummy[8])
				dummy21.append(dummy[10])
				dummy22.append(dummy[11])
				dummy23.append(dummy[12])
				dummy31.append(dummy[14])
				dummy32.append(dummy[15])
				dummy33.append(dummy[16])
				dummy41.append(dummy[18])
				dummy42.append(dummy[19])
				dummy43.append(dummy[20])

	dummy01 = np.array(dummy01)
	dummy02 = np.array(dummy02)
	dummy03 = np.array(dummy03)
	dummy11 = np.array(dummy11)
	dummy12 = np.array(dummy12)
	dummy13 = np.array(dummy13)
	dummy21 = np.array(dummy21)
	dummy22 = np.array(dummy22)
	dummy23 = np.array(dummy23)
	dummy31 = np.array(dummy31)
	dummy32 = np.array(dummy32)
	dummy33 = np.array(dummy33)
	dummy41 = np.array(dummy41)
	dummy42 = np.array(dummy42)
	dummy43 = np.array(dummy43)

	dummy = {}
	dummy['label'] = r'$n = 0$'
	dummy['pk'] = dummy01
	model0['pk'].append(dummy)
	dummy = {}
	dummy['label'] = r'+ $n = 1$'
	dummy['pk'] = dummy02
	model0['pk'].append(dummy)
	dummy = {}
	dummy['label'] = r'+ $n = 2$'
	dummy['pk'] = dummy03
	model0['pk'].append(dummy)

	dummy = {}
	dummy['label'] = r'$n = 0$'
	dummy['pk'] = dummy11
	model1['pk'].append(dummy)
	dummy = {}
	dummy['label'] = r'+ $n = 1$'
	dummy['pk'] = dummy12
	model1['pk'].append(dummy)
	dummy = {}
	dummy['label'] = r'+ $n = 2$'
	dummy['pk'] = dummy13
	model1['pk'].append(dummy)

	dummy = {}
	dummy['label'] = r'$n = 0$'
	dummy['pk'] = dummy21
	model2['pk'].append(dummy)
	dummy = {}
	dummy['label'] = r'+ $n = 1$'
	dummy['pk'] = dummy22
	model2['pk'].append(dummy)
	dummy = {}
	dummy['label'] = r'+ $n = 2$'
	dummy['pk'] = dummy23
	model2['pk'].append(dummy)

	dummy = {}
	dummy['label'] = r'$n = 0$'
	dummy['pk'] = dummy31
	model3['pk'].append(dummy)
	dummy = {}
	dummy['label'] = r'+ $n = 1$'
	dummy['pk'] = dummy32
	model3['pk'].append(dummy)
	dummy = {}
	dummy['label'] = r'+ $n = 2$'
	dummy['pk'] = dummy33
	model3['pk'].append(dummy)

	dummy = {}
	dummy['label'] = r'$n = 0$'
	dummy['pk'] = dummy41
	model4['pk'].append(dummy)
	dummy = {}
	dummy['label'] = r'+ $n = 1$'
	dummy['pk'] = dummy42
	model4['pk'].append(dummy)
	dummy = {}
	dummy['label'] = r'+ $n = 2$'
	dummy['pk'] = dummy43
	model4['pk'].append(dummy)

	list_of_models = []
	list_of_models.append(model0)
	list_of_models.append(model1)
	list_of_models.append(model2)
	list_of_models.append(model3)
	list_of_models.append(model4)
	return list_of_models


def get_model(filename, tag=None):

	kTNS = []
	p0 = []
	p1 = []
	p2 = []
	p3 = []
	p4 = []
	with open(filename, "r") as f:
		for line in f:
			dummy = list(map(float, line.split()))
			if 0.1 > dummy[0] > 0.01:
				kTNS.append(dummy[0])
				p0.append(dummy[1])
				p1.append(dummy[2])
				p2.append(dummy[3])
				p3.append(dummy[4])
				p4.append(dummy[5])

	kTNS = np.array(kTNS)
	p0 = np.array(p0)
	p1 = np.array(p1)
	p2 = np.array(p2)
	p3 = np.array(p3)
	p4 = np.array(p4)

	list_of_TNS_win_models = []
	model = {}
	model['pk'] = p0
	model['k'] = kTNS
	if tag:
		model['label'] = r'TNS model + window %s' % tag
	else:
		model['label'] = r'Kaiser model w/o window'
	model['linestyle'] = '-'
	list_of_TNS_win_models.append(model)
	model = {}
	model['pk'] = p1
	model['k'] = kTNS
	if tag:
		model['label'] = r'TNS model + window %s' % tag
	else:
		model['label'] = r'Kaiser model w/o window'
	model['linestyle'] = '-'
	list_of_TNS_win_models.append(model)
	model = {}
	model['pk'] = p2
	model['k'] = kTNS
	if tag:
		model['label'] = r'TNS model + window %s' % tag
	else:
		model['label'] = r'Kaiser model w/o window'
	model['linestyle'] = '-'
	list_of_TNS_win_models.append(model)
	model = {}
	model['pk'] = p3
	model['k'] = kTNS
	if tag:
		model['label'] = r'TNS model + window %s' % tag
	else:
		model['label'] = r'Kaiser model w/o window'
	model['linestyle'] = '-'
	list_of_TNS_win_models.append(model)
	model = {}
	model['pk'] = p4
	model['k'] = kTNS
	if tag:
		model['label'] = r'TNS model + window %s' % tag
	else:
		model['label'] = r'Kaiser model w/o window'
	model['linestyle'] = '-'
	list_of_TNS_win_models.append(model)
	return list_of_TNS_win_models


def read_Qwindow(filename, cosmo):
	# Read the survey window function
	s = []
	Wlist = [[[],[],[],[],[],[],[],[],[]] for i in range(3)]
	header = False
	with open(filename, 'r') as f:
		for ii, line in enumerate(f):
			if line[:14] == '### header ###':
				header = not header
			elif line[:5] == 'kx_ny':
				dummy = list(map(str, line.split()))
				kmax1 = float(dummy[2])
				kmax2 = float(dummy[5])
				kmax3 = float(dummy[8])
				kmax = min([kmax1, kmax2, kmax3])/1.2
				print("kmax = ", kmax)
			elif not header:
				dummy = list(map(float, line.split()))
				s.append(dummy[0])
				Wlist[0][0].append(dummy[2])
				Wlist[0][1].append(dummy[3])
				Wlist[0][2].append(dummy[4])
				Wlist[0][3].append(dummy[5])
				Wlist[0][4].append(dummy[6])
				Wlist[0][5].append(dummy[7])
				Wlist[0][6].append(dummy[8])
				Wlist[0][7].append(dummy[9])
				Wlist[0][8].append(dummy[10])
				Wlist[1][0].append(dummy[11])
				Wlist[1][1].append(dummy[12])
				Wlist[1][2].append(dummy[13])
				Wlist[1][3].append(dummy[14])
				Wlist[1][4].append(dummy[15])
				Wlist[1][5].append(dummy[16])
				Wlist[1][6].append(dummy[17])
				Wlist[1][7].append(dummy[18])
				Wlist[1][8].append(dummy[19])
				Wlist[2][0].append(dummy[20])
				Wlist[2][1].append(dummy[21])
				Wlist[2][2].append(dummy[22])
				Wlist[2][3].append(dummy[23])
				Wlist[2][4].append(dummy[24])
				Wlist[2][5].append(dummy[25])
				Wlist[2][6].append(dummy[26])
				Wlist[2][7].append(dummy[27])
				Wlist[2][8].append(dummy[28])

	s = np.array(s)
	Wlist = np.array(Wlist)
	# Normalize the window function by r -> 0
	norm0 = 0.
	counter = 0
	for i, value in enumerate(Wlist[0][0]):
		if s[i] > 5 and s[i] < 7:
			norm0 += value
			counter += 1
	norm0 /= counter;

	norm1 = 0.
	counter = 0
	for i, value in enumerate(Wlist[1][0]):
		if s[i] > 5 and s[i] < 7:
			norm1 += value
			counter += 1
	norm1 /= counter;

	norm2 = 0.
	counter = 0
	for i, value in enumerate(Wlist[2][0]):
		if s[i] > 5 and s[i] < 7:
			norm2 += value
			counter += 1
	norm2 /= counter;

	d = cosmo.comoving_distance(redshift)

	for k in range(9):
		Wlist[0][k] /= norm0

	for k in range(9):
		Wlist[1][k] /= norm1*d

	for k in range(9):
		Wlist[2][k] /= norm2*d**2

	# sigma = 1.
	# for i, value in enumerate(Wlist[0][0]):	
	# 	if s[i] < 100:
	# 		Wlist[0][0][i] = (Wlist[0][0][i] - 1.)*np.exp(-0.5*(1. - (s[i] - sigma)/s[i])**2) + 1.
	# 		for k in range(1, 9):
	# 			Wlist[0][k][i] *= np.exp(-0.5*(1. - (s[i] - sigma)/s[i])**2)
	# 		Wlist[1][0][i] = (Wlist[1][0][i] - 1./d)*np.exp(-0.5*(1. - (s[i] - sigma)/s[i])**2) + 1./d
	# 		for k in range(1, 9):
	# 			Wlist[1][k][i] *= np.exp(-0.5*(1. - (s[i] - sigma)/s[i])**2)
	# 		Wlist[2][0][i] = (Wlist[2][0][i] - 1./d**2)*np.exp(-0.5*(1. - (s[i] - sigma)/s[i])**2) + 1./d**2
	# 		for k in range(1, 9):
	# 			Wlist[2][k][i] *= np.exp(-0.5*(1. - (s[i] - sigma)/s[i])**2)

	# from scipy.signal import savgol_filter
	# window_length = 45 # must be odd

	# Wlist[0][0] = savgol_filter(Wlist[0][0], window_length, 3)
	# for k in range(1, 9):
	# 	Wlist[0][k] = savgol_filter(Wlist[0][k], window_length, 3)
	# Wlist[1][0] = savgol_filter(Wlist[1][0], window_length, 3)
	# for k in range(1, 9):
	# 	Wlist[1][k] = savgol_filter(Wlist[1][k], window_length, 3)
	# Wlist[2][0] = savgol_filter(Wlist[2][0], window_length, 3)
	# for k in range(1, 9):
	# 	Wlist[2][k] = savgol_filter(Wlist[2][k], window_length, 3)

	for i, value in enumerate(Wlist[0][0]):	
		if s[i] < 10:
			Wlist[0][0][i] = 1.
			for k in range(1, 9):
				Wlist[0][k][i] = 0.
			Wlist[1][0][i] = 1./d
			for k in range(1, 9):
				Wlist[1][k][i] = 0.
			Wlist[2][0][i] = 1./d**2
			for k in range(1, 9):
				Wlist[2][k][i] = 0.

	filename = filename.rsplit('/', 1)[0] + '/processed_' + filename.rsplit('/', 1)[1]
	filename = filename.replace('wilson', 'pair_counting')
	print("filename = ", filename)
	with open(filename, 'w') as f:
		f.write('# s W00 W10 W20 W30 W40 W50 W60 W70 W80 W01 W11 W21 W31 W41 W51 W61 W71 W81 W02 W12 W22 W32 W42 W52 W62 W72 W82\n')
		for i, value in enumerate(s):
			f.write('%0.16f ' % s[i])
			for j in range(3):
				for k in range(9):
					f.write('%0.16f ' % Wlist[j][k][i])
			f.write('\n')

	return s, Wlist


def read_fft_Qwindow(filename, cosmo):
	# Read the survey window function
	Wlist = [[[],[],[],[],[]] for i in range(3)]
	s = []
	cut = 0
	print("reading ", filename)
	with open(filename, 'r') as f:
		for ii, line in enumerate(f):
			dummy = list(map(float, line.split()))
			if dummy[1] < 0 and cut == 0:
				cut = dummy[0]
			s.append(dummy[0])
			Wlist[0][0].append(dummy[1])
			Wlist[0][1].append(dummy[2])
			Wlist[0][2].append(dummy[3])
			Wlist[0][3].append(dummy[4])
			Wlist[0][4].append(dummy[5])
			Wlist[1][0].append(dummy[6])
			Wlist[1][1].append(dummy[7])
			Wlist[1][2].append(dummy[8])
			Wlist[1][3].append(dummy[9])
			Wlist[1][4].append(dummy[10])
			Wlist[2][0].append(dummy[11])
			Wlist[2][1].append(dummy[12])
			Wlist[2][2].append(dummy[13])
			Wlist[2][3].append(dummy[14])
			Wlist[2][4].append(dummy[15])

	s = np.array(s)
	Wlist = np.array(Wlist)
	print("cut = ", cut)

	d = cosmo.comoving_distance(redshift)
	print("d = ", d)

	Wlist[0][1] /= Wlist[0][0][0]
	Wlist[0][2] /= Wlist[0][0][0]
	Wlist[0][3] /= Wlist[0][0][0]
	Wlist[0][4] /= Wlist[0][0][0]
	Wlist[0][0] /= Wlist[0][0][0]
	Wlist[1][1] /= Wlist[1][0][0]*d
	Wlist[1][2] /= Wlist[1][0][0]*d
	Wlist[1][3] /= Wlist[1][0][0]*d
	Wlist[1][4] /= Wlist[1][0][0]*d
	Wlist[1][0] /= Wlist[1][0][0]*d
	Wlist[2][1] /= Wlist[2][0][0]*d**2
	Wlist[2][2] /= Wlist[2][0][0]*d**2
	Wlist[2][3] /= Wlist[2][0][0]*d**2
	Wlist[2][4] /= Wlist[2][0][0]*d**2
	Wlist[2][0] /= Wlist[2][0][0]*d**2

	# for i, value in enumerate(Wlist[0][0]):	
	# 	if s[i] < 10:
	# 		Wlist[0][0][i] = 1.
	# 		for k in range(1, 5):
	# 			Wlist[0][k][i] = 0.
	# 		Wlist[1][0][i] = 1./d
	# 		for k in range(1, 5):
	# 			Wlist[1][k][i] = 0.
	# 		Wlist[2][0][i] = 1./d**2
	# 		for k in range(1, 5):
	# 			Wlist[2][k][i] = 0.

	filename = filename.rsplit('/', 1)[0] + '/processed_' + filename.rsplit('/', 1)[1]
	print("write to file ", filename)
	with open(filename, 'w') as f:
		f.write('# s W00 W10 W20 W30 W40 W01 W11 W21 W31 W41 W02 W12 W22 W32 W42\n')
		for i, value in enumerate(s):
			f.write('%0.16f ' % s[i])
			for j in range(3):
				for k in range(5):
					if s[i] > cut:
						Wlist[j][k][i] = 0.
					f.write('%0.16f ' % Wlist[j][k][i])
			f.write('\n')
	return s, Wlist


def read_nbar(filename):
	# Read the survey window function
	nbar = {}
	nbar['value'] = []
	nbar['z'] = []
	print("reading ", filename)
	with open(filename, 'r') as f:
		for ii, line in enumerate(f):
			dummy = list(map(float, line.split()))
			nbar['z'].append(dummy[0])
			nbar['value'].append(dummy[1])
	print("nbar['z'] = ", nbar['z'])
	return interp1d(np.array(nbar['z']), np.array(nbar['value']), fill_value="extrapolate")
