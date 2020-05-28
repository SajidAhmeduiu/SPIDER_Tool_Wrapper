#!/usr/bin/env python
import numpy, os, sys

def read_pssm(pssm_file,pssm_directory,pssm_counter):
	# this function reads the pssm file given as input, and returns a LEN x 20 matrix of pssm values.

	# index of 'ACDE..' in 'ARNDCQEGHILKMFPSTWYV'(blast order)
	#this is a modification by Sajid
	#This program sometimes runs without actually giving any output
	#So, I am putting this print statement so that it can be tracked
	#whether the program has actually run or not
	print "Generating predictions for PSSM: " + str(pssm_counter)
	idx_res = (0, 4, 3, 6, 13, 7, 8, 9, 11, 10, 12, 2, 14, 5, 1, 15, 16, 19, 17, 18)

	# open the two files, read in their data and then close them
	if pssm_file == 'STDIN': fp = sys.stdin
	else: fp = open(pssm_directory+'/'+pssm_file, 'r')
	lines = fp.readlines()
	fp.close()

	# declare the empty dictionary with each of the entries
	aa = []
	pssm = []
	
	#print lines

	# iterate over the pssm file and get the needed information out
	for line in lines:
		split_line = line.split()
		# valid lines should have 32 points of data.
		# any line starting with a # is ignored
		try: int(split_line[0])
		except: continue

		if line[0] == '#': continue

		aa_temp = split_line[1]
		aa.append(aa_temp)
		if len(split_line) in (44,22):
			pssm_temp = [-float(i) for i in split_line[2:22]]
		elif len(line) > 70:  # in case double digits of pssm
			pssm_temp = [-float(line[k*3+9: k*3+12]) for k in range(20)]
			pass
		else: continue
		pssm.append([pssm_temp[k] for k in idx_res])
	
	return aa, pssm


def get_input(pssm_file,pssm_directory,pssm_counter):
	# this function takes a path to a pssm file and finds the pssm + phys 7 input
	# to the NN in the required order - with the required window size (8).


	# define the dictionary with the phys properties for each AA
	phys_dic = {'A': [-0.350, -0.680, -0.677, -0.171, -0.170, 0.900, -0.476],
							'C': [-0.140, -0.329, -0.359, 0.508, -0.114, -0.652, 0.476],
							'D': [-0.213, -0.417, -0.281, -0.767, -0.900, -0.155, -0.635],
							'E': [-0.230, -0.241, -0.058, -0.696, -0.868, 0.900, -0.582],
							'F': [ 0.363, 0.373, 0.412, 0.646, -0.272, 0.155, 0.318],
							'G': [-0.900, -0.900, -0.900, -0.342, -0.179, -0.900, -0.900],
							'H': [ 0.384, 0.110, 0.138, -0.271, 0.195, -0.031, -0.106],
							'I': [ 0.900, -0.066, -0.009, 0.652, -0.186, 0.155, 0.688],
							'K': [-0.088, 0.066, 0.163, -0.889, 0.727, 0.279, -0.265],
							'L': [ 0.213, -0.066, -0.009, 0.596, -0.186, 0.714, -0.053],
							'M': [ 0.110, 0.066, 0.087, 0.337, -0.262, 0.652, -0.001],
							'N': [-0.213, -0.329, -0.243, -0.674, -0.075, -0.403, -0.529],
							'P': [ 0.247, -0.900, -0.294, 0.055, -0.010, -0.900, 0.106],
							'Q': [-0.230, -0.110, -0.020, -0.464, -0.276, 0.528, -0.371],
							'R': [ 0.105, 0.373, 0.466, -0.900, 0.900, 0.528, -0.371],
							'S': [-0.337, -0.637, -0.544, -0.364, -0.265, -0.466, -0.212],
							'T': [ 0.402, -0.417, -0.321, -0.199, -0.288, -0.403, 0.212],
							'V': [ 0.677, -0.285, -0.232, 0.331, -0.191, -0.031, 0.900],
							'W': [ 0.479, 0.900, 0.900, 0.900, -0.209, 0.279, 0.529],
							'Y': [ 0.363, 0.417, 0.541, 0.188, -0.274, -0.155, 0.476]}

	aa, pssm = read_pssm(pssm_file,pssm_directory,pssm_counter)


	# set the phys7 data.
	phys = [phys_dic.get(i, phys_dic['A']) for i in aa]

	return(pssm, aa, phys)

def window(feat, winsize=8):
	# apply the windowing to the input feature
	feat = numpy.array(feat)
	output = numpy.concatenate([numpy.vstack([feat[0]]*winsize), feat])
	output = numpy.concatenate([output, numpy.vstack([feat[-1]]*winsize)])
	output = [numpy.ndarray.flatten(output[i:i+2*winsize+1]).T for i in range(0,feat.shape[0])]
	return output

def window_data(*feature_types):
	n = len(feature_types[0])
	features = numpy.empty([n,0])

	for feature_type in feature_types:
		test = numpy.array(window(feature_type))
		features = numpy.concatenate((features, test), axis=1)

	return features


def sigmoid(input):
	# apply the sigmoid function
	output = 1 / (1 + numpy.exp(-input))
	return(output)

def nn_feedforward(nn, input):
	input = numpy.matrix(input)

	# find the number of layers in the NN so that we know how much to iterate over
	num_layers = nn['n'][0][0][0][0]
	# num_input is the number of input AAs, not the dimentionality of the features
	num_input = input.shape[0]
	x = input

	# for each layer up to the final
	for i in range(1,num_layers-1):
		# get the bais and weights out of the nn
		W = nn['W'][0][0][0][i-1].T
		temp_size = x.shape[0]
		b = numpy.ones((temp_size,1))
		x = numpy.concatenate((b, x),axis=1)
		# find the output of this layer (the input to the next)
		x = sigmoid(x * W)

	# for the final layer.
	# note that this layer is done serpately, this is so that if the output nonlinearity
	# is not sigmoid, it can be calculated seperately.
	W = nn['W'][0][0][0][-1].T
	b = numpy.ones((x.shape[0],1))
	x = numpy.concatenate((b, x),axis=1)
	output = x*W
	pred = sigmoid(output)

	return pred


def load_NN(nn_filename):
	#this is a modification by Sajid
	#there was an error in loading the .npz files
	#which stated that allow_Pickle was set to False
	return numpy.load(nn_filename,allow_pickle=True)

	# load in the NN mat file.
#	import scipy.io
#	mat = scipy.io.loadmat(nn_filename)

#	nn = mat['nn']
	#this is a modification by Sajid
	#no need to have this redundant return statement 
	#since the loaded file is already being returned by 
	#the return statement above
	#return nn

dict_ASA0 = dict(zip("ACDEFGHIKLMNPQRSTVWY",
					(115, 135, 150, 190, 210, 75, 195, 175, 200, 170,
		185, 160, 145, 180, 225, 115, 140, 155, 255, 230)))
def run_iter(dict_nn, input_feature0, aa, ofile):
	SS_order = ('C' 'E' 'H')
	list1 = ('SS', 'ASA', 'TTPP')
	list_res1 = []
	for x in list1:
		nn = dict_nn[x]
		norm_max = nn['high'][0][0][0]
		norm_min = nn['low'][0][0][0]
		input_feature1 = (input_feature0 - numpy.tile(norm_min, (input_feature0.shape[0], 1))) / numpy.tile((norm_max - norm_min), (input_feature0.shape[0],1))
		r1 = nn_feedforward(nn, input_feature1)
		list_res1.append(r1)

	pred_ss_1, pred_asa_1, pred_ttpp_1 = list_res1
	
	#this is a modification by Sajid
	#I have converted the numpy matrixes to numpy array in the lines below for avoiding 
	#any unexpected errors while trying to use the numpy specific functions
	#print type(pred_ss_1)
	
	#this is a modification by Sajid
	pred_ss_1 = numpy.asarray(pred_ss_1)
	pred_asa_1 = numpy.asarray(pred_asa_1)
	pred_ttpp_1 = numpy.asarray(pred_ttpp_1)
	
	SS_1 = [SS_order[i] for i in numpy.argmax(pred_ss_1,1)]
	pred_ttpp_1_denorm = (pred_ttpp_1 - 0.5) * 2
	theta = numpy.degrees(numpy.arctan2(pred_ttpp_1_denorm[:,0], pred_ttpp_1_denorm[:,2]))
	tau = numpy.degrees(numpy.arctan2(pred_ttpp_1_denorm[:,1], pred_ttpp_1_denorm[:,3]))
	phi = numpy.degrees(numpy.arctan2(pred_ttpp_1_denorm[:,4], pred_ttpp_1_denorm[:,6]))
	psi = numpy.degrees(numpy.arctan2(pred_ttpp_1_denorm[:,5], pred_ttpp_1_denorm[:,7]))

	if ofile == 'NULL':
		return SS_1, pred_ss_1, pred_asa_1, theta, tau, phi, psi

	fp = open(ofile, 'w')
	print >>fp, '#\tAA\tSS\tASA\tPhi\tPsi\tTheta(i-1=>i+1)\tTau(i-2=>i+1)\tP(C)\tP(E)\tP(H)'
	for ind, x in enumerate(aa):
		asa = pred_asa_1[ind] * dict_ASA0.get(x, dict_ASA0['A'])
		#print ind+1, aa[ind], SS_1[ind], pred_ss_1[ind,0], pred_ss_1[ind,1], pred_ss_1[ind,2], pred_asa_1[ind], theta[ind], tau[ind], phi[ind], psi[ind]
		print >>fp, ('%i\t%c\t%c\t%5.1f' + '\t%6.1f'*4 + '\t%.3f'*3) % (ind+1, x, SS_1[ind], asa, phi[ind], psi[ind], theta[ind], tau[ind], pred_ss_1[ind,0], pred_ss_1[ind,1], pred_ss_1[ind,2])
	fp.close()

	return SS_1, pred_ss_1, pred_asa_1, theta, tau, phi, psi


def main(pssm_file,directory_to_save_spd3_files,pssm_directory,pssm_counter):

	basenm = os.path.basename(pssm_file)
	if basenm.endswith('.pssm'): basenm = basenm[:-5]
	elif basenm.endswith('.mat'): basenm = basenm[:-4]

	if os.path.isfile(directory_to_save_spd3_files+'/'+basenm+'.spd3'): return
	open(directory_to_save_spd3_files+'/'+basenm+'.spd3', 'w').close()

	SS_order = ('C' 'E' 'H')

	pssm, aa, phys = get_input(pssm_file,pssm_directory,pssm_counter)

	list_nn = (dict1_nn, dict2_nn, dict3_nn)
	input_feature = window_data(pssm, phys)

	## DO FIRST PREDICTIONS
	for it1 in (1, 2, 3):
		ofile = directory_to_save_spd3_files+'/'+basenm+'.spd%d' % it1
		if not ball and it1<3: ofile = 'NULL'

		dict_nn = list_nn[it1-1]
		res1 = run_iter(dict_nn, input_feature, aa, ofile)
		SS_1, pred_ss_1, pred_asa_1, theta, tau, phi, psi = res1
	
		## feature after 1st iteration
		#this is a modification by Sajid
		#these arrays were 1 dimensional which was giving error 
		#while concatenating on axis 1 since there is no axis 1 for 
		#1 dimensional arrays
		theta = theta.reshape((len(theta),1))
		tau = tau.reshape((len(tau),1))
		
		tt_input = numpy.sin(numpy.concatenate((numpy.radians(theta), numpy.radians(tau)), axis=1))/2 + 0.5
		tt_input = numpy.concatenate((tt_input, numpy.cos(numpy.concatenate((numpy.radians(theta), numpy.radians(tau)), axis=1))/2 + 0.5), axis=1)
		
		#this is a modification by Sajid
		#these arrays were 1 dimensional which was giving error 
		#while concatenating on axis 1 since there is no axis 1 for 
		#1 dimensional arrays
		phi = phi.reshape((len(phi),1))
		psi = psi.reshape((len(psi),1))
		
		pp_input = numpy.sin(numpy.concatenate((numpy.radians(phi), numpy.radians(psi)), axis=1))/2 + 0.5
		pp_input = numpy.concatenate((pp_input, numpy.cos(numpy.concatenate((numpy.radians(phi), numpy.radians(psi)), axis=1))/2 + 0.5), axis=1)
		ttpp_input = numpy.concatenate((tt_input, pp_input), axis=1)
		input_feature = window_data(pssm, phys, pred_ss_1, pred_asa_1, ttpp_input)
	return

def list_of_files(dir_name):
    return (f for f in os.listdir(dir_name))

if __name__ == "__main__":
	# if there is no filename for the features to be written to given as input, don't write it
	if len(sys.argv) < 3:
		print >>sys.stderr, "Usage: RUN *.pssmfile"
		sys.exit()

	ball = '-all' in sys.argv

	#this is a modification by Sajid
	#nndir was not able to locate the correct directory
	#which was giving the error
	#However, the nndir now has to be specified
	#manually when the SPIDER tool is kept in
	#a different directory
	#nndir = '/home/learning/Run SPIDER/SPIDER_M'
	
	#if os.path.isfile(nndir+'/dat/pp1.npz'): nndir += '/dat/'
	#elif os.path.isfile(nndir+'/../dat/pp1.npz'): nndir += '/../dat/'
	dict1_nn = load_NN('dat/pp1.npz')
	dict2_nn = load_NN('dat/pp2.npz')
	dict3_nn = load_NN('dat/pp3.npz')
	
	#You have to send the directory where the PSSM files are stored as the first #argument 
	pssm_directory = sys.argv[1]
	pssm_files = list_of_files(pssm_directory)
	
	#You have to send the directory where you want to save the .spd3 files as the #second argument
	directory_to_save_spd3_files = sys.argv[2] 
	
	pssm_counter  = 1

	for x in pssm_files:
		main(x,directory_to_save_spd3_files,pssm_directory,pssm_counter)
		pssm_counter = pssm_counter + 1