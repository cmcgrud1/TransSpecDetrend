import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import sys
import utils
import pickle 

def LoadTransSpec_ScipyOpt(full_path, Pkl_file, Bins_base_name ='Bin'):
	if full_path[-1] == '/':
		full_path = full_path[:-1]
	Bins = glob.glob(full_path+"/*/") #to get the bin subdirectories

	#load the orginal pickle file to get the wavelength range of the bins
	with open(Pkl_file, 'rb') as pkl_file:
		LC_ob = pickle.load(pkl_file, encoding="latin1")
	pkl_Bins= LC_ob['wbins']

	#to sort bins and throw out subdirectories without the proper base name
	binNum, base_len = [], len(Bins_base_name)
	for b in Bins:
		subfold = b.split('/')[-2]
		if subfold[:base_len] == Bins_base_name: #just to make sure it's the right subdirectory
			binNum.append(int(subfold[base_len:])) #to keep track of the bin number in the same order of the glob files
	Sorted = np.argsort(binNum)
	Bins = list(np.array(Bins)[Sorted])

	#to get the best fit values for each binn and print its final rp/rs and best fit stats
	contents, Bcnt = '', 0
	for B in Bins:
		print ("Bin"+str(Bcnt))
		if not os.path.isfile(B+'successful_fits_results_tab.txt'): #there should at least be a successful.txt file, even if the code is not done for that bin
			sys.exit("missing 'successful_fits_results_tab.txt' for "+B+"!!!!")
		else:
			##### To check if the results have already been made for this dataset
			with open(B+'successful_fits_results_tab.txt', 'r') as succf_results:
				lines = succf_results.readlines()
				last_lines = lines[-5:]
				if last_lines[0][:20] == 'BEST REDUCED CHI2 = ': #check if the best fits are written at the end of the file
					results = last_lines[0].split(';')
					best_chi2,best_model = results[0][20:], results[1][14:]
					line_cnt, start_line = 0, None
					for line in lines:
						#len('Reduced chi^2 = '+str(best_chi2)) because not sure how many characters are actually in a line. think there's an imposing '\n' that's making the lines not match
						if line[:len('Reduced chi^2 = '+str(best_chi2))] == 'Reduced chi^2 = '+str(best_chi2): #found the line where the best fit info is
							Bck_up, Bck_cnt = True, 0
							while Bck_up: #found relevant lines of the best fit parameters
								if lines[line_cnt-Bck_cnt][:22] == 'Fitted coefficients = ':
									start_line = line_cnt-Bck_cnt
									Bck_up = False #since the start line is the last thing that will be found, end while loop here
								Bck_cnt +=1
							break
						line_cnt+=1
				else:
					print ("final results for "+B+" isn't found.... yet")
		datastrt = lines[start_line][:-1].index('[')
		data = lines[start_line][:-1][datastrt+1:]
		#two ways to split the data
		if len(data.split('  ')) > 1: #then each parameter is deliminated by 2 spaces
			RpRs = float(lines[start_line][24:-1].split('  ')[2])
		else: #otherwise each parameter is deliminated by 1 space
			RpRs = float(lines[start_line][24:-1].split(' ')[2])
		print ("RpRs="+str(RpRs), " best_model:",best_model, " best_chi2:",best_chi2)
		contents+= str(pkl_Bins[Bcnt][0])+'\t\t'+str(pkl_Bins[Bcnt][1])+'\t\t'+str(RpRs)+'\t\t'+best_chi2+'\t\t'+best_model
		Bcnt +=1
		print ("\n")
	results = open(full_path+'/transSpec.dat', 'w')
	results.write("#WavDwn\t\tWavUp\t\t\tRpRs\t\t\tchi^2\t\t\tBest fit\n")
	results.write(contents[:-1]) #to get rid of the last new line
	results.close()
	return None

def LoadTransSpec_BMA(full_path, Pkl_file, Bins_base_name ='Bin', ResultFile = 'BMA_final_results.txt', OutputFile ='transSpec.dat', Depth_name = 'Rp/Rs'): #To create the transpec files
	#full_path == path right before the BinX folders
	if full_path[-1] == '/':
		full_path = full_path[:-1]
	Bins = glob.glob(full_path+"/*/") #to get the bin subdirectories

	#load the orginal pickle file to get the wavelength range of the bins
	with open(Pkl_file, 'rb') as pkl_file:
		LC_ob = pickle.load(pkl_file, encoding="latin1")
	pkl_Bins= LC_ob['wbins']
	
	#to sort bins and throw out subdirectories without the proper base name
	binNum, base_len = [], len(Bins_base_name)
	for b in Bins:
		subfold = b.split('/')[-2]
		if subfold[:base_len] == Bins_base_name: #just to make sure it's the right subdirectory
			binNum.append(int(subfold[base_len:])) #to keep track of the bin number in the same order of the glob files
	Sorted = np.argsort(binNum)
	Bins = list(np.array(Bins)[Sorted])

	#to get the best fit values for each binn and print its final rp/rs and best fit stats
	contents = '#WavDwn\t\tWavUp\t\t\tRpRs\t\t\tupper\t\t\tlower\n'
	Bcnt = 0
	for B in Bins:
		print (Bins_base_name+str(Bcnt))
		if not os.path.isfile(B+ResultFile):
			print ("final results for "+B+" isn't found.... yet")
		else:
			Names, fits = utils.LoadResultsTxt(B+ResultFile, deliminator='\t', noSpaces_tabs=True)
			RpRs_idx = Names.index(Depth_name)
			contents += str(pkl_Bins[Bcnt][0])+'\t\t'+ str(pkl_Bins[Bcnt][1])+'\t\t'+str(fits[RpRs_idx][0])+'\t\t'+str(fits[RpRs_idx][1])+'\t\t'+str(fits[RpRs_idx][2])+'\n'
			print ("WavRange="+str(pkl_Bins[Bcnt][0])+'-'+ str(pkl_Bins[Bcnt][1]), "RpRs="+str(fits[RpRs_idx][0]), " upper:",str(fits[RpRs_idx][1]), " lower:",str(fits[RpRs_idx][2]),'\n')
		Bcnt +=1
	results = open(full_path+'/'+OutputFile, 'w')
	results.write(contents[:-1]) #[:-1] to get rid of the last new line
	print ("created the '"+OutputFile+"' file in folder '"+full_path+"'")
	results.close()
	return None
if __name__ == "__main__":
	# LoadTransSpec_ScipyOpt('CMC_parameteric_modelR', 'FORS2/LC_FOR2_R.pkl')
	path = '/home/mcgruderc/pool/CM_GradWrk/GPTransmissionSpectra3/'
	base1, base2 = path+'outputs/WASP-96/', path+'WASP-96/'
	LoadTransSpec_BMA(base1+'FORS2_180B_noTel/Binned', base2+'FORS2_180B_noTel.pkl')
	LoadTransSpec_BMA(base1+'FORS2_180R_noTel/Binned', base2+'FORS2_180R_noTel.pkl')
	LoadTransSpec_BMA(base1+'LCs_w96_180ob1_noTel/Binned/', base2+'LCs_w96_180ob1_noTel.pkl')
	LoadTransSpec_BMA(base1+'LCs_w96_180ob2_noTel/Binned/', base2+'LCs_w96_180ob2_noTel.pkl')

