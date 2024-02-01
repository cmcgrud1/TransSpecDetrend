import numpy as np
import sys
import os
from scipy.io import readsav
import time
import Run_BMA_Parametric_mthread as runBMA
import pickle
import corner
import matplotlib.pyplot as plt
import utils
import glob
import time 

def ClearUnusedBMAs(Folder_Bin, ResultsFile='BMA_final_results.txt', diff='###Total_samples=', deliminator='\t'): #to delete folders that have 0 weight in the BMA to form the final LC
    #This is done for a given bin
    import shutil
    #diff = the line, where every line following it, will be information about which variants are used
    ToKeep = []
    if os.path.isfile(Folder_Bin+'/'+ResultsFile): #if have the results info for this bin
        txt = open(Folder_Bin+'/'+ResultsFile, 'r')
        Useful =False #to keep track of which lines are useful
        for Line in txt: #to read first line of file, which contains the string titles of data
            if Line[:len(diff)] == diff: #everything below is important
                Useful=True
                continue
            elif Useful:
                if Line[-1] == deliminator:#if the last string of the line is the deliminator
                    Line = Line[:-1] #get rid of it
                UsedVariants = Line.split(deliminator) #To split the list of used variants
                for var in UsedVariants:
                    var = var.replace('#', '') #to get rid of any pesky commments
                    var = var.replace(' ', '') #to get ride of any pesky spaces
                    var = var.replace('\t', '') #to get ride of any pesky tabs
                    try:
                        name, cnt = var.split('=') #to split the var name and the count
                    except:
                        if var == '\n':
                            pass
                        else:
                            print ("problem reading in var name!!!")
                            print ("var.split('='):", var.split('='))
                            cont = input('Do you still want to continue (y/n)? \n  >    ')
                            if cont.lower() == 'y':
                                pass
                            else:
                                sys.exit()
                    ToKeep.append(Folder_Bin+'/'+name) #what not to toss
        Variants = glob.glob(Folder_Bin+'/*')
        print ("For '", Folder_Bin+"'")
        print ("ToKeep:")
        for K in ToKeep:
            print (K)
        RemoveCnt = 0
        for V in Variants:
            if os.path.isdir(V) and V not in ToKeep:
                # print ("removing folder '"+V+"'")
                shutil.rmtree(V, ignore_errors=True)
                RemoveCnt+=1
        print ("Removed ", RemoveCnt, "files\n")
    else:
        print ("Couldn't find '"+ResultsFile+"' file, skipping folder ...")
        return False
    return True

def CheckRepeats(name, Del_dups=False): #to go through a file and see if there is a line that repeats
    Identical_lines, txt, prim_idx = [], open(name, 'r'), 0
    dup_indx = [] #need to keep track of which index has a duplicate, so don't re-add that index and count the duplicate twice
    for prim_line in txt: #1st save a line
        txt1, sec_idx = open(name, 'r'), 0 #have to reopen file everytime to search through each line
        for sec_lin in txt1: #now go through all lines again
            if sec_idx == prim_idx: #don't need to check the same line
                pass
            elif sec_lin != prim_line:
                pass
            else: #If we found an identical line!!!!
                if sec_idx in dup_indx: #to make sure not already included this duplicate
                    sec_idx +=1
                    continue
                Identical_lines.append(sec_lin), dup_indx.append(prim_idx)
            sec_idx +=1
        txt1.close()
        prim_idx +=1
    print (len(Identical_lines), " Duplicate lines found:", Identical_lines, "corresponding to the following lines:", dup_indx)
    if Del_dups: #to delete all the found Duplicates
        print ("Deleting all duplicates")
        utils.RemoveLine(name, list(np.array(dup_indx)+1)) #gotta add 1 to all indeces, becuase RemoveLine starts with index 1
    return None

def Finalize(Tduration, Epars_names, Bins, Max_P_order, Path, Cleaned_Bins=None,Clean=False, PLOTFinReslts=True, WriteBestFit=True, RunShorts=False):
    """To plot the final results (BMA results) for each bin. This must have it's own script: 
    1) because plotting has to be as a job submission
    2) because the variant submission is rolling, there is no way to know when a specific Bin is done, unless check it periodically"""
    #Bins == list of 2 element list, where each element are the wavelength bounds, #Epars_names = list of used external parameters
    #Cleaned_Bins = A list of True/False, to keep track of which bins have been cleaned #Only needed 'Clean' flag is on.
    #RunShorts = To still create the final BMA file, even though one of the variants didn't finish. HOWEVER, DON'T clean that bin
    if not Cleaned_Bins or not Clean: #if Doesn't pass in a Clean list, make one
        Cleaned_Bins = [False]*len(Bins)
    if Path[-1] == '/':
        Path = Path[:-1] #to get rid of excessive back slashes, because assuming there is no backslash after Path
    for b in range(len(Bins)):
        if Cleaned_Bins[b]: #if labeled as already cleaned
            continue #Don't go through process if already cleaned this bin
        elif os.path.isfile(Path+'/'+'Bin'+str(b)+'/BMA_final_results.txt'): #or if find the results file, then don't have to run processing for this bin
            if Clean: #but still might not have cleaned the bin
                ClearUnusedBMAs(Path+'/'+'Bin'+str(b)) #just run the clean directory routine. If it's already cleaned, then nothing will happen
            Cleaned_Bins[b] = True #then must've already been cleaned
            continue #and don't need to go through process
        elif not os.path.exists(Path+'/'+'Bin'+str(b)): #or if can't find the bin folder
            print ("The folder for "+Path+'/'+'Bin'+str(b)+" hasn't even been created yet! skipping ...")
            print("##########################\n#########################\n")
            continue #then skip this and come back to it later, Clean_Bins[b] stays False
        elif not os.path.isfile(Path+'/'+'Bin'+str(b)+'/Global_LnZs.txt'):  #or if can't find the Global_LnZs.txt file
            print ("The 'Global_LnZs.txt' file for Bin"+str(b)+" hasn't even been created yet! skipping ...")
            print("##########################\n#########################\n")
            continue  #then skip this and come back to it later, Clean_Bins[b] stays False           
        else: #lastly if not labeled Clean, don't find the final results file, and started working on this Bin, check to see if we are done
            Model, Results = utils.LoadResultsTxt(Path+'/'+'Bin'+str(b)+'/Global_LnZs.txt', deliminator='\t', noSpaces_tabs=True)
            Variants = utils.getVariants(Epars_names, Max_P_order) #### 1st to get all the different combinations of External parameters and polynomial fit for each variant

            Total_Models2b_tested, models_already_test = len(Variants), len(Model)
            if models_already_test > Total_Models2b_tested: #this is NEVER supposed to happen
                print ("ERROR!!!!\nFound",str(models_already_test),"models in 'Global_LnZs.txt' file for Bin"+str(b))
                print ("However, expected only", str(Total_Models2b_tested),"\nERROR!!!!\n\n")
                CheckRepeats(Path+'/'+'Bin'+str(b)+'/Global_LnZs.txt', Del_dups=True) #try to correct mistake here
                Model, Results = utils.LoadResultsTxt(Path+'/'+'Bin'+str(b)+'/Global_LnZs.txt', deliminator='\t', noSpaces_tabs=True)
                models_already_test = len(Model) #and run through it again
            print ("for Bin"+str(b))
            if models_already_test < Total_Models2b_tested: #if this is the case, then still waiting on some jobs to finish
                if not RunShorts:
                    print ("for Bin"+str(b)+"\nStill waiting on "+str(Total_Models2b_tested-models_already_test)+" models to finish\n\n")
                    continue #check the other submission. Will come bck
                else: #if tell code to, run the final results, regardless of missing variants, do it here
                   print ("making final results, anyways... ")
                   file = open(Path+'/'+'Bin'+str(b)+"/Offset0/needed_data.pkl","rb") #since the Light curve and transit prior information is the same per bin, just use the Offset0 subdirectory to get that information
                   needed_info = pickle.load(file) 
                   runBMA.Results(Path+'/'+'Bin'+str(b), needed_info['LC'], needed_info['transPar'], Tduration, write_bestFit=WriteBestFit, PLOT=PLOTFinReslts) 
            if Total_Models2b_tested == models_already_test: # then finished all runs for this bin      
                file = open(Path+'/'+'Bin'+str(b)+"/Offset0/needed_data.pkl","rb") #since the Light curve and transit prior information is the same per bin, just use the Offset0 subdirectory to get that information
                needed_info = pickle.load(file) 
                file.close() #below is how you make the BMA_final_results.txt file needed to clear the directory
                runBMA.Results(Path+'/'+'Bin'+str(b), needed_info['LC'], needed_info['transPar'], Tduration, write_bestFit=WriteBestFit, PLOT=PLOTFinReslts) #to print the final results. HOWEVER! Not plotting it, because Hydra can't plot in terminal
                if Clean:
                    ClearUnusedBMAs(Path+'/'+'Bin'+str(b))
                print ("\n\n")
                Cleaned_Bins[b] = True
    return Cleaned_Bins
  

def Loop_Finalize(path_base, path2pkl, Epars_names, trans_dur, Max_P_order, add_subDirs='/Binned', wait_T=120, run_shorts=False, pklBaseName='FinalLC_'): #to loop the Finalize() function so constantly tries to clear out directories
    #add_subDirs == additional subdirectory name(s), where data of interest are located
    start_time = time.time()
    #To initiate the dictionary that keep track of which bins of a given iteration has been cleaned
    KeepGoing, indx, iterations = True, 0, []
    while KeepGoing:
        path_i = ''
        if type(path_base) == list: #then there are more than one counting portions to get to the right directory
            for pb in path_base:
                path_i += pb+str(indx)+'/'
        else: #otherwise assuming only one index needed to get to right directory
            path_i = path_base+str(indx)+'/'
        indx +=1
        if not os.path.exists(path_i): #if this path doesn't exists, then assume going through all iterations
            KeepGoing = False
            continue
        iterations.append(path_i) #this method automatically sorts the data

    CleanedBins = {} #to keep track of all the bins that have been clean for a specific Iteration
    for i in range(len(iterations)):
        file = open(path2pkl+str(i)+"/"+pklBaseName+str(i)+".pkl","rb") 
        rawLC = pickle.load(file)
        file.close()
        Bins = rawLC['wbins']
        CleanedBins['Iteration'+str(i)] = [False]*len(Bins)
    #To correct for bug that doesn't write everything correctly in Global_LnZ AND print/plot the final results AND delete unused variants to clear space
    moreWork = True
    while moreWork: #keep on looping through each iteration/each bin while there are still uncleaned variants
        iter_cnt, moreWork = 0, False #initially, assuming finish all bins in all iterations
        for I in iterations: #to repeat per iteration
            print ('\033[1m For iteration'+str(iter_cnt),"\033[0m")
            file = open(path2pkl+str(iter_cnt)+"/"+pklBaseName+str(iter_cnt)+".pkl","rb")
            rawLC = pickle.load(file)
            file.close()
            Bins = rawLC['wbins']
            for b in range(len(Bins)):
                if CleanedBins['Iteration'+str(iter_cnt)][b] == True:
                   continue #no need to run  Add_missing(), if already done it before
                elif os.path.isfile(I+add_subDirs+'/Bin'+str(b)+'/Global_LnZs.txt'):
                    print ("for bin"+str(b))
                    runBMA.Add_missing(I+add_subDirs+'/Bin'+str(b), Epars_names, Max_P_order, WriteBestFit=False, JustCnt=False, RunResults=False, PrintVariants=False, Rm_Unfinished=False, ToPlot=False, DONTrunResults=True) #To fix for the Bug that doesn't fully write some of the variant info in the Global_LnZ files
                else:
                    print ("'Global_LnZs.txt' file hasn't been created yet. Skipping Bin"+str(b))
                    moreWork = True
            print ('\n')
            CleanedBins['Iteration'+str(iter_cnt)] = Finalize(trans_dur, Epars_names, Bins, Max_P_order, I+add_subDirs, Cleaned_Bins=CleanedBins['Iteration'+str(iter_cnt)], Clean=True, PLOTFinReslts=True, WriteBestFit=True, RunShorts=run_shorts)
            if not all(CleanedBins['Iteration'+str(iter_cnt)]): #if there is one case where there's a bin that hasn't been cleaned
                moreWork = True #loop back through all the iterations again
            iter_cnt +=1
        print ("waiting for ", wait_T, "seconds")
        time.sleep(wait_T) #just wait for a short moment to get barings #PUT THIS ON THE TOP of while loop!!!!
    print ('Finished! runtime='+str((time.time()-start_time)/3600.0)+'hrs')
    return None

def ReadFinalResults_wl(out, transPar_names, PlotList, TrueValues, method, results_file, Iteration_base, PrintFits=False, PlotFits=True):
    #Iteration_base=list of all the path info to the results files that needs an index right after it. Assuming that the only thing missing to combine each element in the list is an index
    Results, Uncertainties = {}, {}
    for k in PlotList:
        Results[k], Uncertainties[k] = np.array([]), np.array([])

    for i in range(0,50):
        Path =''
        for b in Iteration_base:
            Path += b+str(i)
        Path += '/'+out+'/'
        names, results = utils.LoadResultsTxt(Path+results_file, deliminator='\t', noSpaces_tabs=True)
        if PrintFits:
            print ("\n\n\nparam\ttrue_value\tfit_value\tabs_diff\t\tuncertainty\tsig_diff")
        for n in range(len(names)):
            if PrintFits: #for printing
                if names[n] in transPar_names: #don't care about axuliary parameters
                    IDX = transPar_names.index(names[n])
                    diff = TrueValues[IDX]-results[n][0]
                    sig_diff = 0
                    #To print how consitent fitted results are to true value
                    if diff > 0:#then the true value is higher, so need to know how many upper bounds away we are from the true result
                        new_diff = 1
                        while new_diff > 0:
                            sig_diff +=1
                            uncertainty = results[n][1]
                            new_diff = TrueValues[IDX]-(results[n][0]+(sig_diff*uncertainty))
                    else: #then the true value is lower, so need to know how many lower bounds away we are from the true result
                        new_diff = -1
                        while new_diff < 0:
                            sig_diff +=1
                            uncertainty = results[n][2]
                            new_diff = TrueValues[IDX]-(results[n][0]-(sig_diff*uncertainty))
                    print (names[n]+"\t"+str(TrueValues[IDX])+"\t"+str(results[n][0])+"\t"+str(abs(diff))+"\t"+str(uncertainty)+"\t"+str(sig_diff))
            if PlotList: #storing info for plotting
                if names[n] in PlotList:
                    Results[names[n]] = np.append(Results[names[n]], results[n][0])
                    Uncertainties[names[n]] = np.append(Uncertainties[names[n]], results[n][1]+results[n][2]) #storing the uncertaintity width
    if PlotList:
        figCnt = 0
        for k in PlotList:
            k_IDX = transPar_names.index(k)
            percentIn, mean_uncertainty = CalcPercINsig(TrueValues[k_IDX]-Results[k], Uncertainties[k])
            plt.figure(figCnt)
            plt.title("Distribution of "+k+"\n"+str(round(percentIn,2))+"$\%$ in 1-$\sigma$ interval")
            plt.hist(Results[k], bins=8)
            plt.axvline(x=TrueValues[k_IDX], linestyle='-', color='r')
            Err_up = TrueValues[k_IDX]+(mean_uncertainty/2) #here assuming  symetric uncertainties
            Err_Dwn = TrueValues[k_IDX]-(mean_uncertainty/2)
            plt.axvline(x=Err_up, linestyle='--', color='k')
            plt.axvline(x=Err_Dwn, linestyle='--', color='k', label="Mean uncertainty WIDTH="+str(np.round(mean_uncertainty, 5)))
            plt.legend()
            plt.ylabel('Number')
            plt.xlabel(k)
            plt.savefig(method+"_"+k.replace("/", "")+'_Hist_wl.png')
            plt.close()
            print ('saved "'+method+"_"+k.replace("/", "")+'_Hist_wl.png"')
            figCnt += 1
    return None

def CalcPercINsig(Data, Uncertainties, checkSTD=False, relativeSigDif=False): #to calculate the percentage of data out of the sigma confidence interval. SHOULD be ~ 68% in ~32% out
    #Data = array of difference (either difference relative to true value or relative to white-light fit #binned_Uncertainties = array/list of the mean uncertainty widths of each iteration.
    #checkSTD = flag, if on, checks if the std of all bins in a given iteration is within that iterations average uncertainty range
    #relativeSigDif = flag, if on, checks the number of bins that are within 1-sigma difference of the WL fit parameter
    Mean_Unct =  np.mean(Uncertainties)
    if checkSTD:
        cnt_in_conf_int = 0
        for u in range(len(Uncertainties)):
            if Uncertainties[u] >= Data[u]: #Data == mean std per iteration when checkSTD==True. std canNOT be 0
                cnt_in_conf_int+=1
        PerIn = cnt_in_conf_int/len(Data)*100.0
    elif relativeSigDif:
        Mean_Unct = np.mean(Data)
        PerIn = len(np.where((-1<np.array(Data)) & (np.array(Data)<1))[0])/len(Data)*100.0
    else:
        #NOTE! even though the wl fit also has uncertainties, I don't want to include this because the binned fits are done with the CMC(PCA) fit to the found wl value. Thus, for practical purposes the wl fit is the true value
        in_conf_interval = np.where((-Mean_Unct/2 <= Data) & (Data <= +Mean_Unct/2))[0] #since the differences SHOULD be centered on 0, finding how many values are +/- in the uncertainity window (where the window is the binned AND wl uncertainty)
        PerIn = len(in_conf_interval)/len(Data)*100.0
    return PerIn, Mean_Unct

def PlotFinalResults_binned(out_wl, wl_RpRs_name, bin_RpRs_name, TrueRpRs, method, wl_results_file, bin_results_file, wl_Iteration_base, bin_Iteration_base, out_bin='', IterationCnt=50, FontSize=20, TickSize=12):
    #wl_Iteration_base=list of all the path info to the results files that needs an index right after it. Assuming that the only thing missing to combine each element in the list is an index
    #bin_Iteration_base = ^^^ same for binned data #out_wl = the last part of the white-light path to the wl results that DOES NOT include indeces
    #out_bin = same as out_wl but for the wavelength path #wl_RpRs_name == string name that the radius of the planet of radius of star is labeled in the white-light results. #bin_RpRs_name == same but for binned results
    RpRs_diff_fit, RpRs_diff_true, Mean_Uncertainty = [], np.array([]), [] #RpRs_diff_fit = binned RpRs fit vs. the white light RpRs fit. RpRs_diff_true - binned RpRs fit vs. the true RpRs value
    VarPerBin, Sig_wl_Diff, Sig_avg_Diff = [], [], [] #To keep track of the standar deviation of each bin in a given iteration, the sigma difference of each bin from the wl fit, and from the avg depth
    for i in range(0,IterationCnt):
        wl_Path =''
        for wl in wl_Iteration_base:
            wl_Path += wl+str(i)
        Wl_base_Path  =  wl_Path+'/' #assuming this will also be the base file name used for the binned data
        wl_Path = Wl_base_Path+out_wl+'/'
        names, results = utils.LoadResultsTxt(wl_Path+wl_results_file, deliminator='\t', noSpaces_tabs=True)
        p_index = names.index(wl_RpRs_name)
        ieration_p = results[p_index][0] #to get the Rp/Rs found for each iteration, becuase want to check the validitiy of differenetial depth relative to the white-light fit (cmc correction)
        still_bins =True #since don't know how many bins there are ahead of time, keep on going until run out of bins
        bin_Path, bin_cnt= '', 0 #to keep track of bin folder, bin cnt, and sigma difference of each bin from the wl fit for a given iteration
        Bin_Spectra, RpRsDiffFit, Uncertainties = [], [], [] #to keep track of the depth for each bin, it's relative difference from the wl fit, and the uncertainty widths per bin
        RpRsDiffFit_mean = [] #The relative difference from the mean transmission spectra depth from each individual bin
        while still_bins: #1st to find the mean depth of the transmission spectra. And record all values relative to this
            bin_Path = ''
            for Bin in bin_Iteration_base:
                bin_Path += Bin+str(bin_cnt)
            bin_Path += '/'+out_bin+'/'
            names, results = utils.LoadResultsTxt(Wl_base_Path+bin_Path+bin_results_file, deliminator='\t', noSpaces_tabs=True)
            p_index = names.index(bin_RpRs_name)
            bin_p = results[p_index][0] #to get the Rp/Rs found for each iteration, becuase want to check the validitiy of differenetial depth relative to the white-light fit (cmc correction)            
            diff_fromWL = ieration_p-bin_p
            if diff_fromWL >= 0: #if diff is positive, then wl depth is greater than binned depth
                Sig_wl_Diff.append(diff_fromWL/results[p_index][1]) #assuming the 1st error bound is the upper bound
            else: #if diff is negative, then wl depth is less than binned depth
                Sig_wl_Diff.append(diff_fromWL/results[p_index][2]) #assuming the 2nd error bound is the lower bound
            RpRs_diff_true =  np.append(RpRs_diff_true, TrueRpRs-bin_p)
            Bin_Spectra.append(bin_p), Uncertainties.append(results[p_index][1]+results[p_index][2]) #storing the depth and the uncertaintity width
            bin_cnt+=1 
            #to check if the next bin exists. If not stop loop
            Next_bin = ''
            for Bin in bin_Iteration_base:
                Next_bin += Bin+str(bin_cnt)
            Next_bin = Wl_base_Path+Next_bin+'/'+out_bin+'/'
            if not os.path.exists(Next_bin):
                still_bins = False       
        AvgDepth = np.average(np.array(Bin_Spectra), weights = 1./((np.array(Uncertainties))/2.0)**2) 
        Mean_Uncertainty.append(np.mean(Uncertainties)) #to store the mean uncertainity for bins in a given iteration
        RpRs_diff_fit.append(np.mean(RpRsDiffFit))
        VarPerBin.append(np.std(Bin_Spectra))

        still_bins, bin_cnt = True, 0 
        while still_bins:
            bin_Path = ''
            for Bin in bin_Iteration_base:
                bin_Path += Bin+str(bin_cnt)
            bin_Path += '/'+out_bin+'/'
            names, results = utils.LoadResultsTxt(Wl_base_Path+bin_Path+bin_results_file, deliminator='\t', noSpaces_tabs=True)
            p_index = names.index(bin_RpRs_name)
            bin_p = results[p_index][0] #to get the Rp/Rs found for each iteration, becuase want to check the validitiy of differenetial depth relative to the white-light fit (cmc correction)
            diff_fromMean = AvgDepth-bin_p
            if diff_fromMean >= 0: #if diff is positive, then mean depth is greater than binned depth
                Sig_avg_Diff.append(diff_fromMean/results[p_index][1]) #assuming the 1st error bound is the upper bound
            else: #if diff is negative, then mean depth is less than binned depth
                Sig_avg_Diff.append(diff_fromMean/results[p_index][2]) #assuming the 2nd error bound is the lower bound
            bin_cnt += 1
            #to check if the next bin exists. If not stop loop
            Next_bin = ''
            for Bin in bin_Iteration_base:
                Next_bin += Bin+str(bin_cnt)
            Next_bin = Wl_base_Path+Next_bin+'/'+out_bin+'/'
            if not os.path.exists(Next_bin):
                still_bins = False


    #To plot the std of all the bins in each iteration, relative to the overall mean error bar widths (printing the std relative to the mean error bars of that iteration)
    percentIn1, mean_uncertainty1 = CalcPercINsig(VarPerBin, Mean_Uncertainty, checkSTD=True) #want to measure how close the std of a given iteration is within the uncertainty width of the data. Thus, *2 because using the FULL width as the bounds
    percentIn2, mean_uncertainty2 = CalcPercINsig(RpRs_diff_true, Uncertainties)
    mean_uncertainty = np.max([mean_uncertainty2, mean_uncertainty1]) #mean uncertainties calculated slightly differently. Method1 is the mean of means (mean of each synth spectrum group). Method2 is take each individual bin for each synth group and take mean of that

    #To plot the mean 
    print ("mean uncertainty:", mean_uncertainty)
    print (str(round(percentIn2,2))+"$\%$ in 1-$\sigma$ interval")
    # plt.figure(1)
    # plt.title("Distribution of "+bin_RpRs_name+"\n"+str(round(percentIn2,2))+"$\%$ in 1-$\sigma$ interval")
    plt.subplot(131)
    plt.hist(RpRs_diff_true, bins=10)
    plt.axvline(x=0, linestyle='-', color='r')
    plt.axvline(x=mean_uncertainty/2, linestyle='--', color='k') #here assuming  symetric uncertainties
    plt.axvline(x=-mean_uncertainty/2, linestyle='--', color='k', label="Mean uncertainty\n=$\pm$"+str(np.round(mean_uncertainty/2, 5))) #also only including the uncertainty of 
    plt.ylabel('Number of Bins', fontsize=FontSize, fontweight='bold')
    # plt.xlabel('True_('+bin_RpRs_name+')- Fitted_('+bin_RpRs_name+')', fontsize=FontSize, fontweight='bold')
    plt.xlabel('True_RpRs - Fitted_RpRs', fontsize=FontSize, fontweight='bold')
    plt.xticks(fontsize=TickSize, fontweight='bold')
    plt.yticks(fontsize=TickSize, fontweight='bold')
    plt.legend(prop={'size':TickSize, 'weight':'bold'})
    fil_name = method+"_"+bin_RpRs_name.replace("/", "")+'diffvsTrueVal_Hist_wave.png'
    print ("TITLE: Distribution of "+bin_RpRs_name+" $\sigma$\n"+str(round(percentIn2,2))+"$\%$ in 1-$\sigma$ interval")
    # plt.savefig(fil_name,bbox_inches='tight')
    # print ('saved "'+fil_name)
    print ("\n")

    #To plot the sigma difference (relative to its own uncertainties) of each bin's RpRps relative to the wl fit
    percentIn, mean_sig_diff = CalcPercINsig(Sig_wl_Diff, 0, relativeSigDif=True) #here 'mean_sig_diff' is the average sigma difference from the fitted wl value
    print ("mean $\sigma$ difference:", mean_sig_diff)
    print (str(round(percentIn,2))+"$\%$ in 1-$\sigma$ interval")
    # plt.figure(2)
    # plt.title("Distribution of "+bin_RpRs_name+" $\sigma$\n"+str(round(percentIn,2))+"$\%$ in 1-$\sigma$ interval")
    # plt.hist(Sig_wl_Diff, bins=10)
    # plt.axvline(x=-1, linestyle='--', color='k')
    # plt.axvline(x=0, linestyle='-', color='r')
    # plt.axvline(x=1, linestyle='--', color='k', label="Mean $\sigma$="+str(np.round(mean_sig_diff, 5))) #also only including the uncertainty of 
    # plt.ylabel('Number of Bins', fontsize=FontSize, fontweight='bold')
    # plt.xlabel('$\sigma$ Difference WL fit', fontsize=FontSize, fontweight='bold')
    # plt.xticks(fontsize=TickSize, fontweight='bold')
    # plt.yticks(fontsize=TickSize, fontweight='bold')
    # plt.legend(prop={'size':TickSize, 'weight':'bold'})
    # fil_name = method+"_"+bin_RpRs_name.replace("/", "")+'SigDiff_fromWL_Hist_wave.png'
    # print ("TITLE: Distribution of "+bin_RpRs_name+" $\sigma$\n"+str(round(percentIn,2))+"$\%$ in 1-$\sigma$ interval")
    # plt.savefig(fil_name,bbox_inches='tight')
    # print ('saved "'+fil_name)
    print ("\n")

    #To plot the sigma difference (relative to its own uncertainties) of each bin's RpRps relative to the mean wl value
    percentIn, mean_sig_diff = CalcPercINsig(Sig_avg_Diff, 0, relativeSigDif=True) #here 'mean_sig_diff' is the average sigma difference from the mean wl value
    print ("mean $\sigma$ difference:", mean_sig_diff)
    print (str(round(percentIn,2))+"$\%$ in 1-$\sigma$ interval")
    # plt.figure(4)
    # plt.title("Distribution of "+bin_RpRs_name+" $\sigma$\n"+str(round(percentIn,2))+"$\%$ in 1-$\sigma$ interval")
    plt.subplot(132)
    plt.hist(Sig_avg_Diff, bins=10)
    plt.axvline(x=-1, linestyle='--', color='k')
    plt.axvline(x=0, linestyle='-', color='r')
    plt.axvline(x=1, linestyle='--', color='k',label='1-$\sigma$\ninterval')#label="Mean $\sigma$="+str(np.round(mean_sig_diff, 5))) #also only including the uncertainty of 
    plt.ylabel('Number of Bins', fontsize=FontSize, fontweight='bold')
    plt.xlabel('$\sigma$ Difference from Mean Depth', fontsize=FontSize, fontweight='bold')
    plt.xticks(fontsize=TickSize, fontweight='bold')
    plt.yticks(fontsize=TickSize, fontweight='bold')
    plt.legend(prop={'size':TickSize, 'weight':'bold'})
    fil_name = method+"_"+bin_RpRs_name.replace("/", "")+'SigDiff_fromAVG_Hist_wave.png'
    print ("TITLE: Distribution of "+bin_RpRs_name+" $\sigma$\n"+str(round(percentIn,2))+"$\%$ in 1-$\sigma$ interval")
    # plt.savefig(fil_name,bbox_inches='tight')
    # print ('saved "'+fil_name)
    print ("\n")

    print ("mean uncertainty:", mean_uncertainty)
    print (str(round(percentIn1,2))+"$\%$ in 1-$\sigma$ interval")
    # plt.figure(3)
    # plt.title("Avg Standard Deviation of "+bin_RpRs_name+" in each iteration\n"+str(round(percentIn1,2))+"$\%$ in 1-$\sigma$ interval (NOTE: slightly different than what's being plotted)")
    plt.subplot(133)
    plt.hist(VarPerBin, bins=10)
    plt.axvline(x=0, linestyle='-', color='r')
    plt.axvline(x=+mean_uncertainty, linestyle='--', color='k', label="Mean uncertainty\nWIDTH="+str(np.round(mean_uncertainty, 5)))
    plt.ylabel('Number of Spectra', fontsize=FontSize, fontweight='bold')
    plt.xlabel('Std of Transmission Spectra', fontsize=FontSize, fontweight='bold')
    Xrange = np.max(VarPerBin+[mean_uncertainty]) # to find the distance from 0 to the edge of the plot. edge could be the uncertainty width or the histagram
    Xrange += Xrange*.03 #add a lil extra buffer
    Xticks = np.round(np.linspace(0,Xrange,num=6),4)
    plt.xticks(Xticks,fontsize=TickSize, fontweight='bold')
    plt.yticks(fontsize=TickSize, fontweight='bold')
    plt.legend(prop={'size':TickSize, 'weight':'bold'})
    #NOTE! what is being plot here is not exactly the same as what's being print as % in. The % is the # of std of a given iteration within its own iterations average uncertainty width
    #      where the dotted line in the plot is a global average of all uncertainity widths
    fil_name = method+"_"+bin_RpRs_name.replace("/", "")+'Iteration_STD_Hist_wave.png'
    print ("TITLE: Avg Standard Deviation of "+bin_RpRs_name+" in each iteration\n"+str(round(percentIn1,2))+"$\%$ in 1-$\sigma$ interval (NOTE: slightly different than what's being plotted)")
    # plt.savefig(fil_name,bbox_inches='tight')
    # print ('saved "'+fil_name)
    combined_fil = method+"_"+bin_RpRs_name.replace("/", "")+"_Allfigs.png"
    plt.savefig(combined_fil,bbox_inches='tight')
    print ('saved "'+combined_fil+'"')    
    print ("TrueRpRs:", TrueRpRs, '\n\n')
    return None    
    
if __name__ == "__main__":
    trans_duration, EparNams = 2+(26/60), ['W96SpecShift','RotAng','Airmas','FWHM','CrossDisp']
    # LCfile = 'SynthDatPkls/Iteration0/FinalLC_0.pkl'
    # with open(LCfile, 'rb') as pkl_file:
    #     LC_ob1 = pickle.load(pkl_file, encoding="latin1")
    # Bins = LC_ob1['wbins']
    # path = 'GPfit_CMCnPoly_FORS2R/Binned/'
    # # Loop_Finalize('SynthDatPkls/Iteration', 'SynthDatPkls/Iteration', EparNams, trans_duration, 2, add_subDirs='/CMCpoly_wavelength', wait_T=120)#2*3600.0 #(clean every 2hrs) how long you want to wait have trying to clear each iteration 
    # # Loop_Finalize(['SynthDatPkls/outputs_py3pca/Iteration', 'FinalLC_'], 'SynthDatPkls/Iteration', EparNams, trans_duration, 2, add_subDirs='/CMCpoly_wavelength', wait_T=120)#2*3600.0 #(clean every 2hrs) how long you want to wait have trying to clear each iteration 

    # LCfile = '../GPTransmissionSpectra3/WASP-96/FORS2_250B_noTel.pkl' #LCfile = 'SynthDatPkls/Iteration0/FinalLC_0.pkl'
    # with open(LCfile, 'rb') as pkl_file:
    #     LC_ob1 = pickle.load(pkl_file, encoding="latin1")
    # Bins = LC_ob1['wbins']
    # path = '../GPTransmissionSpectra3/outputs/WASP-96/FORS2_250B_noTel/Binned/'
    # EparNams = ['TargSpecShift', 'CrossDisp', 'FWHM', 'Airmass', 'Bckg2dspec', 'RotAng']
    # Finalize(trans_duration, EparNams, Bins, 2, path, Clean=True, PLOTFinReslts=False, WriteBestFit=False, RunShorts=False)
    # print ("\n\n\n\n")

    # LCfile = '../GPTransmissionSpectra3/WASP-96/FORS2_250R_noTel.pkl' #LCfile = 'SynthDatPkls/Iteration0/FinalLC_0.pkl'
    # with open(LCfile, 'rb') as pkl_file:
    #     LC_ob1 = pickle.load(pkl_file, encoding="latin1")
    # Bins = LC_ob1['wbins']
    # path = '../GPTransmissionSpectra3/outputs/WASP-96/FORS2_250R_noTel/Binned/'
    # EparNams = ['TargSpecShift', 'CrossDisp', 'FWHM', 'Airmass', 'Bckg2dspec', 'RotAng']
    # Finalize(trans_duration, EparNams, Bins, 2, path, Clean=True, PLOTFinReslts=False, WriteBestFit=False, RunShorts=False)
    # print ("\n\n\n\n")

    # LCfile = '../GPTransmissionSpectra3/WASP-96/LCs_w96_250ob1_noTel.pkl' #LCfile = 'SynthDatPkls/Iteration0/FinalLC_0.pkl'
    # with open(LCfile, 'rb') as pkl_file:
    #     LC_ob1 = pickle.load(pkl_file, encoding="latin1")
    # Bins = LC_ob1['wbins']
    # path = '../GPTransmissionSpectra3/outputs/WASP-96/LCs_w96_250ob1_noTel/Binned/'
    # EparNams = ['Airmass','DeltaWav','FWHM','SkyFlux','TraceCenter']
    # Finalize(trans_duration, EparNams, Bins, 2, path, Clean=True, PLOTFinReslts=False, WriteBestFit=False, RunShorts=False)
    # print ("\n\n\n\n")

    # LCfile = '../GPTransmissionSpectra3/WASP-96/LCs_w96_250ob2_noTel.pkl' #LCfile = 'SynthDatPkls/Iteration0/FinalLC_0.pkl'
    # with open(LCfile, 'rb') as pkl_file:
    #     LC_ob1 = pickle.load(pkl_file, encoding="latin1")
    # Bins = LC_ob1['wbins']
    # path = '../GPTransmissionSpectra3/outputs/WASP-96/LCs_w96_250ob2_noTel/Binned/'
    # EparNams = ['Airmass','DeltaWav','FWHM','SkyFlux','TraceCenter']
    # Finalize(trans_duration, EparNams, Bins, 2, path, Clean=True, PLOTFinReslts=False, WriteBestFit=False, RunShorts=False)
    # print ("\n\n\n\n")

    # LCfile = 'FORS2/LC_FOR2_B.pkl' #LCfile = 'SynthDatPkls/Iteration0/FinalLC_0.pkl'
    # with open(LCfile, 'rb') as pkl_file:
    #     LC_ob1 = pickle.load(pkl_file, encoding="latin1")
    # Bins = LC_ob1['wbins']
    # path = 'GPfit_CMCnPoly_FORS2B/Binned/'
    # Finalize(trans_duration, EparNams, Bins, 2, path, Clean=True, PLOTFinReslts=False, WriteBestFit=False, RunShorts=False)
    # path = 'GPfit_CMCnPoly_FORS2B_NiksWL/'
    # Finalize(trans_duration, EparNams, Bins, 2, path, Clean=True, PLOTFinReslts=False, WriteBestFit=False, RunShorts=False)
    
    # TransParNames = ["P", "t0", "p", "aR", "inc", "q1", "q2"] #GPTransSpec method #["P", "t0", "Rp/Rs", "a/Rs", "inc", "u1", "u2"] #param names for simpl_GP method
    # plot_list = ["p", "aR", "inc"]#["Rp/Rs", "a/Rs", "inc"] #list of parameters to plot as histogram 
    # true_values = [3.4252589578, 2457963.3364990650, 0.1157020309, 9.0068949265, 85.2481401045, 0.36145090647234457, 0.13077070113951147] 
    # Iteration_base = ["SynthDatPkls/outputs_py3pca/Iteration","/FinalLC_"]
    # ReadFinalResults_wl('white-light', TransParNames, plot_list, true_values, 'GPTrans', 'results.dat', Iteration_base, PrintFits=False, PlotFits=True)
    
    plt.figure(figsize=(18,6))
    print ("============== PCA+GP ==============")
    wl_results_file, bin_results_file = ['SynthDatPkls/outputs_py3pca/Iteration', '/FinalLC_'], ['wavelength/wbin']
    PlotFinalResults_binned('white-light','p', 'p', 0.1157020309, 'GP+PCA', 'results.dat', 'results.dat', wl_results_file, bin_results_file)
    plt.close()

    plt.figure(figsize=(18,6))
    print ("============== CMC+POLY ==============")
    wl_results_file, bin_results_file = ['SynthDatPkls/outputs_py3pca/Iteration', '/FinalLC_'], ['CMCpoly_wavelength/Bin']
    PlotFinalResults_binned('white-light','p', 'Rp/Rs', 0.1157020309, 'CMC+Poly', 'results.dat', 'BMA_final_results.txt', wl_results_file, bin_results_file, IterationCnt=50)
    plt.close()
