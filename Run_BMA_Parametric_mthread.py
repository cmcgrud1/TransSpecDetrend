import numpy as np
import sys
import os
from scipy.io import readsav
import time
import itertools
import pickle
import corner
import matplotlib.pyplot as plt
import batman
import utils
import glob 

def Add_missing(out_path, Epars_names, Max_P_order, epar_inpts=None, LC=None, TransPar=None, TranDur=None, WriteBestFit=True, JustCnt=False, RunResults=False, DONTrunResults=False, PrintVariants=True, Rm_Unfinished=False, ToPlot=True, saveNPY=False):
    """If ran through the code, but the Global_LnZ.txt file isn't full. Go through that file and see which Variants are missing.
    Also a good way to run the Results() function, after making sure the Global_LnZ.txt file is finished
    #RunResults= if want to run the Results() function, regardless of missing variants
    #JustCnt = if want to just count how many options there are, then stop
    #LC, TransPar, TranDur, epar_inpts == need these paramters when DONTrunResults==False
    #Rm_Unfinished=to remove the directory that contains the unfinished variant. Would want to do this in the cases where we want to rerun that variant
    #DONTrunResults == if don't want to run the final results script NO MATTER WHAT. Watch out for the double neg
    """
    if Rm_Unfinished:
        import shutil
    if out_path[-1] == '/': #to make sure we stay consitent with backslashes
        out_path = out_path[:-1]
    Names, LnZs = utils.LoadResultsTxt(out_path+'/Global_LnZs.txt', deliminator='\t\t\t\t', noSpaces_tabs=True)
    Variants = utils.getVariants(Epars_names, Max_P_order) 
    print ("Total Variants:", len(Variants))
    if JustCnt: #just count the number of variants, then exit code
        return Variants
    
    def Add_var(var, del_lines): #internal function to correct the missing or errored variant info
        print ("trying to add "+var+"'s results to Global_LnZ.txt file ....")
        if os.path.isfile(out_path+'/'+var+'/fit_results.txt'):
            last_line = utils.ReadLastLine(out_path+'/'+var+'/fit_results.txt')
            idx1, idx2, idx3 = last_line.index('#Global lnZ: ')+13, last_line.index('ndim: '), last_line.index('sample_number: ')
            LnZ, ndim, Nsample = float(last_line[idx1:idx2]), int(last_line[idx2+6:idx3]), int(last_line[idx3+15:])
            LnZs_tab = open(out_path+'/Global_LnZs.txt','a')
            LnZs_tab.write(var+'\t\t\t\t'+str(LnZ)+'\t\t\t\t'+str(Nsample)+'\n')
            LnZs_tab.close()
            print ("Done\n") 
            #since there was a missing variant, there is likely a random int/string that just doesn't fit. Here we find and eradicate it 
            if len(del_lines) == 0: #only have to do this once be file
                nm_cnt = 1 #starting with 1, becaues utils.RemoveLine() starts with index 1
                for name in Names:
                    if name.isnumeric() or '\n' in name or len(LnZs[nm_cnt-1]) != 2: # the signs of the error are if the 1st column (which is supposed to be a str) is a int, OR the length of the global_ev/samp_num isn't 2
                        del_lines.append(nm_cnt+1) #plus another extra 1 because the 1st row is the labels, which is skipped
                    nm_cnt +=1
        else:
            print ("no final results for '"+var+"'. You need to rerun this!")
            UnfinishedVars.append(var)
            if Rm_Unfinished:
                print ("deleting the '"+out_path+'/'+var+"' directory, in a hope to start fresh")
                shutil.rmtree(out_path+'/'+var)
            print ('\n')
        return del_lines

    #### Now to check if that specific combination is in the final Global_LnZs.txt file. ASSUMING epars are written in same order
    fnd_idxs, UnfinishedVars = [], []
    DelLines = [] #to keep track of which lines to delete at the end of checking which variants are missing
    for V in Variants:
        if V not in Names: #if that specific variant isn't in the Global_LnZs.txt file, something is wrong!
            DelLines = Add_var(V, DelLines) #Try to correct mistake
        elif len(LnZs[Names.index(V)]) != 2: #another issue, could be that the name is there, but the LnZ/sample# values aren't (caused by same bug)
            DelLines = Add_var(V, DelLines) #Try to correct mistake
        else: #otherwise, all good :)
            continue
    if len(DelLines) > 0: #to get rid of those weird glitched out lines
        utils.RemoveLine(out_path+'/Global_LnZs.txt', DelLines)
    if RunResults: #to run the results regardless of if missing any variants
        Results(out_path, LC, TransPar, TranDur, epar_inpts, write_bestFit=WriteBestFit, PrintVars=PrintVariants, PLOT=ToPlot, SaveAsNPY=saveNPY)
    elif len(UnfinishedVars) > 0: #don't run the results if missing any variants
        print ("Because missing "+str(len(UnfinishedVars))+" variants, not running the Results() function")
        print ("Missing the following variants:",UnfinishedVars)
    else: #to run the results ONLY if have all the variants
        if not DONTrunResults:
            Results(out_path, LC, TransPar, TranDur, epar_inpts, write_bestFit=WriteBestFit, PrintVars=PrintVariants, PLOT=ToPlot, SaveAsNPY=saveNPY)
    return None

def Plot_Corner(Path,Quants=[0.15865,0.50, 0.84135]):
    if Path[-1] == '/':
        Path = Path[:-1]
    file = open(Path+'/equal_weighted_posteriors.pkl','rb')
    posterior_samples = pickle.load(file)
    file.close()
    Param_names, Postiers = utils.LoadResultsTxt(Path+'/fit_results.txt')
    fig = corner.corner(posterior_samples, labels=Param_names, quantiles=Quants)
    fig.savefig(Path+'/corner_'+Path.split('/')[-1]+'.png')
    print ("Saved 'corner_"+Path.split('/')[-1]+".png' in '"+Path+"'")
    plt.close()
    return None

def Plot_LC(time_input, flux_input, error_input, trans_params, trans_dur, ExtParInput, Path, BMA=False):
    if Path[-1] == '/':
        Path = Path[:-1]
    LC = [time_input, flux_input, error_input]
    if BMA: #TODO!!! NEED TO FINISH THE BMA SECTION
        utils.PlotLC(LC, 'These are', 'place', 'holders', trans_dur, transit_coeff, FULLpath, plot_name='BMA_Model', poly_model=None)
    else:
        Param_Names, Posteriors = utils.LoadResultsTxt(Path+'/fit_results.txt', deliminator='\t')
        parm_nams = [] #list of parameters without the whitespaces
        for P in Param_Names:
            parm_nams.append(P.strip())
        trans_names = ['P', 't0', 'Rp/Rs', 'a/Rs', 'inc', 'u1', 'u2']
        tn_cnt, used_cnt = 0, 0 #to keep track of which transit parameter, and posterior transit parameter (fitted param we are on)
        for tn in trans_names:  #to replace the given transit parameters with the ones that are fitted for. In the same order they are consistently written
            if tn in parm_nams: #if priors were specified then used the best fit posterior
                trans_params[tn_cnt] = Posteriors[used_cnt][0] #0th index is mean fit
                used_cnt +=1
            tn_cnt +=1
        EparsIdx = parm_nams.index('poly_offset') #deliminator where this index and everything preceding it is an external parameter
        Ecoff_name = parm_nams[EparsIdx+1:]
        Epar_names, poly_order, poly_coeffs= [], [], [Posteriors[EparsIdx][0]]
        Epar_place = -1 #start with -1, because adding 1 if this is the 1st occurance and want to start with 0 index
        for Ep in range(len(Ecoff_name)):
            Ename = Ecoff_name[Ep].split('_')[0]
            if Ename not in Epar_names: #if this is the 1st occurance of the Epar name
                Epar_place +=1 #move on to the next iteration of the Epar names. This works because assuming that each Eparm is group togther, i.e. doesn't ossiclate from one Epar name to another and back
                Epar_names.append(Ename) #add it to the listed of used epars
                poly_order.append(1) #add at least 1 order to the poly order for that same Eparname
            else: #otherwise add another cnt to the poly_order
                poly_order[Epar_place] += 1
            poly_coeffs.append(Posteriors[EparsIdx+1+Ep][0])
        Eparams = []
        for name in Epar_names:
            Eparams.append(ExtParInput[name])
        print ("poly_order:", poly_order, "Epar_names:", Epar_names) 
        utils.PlotLC(LC, poly_coeffs, Eparams, poly_order, trans_dur, trans_params, Path, plot_name='ModelFit_'+Path.split('/')[-1])
    return None

def Results(path_outfile, lc, Trans_Pars, Tran_dur, epar_inputs, write_bestFit=True, PrintVars=False, PLOT=True, SaveAsNPY=False):# Load LnZs and determine best model and plot it
    #epar_inputs = dictionary of all external parameters used (aside from sig clipped data). The code determines which external parameters are actually plotted
    #SaveAsNPY = if you want to save the data used for plotting. Can be either boolan, or a string consisting of the path name where the data should be saved. DON'T FORGET TO INCLUDE THE '/' at the end of the last path!!
    if path_outfile[-1] == '/':
        path_outfile = path_outfile[:-1]
    ### To convert log-Evidences to probabilities AND determine the best model 
    Names, LnZs = utils.LoadResultsTxt(path_outfile+'/Global_LnZs.txt', deliminator='\t\t\t\t')
    lnZ = LnZs[:,0]
    # Calculate posterior probabilities of the models from the Bayes Factors:
    lnZ = lnZ - np.max(lnZ)
    Z = np.exp(lnZ) #convert to Evidence
    Pmodels = Z/np.sum(Z) #normalize evidence to one, to convert it to probablility
    MaxMod = np.argmax(Pmodels)
    Param, Posteriors = utils.LoadResultsTxt(path_outfile+'/'+Names[MaxMod]+'/fit_results.txt', noSpaces_tabs=True)
    if PrintVars:
        print ("Variants and probabilities:")
        Print_str, liNes = '', 0
        for nm in range(len(Names)): 
            if liNes < 10: #print 10 variants and their respective probabilities per line
                Print_str +=  Names[nm]+'='+str(Pmodels[nm])+', '
                liNes +=1
            else:
                Print_str +=  Names[nm]+'='+str(Pmodels[nm])+', \n'
                liNes = 0
        print (Print_str)
    if write_bestFit: #the 1st pass, write the best fit results in the Global_LnZs file
        LnZs_tab = open(path_outfile+'/Global_LnZs.txt','a')
        LnZs_tab.write('#\n###\n#\n#best fit model: '+Names[MaxMod]+' with probability of '+str(Pmodels[MaxMod])+' and log(Evidence) of '+str(LnZs[MaxMod,0]))
        LnZs_tab.close()
    print ("Total # of models tested:", len(LnZs))#, '\n')
    print ('best fit model: '+Names[MaxMod]+'\t model probability: '+str(Pmodels[MaxMod])+'\t model log(Evidence): '+str(LnZs[MaxMod,0]))

    ### To keep track of which transit parameters where actually fit for
    best_params, Used_trans_names, used_transPs = {}, [], 0
    trans_names = ['P', 't0', 'Rp/Rs', 'a/Rs', 'inc', 'u1', 'u2']
    for TP in trans_names:  #to fill a list of used transit params in the same order they are consistently written
        if Trans_Pars[TP][1]: #if priors were specified then used the best fit posterior
            IDX = Param.index(TP)
            best_params[TP]  = Posteriors[IDX,0] #to get the mean of the distribution for that parameter
            Used_trans_names.append(TP)
            used_transPs +=1
        else: #if priors weren't specified then used the initial parameters given
            best_params[TP]  = Trans_Pars[TP][0]
    print ("# of fitted transit parameters:", used_transPs)
    ### To print/plot the results of the best fit model and corner
    best_P, best_t0, best_RpRs =  best_params['P'], best_params['t0'],best_params['Rp/Rs']
    best_a, best_inc, best_u1, best_u2 = best_params['a/Rs'], best_params['inc'], best_params['u1'], best_params['u2']
    best_offset = Posteriors[used_transPs,0]
    best_coeffs = Posteriors[used_transPs+1:,0]
    epars_N_orders = Names[MaxMod].split('_')
    best_Eparms, best_polyOrders = [], []
    for eNo in epars_N_orders:
        split_eNo = eNo.split('-')
        if len(split_eNo) == 2: #most common case where there's an external parameter used
            b_epar_name, b_porder = split_eNo[0], split_eNo[1]
            b_epar_arry = epar_inputs[b_epar_name]
            best_Eparms.append(b_epar_arry), best_polyOrders.append(int(b_porder))
        elif len(split_eNo) == 1: #the one off case where there's no external parameter used
            best_Eparms, best_polyOrders = [], [0] #in that case len(EparNorder) = 1 so define list of epar values and orders here
        else:
            sys.exit("Error! couldn't find the external parameter and/or order")
    best_poly_coeff = [best_offset]+list(best_coeffs)
    best_trans_pars = [best_P, best_t0, best_RpRs, best_a, best_inc, best_u1, best_u2]
    print ("best_trans_pars:", best_trans_pars)
    file = open(path_outfile+'/'+Names[MaxMod]+'/equal_weighted_posteriors.pkl','rb')
    Bestposteriors = pickle.load(file)
    file.close()
    if PLOT:
        utils.PlotLC(lc, best_poly_coeff, best_Eparms, best_polyOrders, Tran_dur, best_trans_pars,path_outfile, plot_name='BestFitModel_'+Names[MaxMod])
        fig = corner.corner(Bestposteriors, labels=Param, quantiles=[0.15865,0.50, 0.84135])
        fig.savefig(path_outfile+'/corner_'+Names[MaxMod]+'.png')
        print ("Saved 'corner_"+Names[MaxMod]+".png' in '"+path_outfile)#+"'\n")  
        plt.close()

    ### To produce the BMA final results
    min_sample = np.min(LnZs[:,1]) #to keep track of how many samples that can be drawn for BMA
    total_samples = 0 #to keep track of the total number of samples, because that might not be exactly min_sample
    for m in range(len(Pmodels)):
        nextract =int(Pmodels[m]*min_sample)
        total_samples += nextract
    # print ("total_samples:", total_samples)

    #given that this is a model averaging of different polynomials with different orders and against different external parameters
    #I THINK the best way to get the polynomial correction postieror is save each samples polynomial LC, rather than any polynomial coefficients
    parametric_fits = np.zeros((len(lc[0]),total_samples))
    BMA_Postierior = {}
    for tp_names in Used_trans_names: #to initilize an array of all the used transit parameters
        BMA_Postierior[tp_names] = np.zeros(total_samples)

    finalstat_str ='###Total_samples='+str(total_samples)+'###\n#' #to keep track of how the final BMA results was sampled
    strt, ln_cnt = 0, 0
    for m in range(len(Pmodels)):
        #P, t0, Rp/Rs, a/Rs, inc, u1, u2 #order of transit parameters
        nextract =int(Pmodels[m]*min_sample)
        # Extract transit parameters:
        file = open(path_outfile+'/'+Names[m]+'/equal_weighted_posteriors.pkl','rb')
        posteriors = pickle.load(file)
        file.close()
        if nextract > 0: #save the number of samples draw for each variant
            if ln_cnt < 10: #10 params per line
                finalstat_str +=Names[m]+'='+str(nextract)+'\t'
                ln_cnt +=1
            else:
                finalstat_str +='\n#'+Names[m]+'='+str(nextract)+'\t'
                ln_cnt =0 #reset line count 
        idx_extract = np.random.choice(np.arange(len(posteriors[:,0])),nextract,replace=False)
        ext_cnt = len(idx_extract) #number of samples to be pulled for this model (based on probability)
        Utn_cnt = 0
        for Utn in Used_trans_names:
            BMA_Postierior[Utn][strt:strt+ext_cnt] = posteriors[idx_extract,Utn_cnt]
            Utn_cnt +=1
        EparNorder = Names[m].split('_')
        Epars_vals, Orders = [], []
        for EnO in EparNorder: #To get the external parameter name and order for this model
            split_En0 = EnO.split('-')
            if len(split_En0) == 2: #most common case where there's an external parameter used
                E,O = split_En0[0], split_En0[1]
            elif len(split_En0) == 1: #the one off case where there's no external parameter used
                Epars_vals, Orders = [], [0] #in that case len(EparNorder) = 1 so define list of epar values and orders here
            else:
                sys.exit("Error! couldn't find the external parameter and/or order")
            Epars_vals.append(epar_inputs[E]), Orders.append(int(O))
        for n in range(nextract): #to draw a sys_model lc for each number of samples needed for this model based on it's specific probability
            sample = idx_extract[n]
            p0 = posteriors[sample,used_transPs:]
            sys_model = utils.systematics_model(p0, Epars_vals, Orders, len(lc[0]))
            parametric_fits[:,strt+n] = sys_model #have to add each sys_model individually
        strt = strt+ext_cnt #to keep track of the range of the index range, for this specific model
    parametric_model = np.percentile(parametric_fits, 50, axis=1)
    BMA_transit_samples = [] #to save the posterior of the transit parameters, for making the corner plot
    for key in Used_trans_names:
        BMA_transit_samples.append(BMA_Postierior[key])
    BMA_Postierior["parametric_fits"] = parametric_fits
    pickle.dump(BMA_Postierior, open(path_outfile+'/equal_weighted_BMAposteriors.pkl', 'wb'))

    ### To print, save, and plot the BMA results
    mean_trans_pars, TNs_cnt, Results_str = [], 0, '#parameter   best_fit        upper_bnd       lower_bnd\n'
    for TNs in trans_names:
        if TNs in Used_trans_names: #if fit for use the 50th precentile as the BMA value
            samp_reslts = np.percentile(BMA_Postierior[TNs], [15.865,50, 84.135])
            mean_trans_pars.append(samp_reslts[1])
            p_mean, p_upper, p_lower = np.round(samp_reslts[1],7), np.round(samp_reslts[2]-samp_reslts[1],8), np.round(samp_reslts[1]-samp_reslts[0],8) 
            Results_str += TNs+'      \t'+str(p_mean)+'\t'+str(p_upper)+'\t'+str(p_lower)+'\n'
        else: #otherwise use the initial transit param
            mean_trans_pars.append(Trans_Pars[trans_names[TNs_cnt]][0])
        TNs_cnt +=1
    results_tab = open(path_outfile+'/BMA_final_results.txt','w+')
    results_tab.write(Results_str+finalstat_str) 
    print ("Saved 'BMA_final_results.txt' in '"+path_outfile)#+"'\n")
    results_tab.close()

    print ("mean transit parameters:", mean_trans_pars)
    if PLOT or SaveAsNPY:
        utils.PlotLC(lc, 'These are', 'place', 'holders', Tran_dur, mean_trans_pars, path_outfile, plot_name='BMAmodel', poly_model=np.mean(parametric_fits, axis=1), save_dats=SaveAsNPY)
        fig = corner.corner(np.array(BMA_transit_samples).T, labels=Used_trans_names, quantiles=[0.15865,0.50, 0.84135])
        fig.savefig(path_outfile+'/BMA_corner.png')
        print ("Saved 'BMA_corner.png' in '"+path_outfile)#+"'\n")
        plt.close()        
    return None

def Correct4Bugg(sub_folder, Eparams_names, transit_pars, PolyOrders, Max_Poly_order, Quants=[0.15865,0.50, 0.84135]): #To create a 'fit_results.txt' file for this specific variant. To correct for a bug that didn't create it before
    ######To get the list of parameters used for this run
    parameter_list = []
    trans_param_str = ['P', 't0', 'Rp/Rs', 'a/Rs', 'inc', 'u1', 'u2'] #1st 7 external parameters are these transit parameters, IF it's prior is not 0
    for tp in trans_param_str: #to keep track of the used transit parameters
        if transit_pars[tp][1]: #2nd element is the prior function. if None, then kept fixed and not fitting for
            parameter_list.append(tp)
    parameter_list += ['poly_offset'] #one polynomial offset term for all of the polynomial fits
    E_cnt, results_name= 0, ''
    if len(Eparams_names) == 0:#the case when there is only an offset fit for
        results_name = 'Offset'+str(PolyOrders[0])
    for E in Eparams_names:
        NumCoeffs = PolyOrders[E_cnt] #number of coefficents for this variable
        if E_cnt == 0:
            results_name += E+'-'+str(NumCoeffs)
        else:
            results_name += '_'+E+'-'+str(NumCoeffs)
        Poly_Coeffs = []
        for nc in range(NumCoeffs):
            Poly_Coeffs += [E+'_coeff'+str(nc)] 
        Poly_Coeffs = Poly_Coeffs[::-1] #need to change the order, since poly1d() goes in decending order
        E_cnt +=1
        parameter_list += Poly_Coeffs
    ######

    ######To write the fit_results.txt file
    if os.path.isfile(sub_folder+'/equal_weighted_posteriors.pkl'):
        file = open(sub_folder+'/equal_weighted_posteriors.pkl','rb')
    else:
        print ("Couldn't find '"+sub_folder+"/equal_weighted_posteriors.pkl' file\nSkipping the 'fit_results.txt' file rewritting for the '"+sub_folder+"' directory.")
        return None
    posterior_samples = pickle.load(file)
    file.close()
    results_tab = open(sub_folder+'/fit_results.txt','w+')
    results_tab.write('#parameter   best_fit        upper_bnd       lower_bnd\n')
    ndim = 0
    for p in range(len(parameter_list)): #to keep track of the summary of the posterior results
        results = utils.quantile(posterior_samples[:,p],[Quants[1], Quants[2], Quants[0]])
        p_mean, p_upper, p_lower = np.round(results[0],7), np.round(results[1]-results[0],8), np.round(results[0]-results[2],8) 
        results_tab.write(parameter_list[p]+'                 \t'+str(p_mean)+'\t'+str(p_upper)+'\t'+str(p_lower)+'\n')
        ndim+=1
    Global_file = ''
    folders = sub_folder.split('/')[:-1]
    for f in folders:
        Global_file += f+'/'
    Global_file += 'Global_LnZs.txt'
    Model, Results = utils.LoadResultsTxt(Global_file, deliminator='\t', noSpaces_tabs=True)
    try:
        IDX = Model.index(results_name)
        a_lnZ = Results[IDX][0]
        CorrectMissingLnZs = False
    except: #in some weird case a line in the Global_file is replaced with a number. If that happens correct it here
        import pymultinest
        output = pymultinest.Analyzer(outputfiles_basename=sub_folder+'/', n_params=ndim)
        a_lnZ = output.get_stats()['global evidence']
        CorrectMissingLnZs = True
    print ("Had to correct '"+Global_file+"' in '"+sub_folder+"'!!!")
    print ("global_evidence for model '"+results_name+"'="+str(a_lnZ))
    results_tab.write('###\n#Global lnZ: '+str(a_lnZ)+"\t ndim: "+str(len(parameter_list))+"\t sample_number: "+str(len(posterior_samples[:,0]))) #to print all stats needed for the Global_LnZs.txt file incase it's not written in that file for some reason
    results_tab.close()
    print ("rewrote 'fit_results.txt' file in directory: '"+sub_folder+"'")
    #To correct the Global_file for the missing variant info
    if CorrectMissingLnZs: #NOTE! this might crash if there are 2 cases in the same 'Global_lnZ.txt' file where the bug occurs. Fingers crossed I don't run into that
        SpLiT, out_path = sub_folder.split('/'), ''
        for f in range(len(SpLiT)-1):
            out_path+=SpLiT[f]+'/'
        Add_missing(out_path, Eparams_names, Max_Poly_order, WriteBestFit=True, DONTrunResults=True, JustCnt=False, RunResults=False, PrintVariants=False, Rm_Unfinished=False, ToPlot=False)
        CorrectMissingLnZs = False #Don't go through Add_missing() function again, unless need be
    return None 

def polynomial_fitting(time_input, flux_input, error_input, epar_inputs, Tduration, TransitPars, previousSubmits=[], poly_bnds=(-10,10), maxPolyOrder=3, Jobs=1, Gigs=8, Clean=False,\
    path='', outfile='white_light_parametric_fits', normalise_inputs=True, n_live=1000, plot_corner=False, plot_fit=False, wait_times=60, PRINTfinReslts=True, PLOTFinReslts=False,\
    fitting_utils_path='/home/mcgruderc/pool/CM_GradWrk/fitting_utils06.26.21/', UpdateC=3, Rerun_incompletes=False):

    """A function that fits all possible combinations of parametric model to out of transit data.
    This detrends against N given external parameters and fits polynomials (up to cubic) in each.

    Takes as input:
    time_input -- array of time. NOTE! must be on same time scale as t0!
    flux_input -- array of lc (normalized(target/comp)) flux
    error_input -- array of errors on flux measurements
    epar_inputs -- dictonary of arrays of external parameters to be used (ex: {'spec_shift':np.array(), 'cross_disp_px':np.array()
                -- 'fwhm_px':np.array(), 'airm_cen':np.array(), 'rot_ang_speed_rad_day':np.array()}). All external parameters should be normalized to 1
    t0 -- point of mid transit (in JD). If using shorten JD, make sure the shorten t0 JD matches the time stamps used
    Tduration -- duration of the transit, in hours 
    TransitPars  --  Mandel & Agol transit parameters: P (orbital period [days] of planet), t0 (time of mid transit), Rp/Rs, a/Rs, inclination, u1, u2. Assuming quadratic limb darkening.
    TransitPriors -- for right now, setting priors to normal, aside for the LD params, which will be set to unifrom. The bounds are just the sig uncertaintiy of P through inc params
    #processes -- (int) number of cpu cores to use
    #transit_priors  -- Dictionary of priors where each directory entry name is the name of the prior type (i.e. uniform, loguniform, normal, beta, exponential, or truncated_normal)
                    -- the input of the directory entry is a list with length that depends on the specific prior type. (e.g transit_priors = {'truncated_normal': [x, mu, sigma, a, b], 'exponential':[x, a]})
    Buffer -- extended buffer in time added to both ends of the calculated transit time (in minutes), to ensure no transit data is included
    maxPolyOrder -- the maximum polynomial order, in which you want to test in your system of polynomial fits
    #Poly_coeff_bnds -- 2-D tuple i.e (min1,max1). The bounds for each polynomial fit coefficient. For now, given each coefficient the same bounds
    n_live -- number of live points
    plotCorner -- If want to make the corner plot for each iteration
    ploFit -- If want to plot the best fit data for each iteration
    Jobs -- Total number of jobs that can be running at a time
    Clean -- True or False. If you want to wipe out the polynomial fit variants that don't have high enough logevidence to be used in final results
    Returns:

    best_chi2 -- the best reduced chi2 found from all models
    best_model -- the model that corresponds to the best reduced chi2
    best_coefficients -- the coefficients corresponding to the best model

    """
    if len(path) > 1 and path[-1] != '/': #to make sure there's a backslash at the parent directory
        path += '/'

    #To make sure time is given in full JD dates
    t0 = TransitPars['t0'][0]
    JD_base, JD_add = 2450000, 0 #starter that all jd dates have from Otober 9th 1995 to Feb 24th 2023
    JD_missing = len(str(JD_base))-len(str(t0).split('.')[0])
    if JD_missing > 3:
        sys.exit("t0 JD date is too small!, Was given t0 of", t0)
    if len(str(t0).split('.')[0]) < len(str(JD_base)):
        JD_add = str(JD_base)[:JD_missing]
        for s in range(len(str(JD_base))-JD_missing):
            JD_add+='0'
        JD_add = float(JD_add)
    time_input = time_input+JD_add
    TransitPars['t0'][0] = t0+JD_add


    ######################## INTERNAL FUNCTIONS ########################
    #To count how many jobs haven't been submitted
    def CheckPrevSubmits(previous_submits, JobTitle): #to create a list of jobs that we are waiting on 
        #how2handle_unsubmits = if wait, wait when a job is qued. if add, add it to the list of jobs that are being waited for
        fulL_path, running_jobs = path+outfile, []
        for rj in previous_submits:
            if os.path.isfile(rj+JobTitle+'.log'):
                last_line = utils.ReadLastLine(rj+JobTitle+'.log')
                if last_line: #if last_line is not None, which occurs if there was an error
                    if not last_line[:37] == '/cm/local/apps/uge/var/spool/compute-': #if the job isn't yet finished
                        running_jobs.append(rj)  #add it to the list of jobs we're waiting on
                else: #if can't read last line, then still waiting on this job to finish
                    running_jobs.append(rj)
            else: 
                #1st check to see if the analysis of this whole bin is already done
                Bin_path = '' #To keep track of the main bin that this particular variant is in
                sPlIt = rj.split('/')
                for s in sPlIt:
                    Bin_path += s+'/'
                if os.path.isfile(Bin_path[:-1]+'Global_LnZs.txt'):
                    with open(Bin_path[:-1]+'Global_LnZs.txt', 'r') as Global_LnZsF:
                        lines = Global_LnZsF.readlines()
                        if lines[-1][:17] == '#best fit model: ': #if the best fit is written at the end of the file
                            continue #Then analysis is already done, so DON'T add it to running_jobs list
                        else:
                            running_jobs.append(rj) #otherwise, must still be in the que, so waiting on that job too
                else:
                    running_jobs.append(rj) #if any of the previously submited job's logs don't exists AND can't find relavent Global_LnZ.txt file. Then must still be in the que, so waiting on that job too
        return running_jobs
   
    def SubmitJob(LC_dat, transitParams, EparArrys, EparNames, PlyOrders, BatmanBegins, Running_jobs, updatCriteria=3, Gigs=Gigs):
        #Running_jobs = list used to keep track of which jobs were submitted and haven't yet finished
        job_out_path, job_subfolder = path+outfile, ''
        JobTitle = 'parametricFitting'
        #get the subdirectory where all info for this specific Eparm variant will be stored
        plyorders_str = '[' #to write out the PlyOrders as a string, to be passed in the script written .py code
        if len(EparNames) == 0: #the case when there is only an offset fit for
            job_subfolder += 'Offset'+str(PlyOrders[0])
            plyorders_str = '['+str(PlyOrders[0])+']'
        for i in range(len(EparNames)):
            job_subfolder += EparNames[i]+'-'+str(PlyOrders[i])
            plyorders_str +=str(PlyOrders[i])
            if i < len(EparNames)-1:
                job_subfolder+= '_'
                plyorders_str+=','
            else:
                plyorders_str+=']'
        if not os.path.exists(job_out_path+'/'+job_subfolder):
            os.mkdir(job_out_path+'/'+job_subfolder)
        else: #if directory has already been made
            if os.path.isfile(job_out_path+'/'+job_subfolder+'/'+JobTitle+'.log'): #and log file already exists
                last_line = utils.ReadLastLine(job_out_path+'/'+job_subfolder+'/'+JobTitle+'.log')
                if Rerun_incompletes: #if want to re-run the incomplete variants, only return out of the function IF that variant is COMPLETELY done
                    if last_line[:37] == '/cm/local/apps/uge/var/spool/compute-': #otherwise, submit this job and DON'T add it to running jobs list
                        return Running_jobs
                else:
                    if last_line: #if last_line is not None
                        if not last_line[:37] == '/cm/local/apps/uge/var/spool/compute-': #if the job isn't yet finished
                            Running_jobs.append(job_out_path+'/'+job_subfolder+'/') #add it to the list of jobs we're waiting on
                    else:#if can't read last line, then still waiting on this job to finish
                        Running_jobs.append(job_out_path+'/'+job_subfolder+'/') #add it to the list of jobs we're waiting on
                    # Correct4Bugg(job_out_path+'/'+job_subfolder, EparNames, transitParams, PlyOrders, maxPolyOrder)
                    print ("'"+outfile+'/'+job_subfolder+"/' has already been submitted. Skipping this ...\n") #let us know that we aren't resubmitting this job regardless of if adding it to the list of running jobs or not
                    return Running_jobs #and stop function here
            #DELETE Initially thought I'd need code below, because it might imply that a job is in a que. HOWEVER, it's pretty unlikely that jobs would be queued after rerunning this code, which is the only case that this else statement would be valid
            #It's more likely that the .job file is created, but the code quit for some reason 
            # else: #if log file doesn't exists
            #     # if os.path.isfile(job_out_path+'/'+job_subfolder+'/'+JobTitle+'.job'): #only add job to job list if .job file was created. Otherwise assuming some error made the folder, but not submitted the job
            #     #     Running_jobs.append(job_out_path+'/'+job_subfolder+'/') # we are still def waiting on the job to finish, because we are assuming job is in que
            #     #     print ("'"+outfile+'/'+job_subfolder+"/' has already been submitted. Skipping this ...") #let us know that we aren't resubmitting this job regardless of if adding it to the list of running jobs or not
            #     #     return Running_jobs #and stop function here
            #     #Otherwise, job isn't submitted. Submit it down below

        ### Now check to see if too many jobs are running
        Repeat, waits = True, 0
        while Repeat:
            Running_jobs = CheckPrevSubmits(Running_jobs, JobTitle)
            RunningJobsCnt = len(Running_jobs)
            if RunningJobsCnt < Jobs: #if not, submit the next job
                #store all relavent data in this directory
                needed_data = {"transPar":transitParams, "epar_arrys_i":EparArrys, 'batman_init':BatmanBegins, 'LC':LC_dat}
                pickle.dump(needed_data, open(job_out_path+'/'+job_subfolder+"/needed_data.pkl", "wb"))
                #make python file that's for this specific iteration
                py_script = open(job_out_path+'/'+job_subfolder+'/'+JobTitle+'.py', 'w') 
                py_contents = 'import sys\nsys.path.append("'+fitting_utils_path+'")\n'
                py_contents += 'import numpy as np\nimport pickle\nimport PolyMultiNestClass as polyMN\n\n'
                py_contents += 'path = "'+job_out_path+'/'+job_subfolder+'"\nfile = open(path+"/needed_data.pkl","rb")\nneeded_data = pickle.load(file)\nfile.close()\n'
                py_contents += 'initilize = polyMN.Parametric_MultiNest(needed_data["LC"], needed_data["epar_arrys_i"], '+str(EparNames)+', '+plyorders_str
                py_contents += ', needed_data["transPar"], needed_data["batman_init"], '+str(Tduration)+') # to initilize a parametric_multiNest instance\n'
                py_contents += 'initilize.Run("'+job_out_path+'", '+str(n_live)+', plotCorner='+str(plot_corner)+', plotFit='+str(plot_fit)+')'            
                py_script.write(py_contents)
                py_script.close()

                #make job file
                job_script = open(job_out_path+'/'+job_subfolder+'/'+JobTitle+'.job', 'w')                    
                job_contents ="# /bin/sh\n# ----------------Parameters---------------------- #\n#$ -S /bin/sh\n"
                # if job_subfolder == 'Airmas-1_CrossDisp-2': #For some nearly every red bin uses a lot of memeory to run this variant. 
                #     Gigs = 30 #Need to investigate why, but for now just give it all the mem it needs
                if Gigs > 8: #when requesting more than 8 Gs, on the himem region. Must change job submission accordingly
                    job_contents += "#$ -q mThM.q\n#$ -l mres="+str(int(Gigs))+"G,h_data="+str(int(Gigs))+"G,h_vmem="+str(int(Gigs))+"G,himem\n#$ -cwd -j y\n"
                else:
                    job_contents += "#$ -q mThC.q\n#$ -l mres="+str(int(Gigs))+"G,h_data="+str(int(Gigs))+"G,h_vmem="+str(int(Gigs))+"G\n#$ -cwd -j y\n"
                job_contents +="#$ -N "+outfile+"_"+job_subfolder+"\n#$ -o "+job_out_path+'/'+job_subfolder+'/'+JobTitle+".log\n"
                job_contents +="#$ -m a\n#$ -M chima.mcgruder@cfa.harvard.edu" #it's going to be annoying AF to get the email for each of the jobs, but I need to do it so I can know if a code fails
                job_contents +='#\n# ----------------Modules------------------------- #\nexport MKL_NUM_THREADS=$NSLOTS\nexport NUMEXPR_NUM_THREADS=$NSLOTS\n'
                job_contents +="export OMP_NUM_THREADS=$NSLOTS\nexport OPENBLAS_NUM_THREADS=$NSLOTS\nexport VECLIB_MAXIMUM_THREADS=$NSLOTS\n"
                job_contents +='export PATH="/pool/sao_access/miniconda3/envs/access/bin/:$PATH"\n'
                job_contents +="export LD_LIBRARY_PATH=/pool/sao_access/MultiNest/lib:$LD_LIBRARY_PATH\n"
                job_contents +="export PATH=$PATH:$HOME/.local/bin/\n#\n# ----------------Your Commands------------------- #\n#\n"
                job_contents +="echo + `date` job $JOB_NAME started in $QUEUE with jobID=$JOB_ID on $HOSTNAME\n#\nmodule load tools/mthread-numpy\n"
                job_contents +="python "+job_out_path+'/'+job_subfolder+'/'+JobTitle+".py\n#\n"+repr("echo = \n")[1:-1]+"\necho = `date` $JOB_NAME done\n~"
                job_script.write(job_contents)
                job_script.close()
                Repeat = False
            else: # else wait until one of the jobs finish
                # print ("too many jobs running["+str(RunningJobsCnt)+"]. Waiting "+str(wait_times)+"sec for one to finish...")
                waits +=1
                if waits > updatCriteria-1: #every time waited for 'updatCriteria' iterations, print what's taking so damn long
                    print ("too many jobs running["+str(RunningJobsCnt)+"] running. Waited for "+str(wait_times*waits)+"seconds for one to finish...")
                    print ("waiting on the following jobs:",Running_jobs)
                    waits = 0 #rest, to give more time before printing all jobs we waiting on again
                time.sleep(wait_times)
                Repeat = True #then go back and check again
        os.system('qsub '+job_out_path+'/'+job_subfolder+'/'+JobTitle+'.job')
        print ('\n')
        Running_jobs.append(job_out_path+'/'+job_subfolder+'/')
        # print ("waiting "+str(wait_times)+"sec for the submitted job to initilize")
        # time.sleep(wait_times)
        return Running_jobs
    ######################## END of internal functions ########################
    lc_dat = [time_input, flux_input, error_input] #save the parameters for each bin

    if os.path.exists(path+outfile): #1st check if results folder exists
        if os.path.isfile(path+outfile+'/Global_LnZs.txt'): #then check if LnZs file exists
            with open(path+outfile+'/Global_LnZs.txt', 'r') as Global_LnZs:
                lines = Global_LnZs.readlines()
                if lines[-1][:17] == '#best fit model: ': #2nd check if the best fit is written at the end of the file
                    print ('Analysis already done!')
                    if PRINTfinReslts:
                        Results(path+outfile, lc_dat, TransitPars, Tduration, epar_inputs, write_bestFit=False, PLOT=PLOTFinReslts) #to print the final results. HOWEVER! Not plotting it, because Hydra can't plot in terminal
                    if Clean:
                        ClearUnusedBMAs(path+outfile)
                    return [] #return empty list, signifying that we aren't waiting on any jobs
        else: #if not, initiate the LnZs file
            LnZs_tab = open(path+outfile+'/Global_LnZs.txt','w')
            LnZs_tab.write('#model\t\t\t\tglobal_evidence\t\t\t\tsample_number\n')
            LnZs_tab.close()            
    else: #if file doesn't exits. Create it and continue through process
        os.mkdir(path+outfile)
        LnZs_tab = open(path+outfile+'/Global_LnZs.txt','w')
        LnZs_tab.write('#model\t\t\t\tglobal_evidence\t\t\t\tsample_number\n')
        LnZs_tab.close()   

    # generate boolean arrays of all possible combinations of True and False for an array of length number of external parameters
    # quntL, Mean, quntH = 0.15865,0.50, 0.84135 #upper and lower bound of 1sig and mean value
    epars = list(epar_inputs.keys())
    all_possible_combinations = itertools.product([True,False],repeat=len(epars))

    poly_orders = np.arange(1, maxPolyOrder+1) #+1 because arange last index is not inclusive

    #to initiate batman to be used in sampling and define global variables
    bat_transPar = [TransitPars['P'][0], TransitPars['t0'][0], TransitPars['Rp/Rs'][0], TransitPars['a/Rs'][0], TransitPars['inc'][0], TransitPars['u1'][0], TransitPars['u2'][0]]
    params, m = utils.batman_model(bat_transPar,time_input)
    mean_sigma, data = np.mean(error_input), flux_input

    ModelsTsted = 0 # To count total models tested
    for combination in all_possible_combinations: 
        #list of all the external parameter names and array values used in this iteration
        epar_names_i, epar_arrys_i = [], []
        for c in range(len(combination)):# switch on/off the detrend inputs
            if combination[c]: #because using the same epars list from the begining, always going to have the same order of epar names
                epar_names_i.append(epars[c]) #For example, if epars is ['time', 'airmass', etc.] epars[0] will always be TIME, epars[1] will always be AIRMASS, etc. and we will run through all possible t/f options
                epar_arrys_i.append(epar_inputs[epars[c]])
        all_possible_polys = itertools.product(list(poly_orders),repeat=len(epar_names_i))
        for app in all_possible_polys: #to have each used external parameter fit a polynomial of each allowed order
            ply_orders_i = np.array(list(app))
            if ply_orders_i.shape[0] == 0: #if there are no external parameters
                ply_orders_i = np.array([0]) #then test the case when there is just an offset term
            ModelsTsted +=1
            if Jobs > 1:
                previousSubmits = SubmitJob(lc_dat, TransitPars, epar_arrys_i, epar_names_i, ply_orders_i, [params,m], previousSubmits, updatCriteria=UpdateC)
            else: #otherwise just run the multinest routine in serial, here  
                import PolyMultiNestClass as polyMN
                initilize = polyMN.Parametric_MultiNest(lc_dat, epar_arrys_i, epar_names_i, ply_orders_i, TransitPars, poly_bnds, [params,m]) # to initilize a parametric_multiNest instance
                initilize.Run(path+outfile+'/', n_live, Tduration, plotCorner=plot_corner, plotFit=plot_fit)  # Run MultiNest 
        print ('\n')       	
    print ("Total models being tested:", ModelsTsted)

    # # To keep track of when all submitted jobs are done running, and prevent continuation until they are all done
    # running_jobs = previousSubmits.copy()
    # while len(running_jobs) > 0:
    #     WAIT = wait_times*5
    #     running_jobs = CheckPrevSubmits(running_jobs, JobTitle)
    #     if len(running_jobs) > 0:
    #         print ('waiting on the following '+str(len(running_jobs))+' jobs to finish:', running_jobs)
    #         print ("\nWaiting ", (WAIT)/60, 'minutes for codes to finish')
    #         time.sleep(WAIT)

    if PRINTfinReslts:
        Results(path+outfile, lc_dat, TransitPars, Tduration, epar_inputs, write_bestFit=True, PLOT=PLOTFinReslts) #to print the final results. HOWEVER! Not plotting it, because Hydra can't plot in terminal
    if Clean:
        ClearUnusedBMAs(path+outfile)
    return previousSubmits

def ClearUnusedBMAs(Folder, ResultsFile='BMA_final_results.txt', diff='###Total_samples=', deliminator='\t'): #to delete folders that have 0 weight in the BMA to form the final LC
    #This is done for each bin in the subfolder
    import shutil
    #diff = the line, where every line following it, will be information about which variants are used
    folderContents = glob.glob(Folder+'/*') #to pull out ALL contents in dirctory
    for fc in folderContents: #for each bin in our master directory
        ToKeep = []
        if not os.path.isdir(fc): 
            continue
        else: #1st check to make sure this specific item is a folder
            if os.path.isfile(fc+'/'+ResultsFile): #if have the results info for this bin
                txt = open(fc+'/'+ResultsFile, 'r')
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
                            ToKeep.append(fc+'/'+name) #what not to toss
            else:
                print ("Couldn't find '"+ResultsFile+"' file, skipping folder ...")
                continue
        Variants = glob.glob(fc+'/*')
        print ("For '", fc+"'")
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
    return None

def SimgaClipping(LC, Nig=3.01,  Nneighbors=10, ExtraPars=None): 
    """sigma clipping, to get rid of the outliers. If 'Nsig'-simga further than the surrounds 'Nneighbors' points ('Nneighbors'/2 bck, 'Nneighbors'/2 front)"""
    #LC = 3 element list of light-curve info (i.e. [time, flux, flx_err]) #Nsig = how many standard deviations a given element can vary from another
    #Nneighbors = How many surrounding datapoints that will be used to calculate the standard deviation from. 
    #            Want this to be pretty low, so doesn't get confused by astrophysical features (i.e. the transit)
    #ExtraPars = dictonary of external parameters (epars as arrays). Because clipping transit data, need to also clip the external parameters so they are consistent with one another
    if Nneighbors%2: 
        Nneighbors+=1 #want Neighbors to be even, so can have equal number of datapoints infront asn behind the datapoint of interest
    if ExtraPars is not None:
        keys = list(ExtraPars.keys())
    i, TrueIdx, removed_ideces = 0, 0, []
    time, flux, err = LC[0].copy(), LC[1].copy(), LC[2].copy()
    while i < len(time):
        if i < Nneighbors/2.0: #Then at the front edge of the data and there will be more ahead points than behind points
            ahead = Nneighbors-i
            behind = Nneighbors-ahead
        elif len(time)-1 < i+Nneighbors/2.0: #Then at the back edge of the data and there will be more behind points than ahead points
            ahead = len(time)-1-i #-1 because python starts at 0
            behind = Nneighbors-ahead
        else:
            ahead, behind = Nneighbors/2.0, Nneighbors/2.0
        bck_idx, frnt_idx = int(i-behind), int(i+ahead)
        surround = np.append(flux[bck_idx:i],flux[i+1:frnt_idx+1])#get the values of the surrounding indeces, not including the actual index of interest because would skew std if it's actually on outlier
        sigma, mean = np.std(surround), np.mean(surround)
#         surround_t = np.append(time[bck_idx:i],time[i+1:frnt_idx+1])
#         print ("time[i]:", time[i], "i:", i, "sigma:", sigma, "mean:", mean)
#         print ("surround_t:", surround_t)
#         print ("sigma, mean, time[i]", sigma, mean, time[i])
#         print ("surrond:", surround)
        if abs(flux[i]-mean) > Nig*sigma:
            print ("removing timestamp:", time[i])
            time, flux, err = np.delete(time,i), np.delete(flux,i), np.delete(err,i)
            removed_ideces.append(TrueIdx) #to keep track of the original index of datapoints that have been clipped
            i = i - int(Nneighbors/2) #if removed an outlier, go back and check the other datapoints that had the outlier in the std calculation
            if ExtraPars is not None:
                for k in keys:
                    ExtraPars[k] = np.delete(ExtraPars[k],i)
        else:
            i+=1
        extras = len(np.where(np.array(removed_ideces) < TrueIdx)[0]) #indecies that have already been removed, but need to be counted for the TrueIdx counts
        TrueIdx = i+extras
        # print("\n")
    per_removed = (len(removed_ideces)/len(LC[0]))*100.0
    print ("removed "+str(len(removed_ideces))+" out of "+str(len(LC[0]))+" files. "+str(np.round(per_removed, 2))+"%")
    LC = [time, flux, err]
    if ExtraPars is not None:
        return LC, ExtraPars, per_removed, removed_ideces
    else:
        return LC, per_removed, removed_ideces

def GetRawLC(LC,TransDuration,t0,good_frames=None,used_comps=None,Bin=None, Buffer=2): 
    """determine the raw white-light lc by dividing the sum of the desired comp stars by the target lc"""
    #good_frames=list of 2 element list. range of useful frames. if None, assuming all frames are good. i.e. [[0,20],[25,100],[130,160]]
    #used_comps=list of desired comp stars for deterending. Accepts list of names of comps or list of int values representing each of the comps used in same order as LC 'cNames'
    #bin = int, the count of the bin you'd want to get the raw lc of. If None, assuming working with white-light curve
    if type(Bin) == int: #had to do this instead of "if not Bin" because Bin == 0 is the same thing as it being None
        LC_trg, LC_cmps = LC['oLCw'][:,Bin], LC['cLCw'][:,:,Bin]
    else:
        LC_trg, LC_cmps = LC['oLC'], LC['cLC']
    if good_frames: #to only use good frames
        gf_cnt = 0
        for Range in good_frames:
            if gf_cnt == 0: #to initiate the arrays with the right shapes
                T1, fin_oLC, fin_cLC = LC['t'][Range[0]:Range[1]], LC_trg[Range[0]:Range[1]], LC_cmps[Range[0]:Range[1],:]
            else:
                T1 = np.concatenate((T1, LC['t'][Range[0]:Range[1]]))
                fin_oLC = np.concatenate((fin_oLC, LC_trg[Range[0]:Range[1]]))
                fin_cLC = np.concatenate((fin_cLC, LC_cmps[Range[0]:Range[1]]), axis=0)
            gf_cnt +=1
    else:
        T1, fin_oLC, fin_cLC = LC['t'], LC_trg, LC_cmps
    if used_comps:
        if type(used_comps[0]) == int: #assume list is the int values of comps that can be used. in the same order as 'cNames'
            summed_comps = np.sum(fin_cLC[:,used_comps], axis=1)
        if type(used_comps[0]) == str: #assume list is the exact names of comps to be included (NOT case senstive)
            used_comp_ints = []
            Lowered_cNames = list((map(lambda x: x.lower(), LC['cNames']))) #to remove the case sensitivity by converting all string elements in list to lower case
            for c in used_comps: 
                used_comp_ints.append(Lowered_cNames.index(c.lower())) #convert c to lower case to be consistent with 'Lowered_cNames'
            summed_comps = np.sum(fin_cLC[:,used_comp_ints], axis=1)
    else: #if no comps were specified, assuming using all comps
        summed_comps = np.sum(fin_cLC, axis=1)
    RawLC = fin_oLC/summed_comps
    #To normalize the lc by out of transit data
    Buff_days = Buffer/(24.0*60) #minutes to day
    Duration_days = TransDuration/24.0 #hrs to days
    ingres, egres = t0-Duration_days/2.0, t0+Duration_days/2.0
    OOT1, OOT2 = np.where((T1<=ingres-Buff_days))[0], np.where((T1>=egres+Buff_days))[0]
    OOT = np.append(OOT1,OOT2)
    return T1, RawLC/np.mean(RawLC[OOT]) 

def PlotFinalBMA(lc, Tran_dur, FixedTransPars, PosteriorPath, path_outfile, SaveAsNPY, PLOT=False): 
    """ To print, save, and plot the BMA results """
    #lc = the original lc data #FixedTransPars = dictionary of all the batman transit parameters, held fixed #Tran_dur = duration of transit
    #PosteriorPath = string path to the file where the 'BMA_final_results.txt' and 'equal_weighted_BMAposteriors.pkl' results files are located (should both be found in the same path)
    #SaveAsNPY = if you want to save the data used for plotting. Can be either boolan, or a string consisting of the path name where the data should be saved. DON'T FORGET TO INCLUDE THE '/' at the end of the last path!!
    #PLOT = same as 'SaveAsNPY', but for the .png figure #path_outfile = where plot and save .npy file will be located 
    TransParOrder = ['P', 't0', 'Rp/Rs', 'a/Rs', 'inc', 'u1', 'u2']
    PostSum = np.loadtxt(PosteriorPath+'/BMA_final_results.txt', dtype=str)
    Params, Means = list(PostSum[:,0]), PostSum[:,1].astype(float)
    FixedParsNames = list(FixedTransPars.keys())
    mean_trans_pars = []
    for TPO in TransParOrder: #to poplulate mean_trans_pars in the right order
        if TPO in Params:
            Par_idx = Params.index(TPO)
            mean_trans_pars.append(Means[Par_idx])
        elif TPO in FixedParsNames:
            mean_trans_pars.append(FixedTransPars[TPO])
        else:
            print ("ERROR!!! Missing parameter '"+TPO+".' Can't find it in 'FixedParsNames' or 'BMA_final_results.txt'")
            print ("'BMA_final_results.txt's parameters:", Params, "\n'FixedParsNames's parameters:", FixedParsNames)
            sys.exit("EXITING!")
    BMAposteriors = pickle.load(open(PosteriorPath+'/equal_weighted_BMAposteriors.pkl', 'rb'), encoding="latin1")
    parametric_fits = BMAposteriors["parametric_fits"]
    if type(PLOT) == str:
        figName = PLOT
    else: 
        figName = 'BMAmodel'
    utils.PlotLC(lc, 'These are', 'place', 'holders', Tran_dur, mean_trans_pars, path_outfile, plot_name=figName, poly_model=np.mean(parametric_fits, axis=1), save_dats=SaveAsNPY)
    plt.close()     

if __name__ == "__main__":
    ############## INITIAL INPUT ##############
    ##### Target Specific
    file_name = 'LCs_w96_180ob2_noTel' #'LCs_w96_Ob2_Final2' 'FORS2R_Final2' 'FinFORS2BnoTel' 'LCsOb2Final2_noTel' 'LCs_w96_250ob2' 'FORS2_250B', 'FORS2_180B_noTel', 'LCs_w96_180ob1_noTel'
    epar_file = 'eparams_IMACSob2.dat' #eparams_IMACSob1_nobad, eparams_IMACSob2.dat, eparams_FORS2R.dat
    epar_names = ['Airmass','DeltaWav','FWHM','SkyFlux','TraceCenter'] #['TargSpecShift', 'CrossDisp', 'FWHM', 'Airmass', 'Bckg2dspec', 'RotAng'] # # NOTE!!! CAN'T have spaces or underscores "_" in dictionary name!!!! #to keep track of which external parameter is being loaded and in the SAME ORDER
    # t0, rprs, q1, q2 =  2457963.3366167303, 0.1147791690, 0.3657441039, 0.2941807425 #ut170729 (FORS2 blue) 
    # t0, rprs, q1, q2 =  2457970.6909287609, 0.1167987595, 0.2297468318, 0.3337791805 #ut170804 (ACCESS ob1)
    # t0, rprs, q1, q2 =  2457987.3121593646, 0.1188224937, 0.3104588535, 0.6316688087 #ut170822 (FORS2 red) 
    t0, rprs, q1, q2 =  2458066.5970282881, 0.1018326820, 0.3194787602, 0.3137249979 #ut171108 (ACCESS ob2)  
    Target = 'UT171108'
    UsedComps = ['COMP14','COMP15'] #None
    GoodFrames = None #[[0,148],] 

    ##### General operations
    PLOT = False
    full_path = '/home/mcgruderc/pool/CM_GradWrk/GPTransmissionSpectra3/'
    Base1, Base2 = full_path+'WASP-96/', full_path+'outputs/WASP-96/' #path to the initial pkl folder and the eparm folder
    eparm_dat = np.genfromtxt(Base2+epar_file, unpack=True) #Need to load the original eparams file that doesn't have the 3sigma clipped data already removed, because will do each bin's sigma clipping on their own
    if len(epar_names) != eparm_dat.shape[0]:
        sys.exit("len of epar_names ("+str(len(epar_names))+") != eparm_dat.shape ("+str(eparm_dat.shape[0])+")") 
    EparInputs = {}
    for en in range(len(epar_names)):
        EparInputs[epar_names[en]] = eparm_dat[en,:]
    transPar = {'P':[3.4252602, None], 't0':[t0, None], 'Rp/Rs':[rprs, 'normal', 0.05], 'a/Rs':[8.84, None], 'inc':[85.14, None], 'q1':[q1, 'uniform', [0,1]], 'q2':[q2, 'uniform', [0,1]]} #fix all parameters to wl fit except for Rp/Rs (prior on wl fit), and q1/q2 (uniform from 0-1)
    tranDur = 2+(26/60) # transPar = [3.4252602, 2457987.31195, 0.1172, 8.80, 85.11, .26, .23]#[3.4252602, 2457963.33672, 0.1141, 8.93, 85.21, .339, .148]
    #########################################

    # ClearUnusedBMAs('SynthDatPkls/IterationTst/Binned/')
    # Bin = 14
    # OutFle = 'Binned_parametric_model_fitsB/Bin'+str(Bin)
    # ligh_kurv = [Time, FLUX, Flux_err]
    # polynomial_fitting(Time, FLUX, Flux_err, EparInputs, tranDur, transPar, transPriors, poly_bnds=polynol_bnds, outfile=OutFle, n_live=300, maxPolyOrder=2, plot_corner=False, plot_fit=True,  Jobs=30, wait_times=40) #plotting the corner plots uses WAY too much memeory
    # Add_missing(OutFle, ['W96SpecShift', 'RotAng', 'Airmas', 'FWHM', 'CrossDisp'], transPriors, LC=ligh_kurv, TransPar=transPar, TranDur=tranDur,WriteBestFit=False)


    # OutFle = 'parametric_model_fits_WL_test1'
    # sub_dir = 'W96SpecShift-1_RotAng-1_Airmas-1_FWHM-1_CrossDisp-1'
    # Plot_Corner(OutFle+'/'+sub_dir)
    # Plot_LC(Time, FLUX, np.ones(len(FLUX))*(400/1e6), transPar, tranDur, EparInputs, OutFle+'/'+sub_dir)
    # print ("\n")
    # Plot_Corner('parametric_model_fits_WL_test1/W96SpecShift-1_RotAng-1_Airmas-1_FWHM-1_CrossDisp-1')
    # Plot_LC(Time, FLUX, np.ones(len(FLUX))*(400/1e6), transPar, tranDur, EparInputs, 'parametric_model_fits_WL_test1/W96SpecShift-1_RotAng-1_Airmas-1_FWHM-1_CrossDisp-1')
    
    # # ###############################################################################################
    #To pull out the light curve info
    LCfile = Base1+file_name+'.pkl' #LCfile = 'SynthDatPkls/Iteration0/FinalLC_0.pkl'
    with open(LCfile, 'rb') as pkl_file:
        LC_ob1 = pickle.load(pkl_file, encoding="latin1")
    Bins = LC_ob1['wbins']
    TIME, FLUX = GetRawLC(LC_ob1, tranDur, transPar['t0'][0], good_frames=GoodFrames,used_comps=UsedComps)
    # #To keep record of which indecies are not used
    # clipped_LC, percentClipped, OmitIdx = SimgaClipping([TIME, FLUX, np.ones(len(FLUX))], Nig=3.01, Nneighbors=10)
    # print ("TIME[OmitIdx]:", TIME[OmitIdx], "OmitIdx:", OmitIdx)
    # sys.exit()

    #To Estimate the modeling error 
    PCA1_time, PCA1_DetFlux, PCA1_DetFluxErr, PCA1_Model = np.loadtxt(Base2+file_name+'/white-light/PCA_1/detrended_lc.dat', unpack=True) #always just use the 1st PCA to estmate uncertainity. It's an estimate anyways
    resid = PCA1_DetFlux-PCA1_Model
    wl_ErrLvl = np.std(resid,ddof=1)
    Bin_ErrLvl = wl_ErrLvl*np.sqrt(len(Bins)) #assuming 1) that the error is dominated by photon noise, 2) each bin has equal number of counts
    # Bin_ErrLvl *= 3 #Specifically for the FORS2B data, because I think the low estimated error bars are messing up the convergence of the fits? Use similar uncertainties as Red spectra
    print ("wl_ErrLvl [ppm]:", wl_ErrLvl*1e6, "Binned_ErrLvl [ppm]:", Bin_ErrLvl*1e6)
    print ("t0:", t0, "\trprs:", rprs, "\tq1:", q1, "\tq2:", q2)

    #### Use CMC to correct for binned data. Then use BMA to determine the best polynomial fit to correct for remaining systematics
    #create white-light model from best fit white-light curve from Nestor's Code (which does a GP fit)
    params = batman.TransitParams()
    params.t0 = transPar['t0'][0]                       #time of inferior conjunction
    params.per = transPar['P'][0]                       #orbital period
    params.rp = transPar['Rp/Rs'][0]                        #planet radius (in units of stellar radii)
    params.a = transPar['a/Rs'][0]                         #semi-major axis (in units of stellar radii)
    params.inc = transPar['inc'][0]                      #orbital inclination (in degrees)
    params.ecc = 0.                     #eccentricity
    params.w = 90.                        #longitude of periastron (in degrees)
    params.limb_dark = "quadratic"        #limb darkening model
    if 'q1' in list(transPar.keys()): #then assuming the q1/q2 parameterization of the ld parameters was given
        u1, u2 = utils.reverse_ld_coeffs("quadratic", transPar['q1'][0], transPar['q2'][0]) #TODO!!! need to edit 'PolyMultiNestClass.py' to take in either q1/q2 or u1/u2
        transPar['u1'], transPar['u2'] = [u1, 'uniform', [0,1]], [u2, 'uniform', [0,1]] #Even though.....!!!
    else:
        u1, u2 = transPar['u1'][0], transPar['u2'][0]
    params.u = [u1, u2]  #limb darkening coefficients [u1, u2, u3, u4]
    m = batman.TransitModel(params, TIME)
    model_flux = m.light_curve(params)

    CMC = FLUX/model_flux
    Flux_err = np.ones(len(FLUX))*Bin_ErrLvl

    Path = Base2+file_name+'/Binned' # Path = 'SynthDatPkls/Iteration0/Binned'
    if not os.path.exists(Path):
        os.mkdir(Path)

    # blue_u2 = [0.207,0.219,0.223,0.227,0.231,0.235,0.239,0.242,0.246,0.251,0.255,0.256,0.258,0.262,0.269,0.273,0.275,0.277,0.278,0.283,0.287,0.290,0.291,0.295,0.296,0.296,0.297,0.302]
    # red_u2 = [0.277,0.278,0.283,0.287,0.290,0.291,0.295,0.296,0.296,0.297,0.302,0.313,0.314,0.314,0.314,0.313,0.313,0.313,0.314,0.314,0.314,0.314,0.315,0.315,0.315,0.316,0.316,0.315,0.315,0.315,0.314,0.314]
    # #### To run the BMA polynomial fitting routine 
    # Submitted, PerClip = [], [] #to keep a running list of submitted jobs that aren't finished, #and the percent of datapoints clipped per bin
    # for b in range(len(Bins)):
    #     print ("Starting submission of jobs for Bin "+str(b))
    #     # if b == 2: #for some reason there are a few variants that need a lot of memory. Give it to em
    #     #     sys.exit()
    #     time_i, flux_i = GetRawLC(LC_ob1, tranDur, transPar['t0'][0], good_frames=GoodFrames, used_comps=UsedComps, Bin=b) #LC_ob1['oLCw'][:,b]/LC_ob1['cLCw'][:,0,b]
    #     cmcFlux = flux_i/CMC
    #     clipped_epars = EparInputs.copy()
    #     clipped_LC, clipped_epars, percentClipped, OmitIdx = SimgaClipping([time_i, cmcFlux, Flux_err], Nig=3.01, Nneighbors=10, ExtraPars=clipped_epars)
    #     PerClip.append(percentClipped)
    #     if PLOT:
    #         plt.figure(b)
    #         plt.plot(time_i, cmcFlux, 'r.')
    #         plt.plot(clipped_LC[0], clipped_LC[1], 'b.')
    #         Zer0 = np.zeros(len(clipped_LC[1]))
    #         plt.errorbar(clipped_LC[0], clipped_LC[1], [clipped_LC[2],clipped_LC[2]], [Zer0,Zer0], 'b.')
    #         plt.savefig("CM_corr_"+Target+'_Bin'+str(b))
    #         plt.close()
    #     # if b < 7:
    #     # transPar['u2'][0] = red_u2[b]
    #     Submitted = polynomial_fitting(clipped_LC[0], clipped_LC[1], clipped_LC[2], clipped_epars, tranDur, transPar, previousSubmits=Submitted, path=Path, outfile='Bin'+str(b), n_live=500, maxPolyOrder=2, plot_corner=True, plot_fit=True,  Jobs=30, wait_times=1, PRINTfinReslts=False, UpdateC=30, Gigs=12, Rerun_incompletes=False) #plotting the corner plots uses WAY too much memeory
    #     print ("\n\n\n")
    # print ("Mean percent of datapoints clipped:", str(np.round(np.mean(PerClip),3))+"%")
    
    #### To run the results, because plotting and can't do that in terminal
    for b in range(len(Bins)):
        if not os.path.exists(Path+'/'+'Bin'+str(b)):
            print ("The folder for Bin"+str(b)+" hasn't even been created yet! skipping ...")
            print("##########################\n#########################\n########################\n")
            continue
            # sys.exit()
        print ("for Bin"+str(b))
        time_i, flux_i = GetRawLC(LC_ob1, tranDur, transPar['t0'][0], good_frames=GoodFrames, used_comps=UsedComps, Bin=b) #LC_ob1['oLCw'][:,b]/LC_ob1['cLCw'][:,0,b]
        cmcFlux = flux_i/CMC
        clipped_epars = EparInputs.copy()
        clipped_LC, clipped_epars, percentClipped, OmitIdx = SimgaClipping([time_i, cmcFlux, Flux_err], Nig=3.01, Nneighbors=10, ExtraPars=clipped_epars)
        # To finish the LC analysis (produce the BMA results and the final LCs)
        Add_missing(Path+'/'+'Bin'+str(b), list(EparInputs.keys()), 2,  TransPar=transPar, TranDur=tranDur, LC=[clipped_LC[0], clipped_LC[1], clipped_LC[2]], epar_inpts=clipped_epars, WriteBestFit=True, PrintVariants=False, RunResults=True, Rm_Unfinished=False, ToPlot=True, saveNPY=Path+'/NPY/Bin'+str(b)+"_")
        ## To save the final data needed for plotting
        if not os.path.isdir(Path+'/NPY/'): #if the folder doesn't exist, 
            os.makedirs(Path+'/NPY/', exist_ok=True) # create it
        np.save(Path+'/NPY/Bin'+str(b)+'_RawLC', np.array([time_i, flux_i]))
        np.save(Path+'/NPY/Bin'+str(b)+'_unclippedCMClc', np.array([time_i, cmcFlux, Flux_err]))
        FixdTransPar = {} #to create a directory of ONLY the fixed parameters
        for k in list(transPar.keys()):
            if not transPar[k][1]:
               FixdTransPar[k] =  transPar[k][0]
        PlotFinalBMA([clipped_LC[0], clipped_LC[1], clipped_LC[2]], tranDur, FixdTransPar, Path+'/'+'Bin'+str(b)+'/', Path+'/NPY/', Path+'/NPY/', PLOT='Bin'+str(b)+'_BMAmodel')
        print("##########################\n#########################\n########################\n")
    # sub_dir = 'CrossDisp-1'
    # Plot_Corner(OutFle+'/'+sub_dir)
    # Plot_LC(Time, cmcFlux, Flux_err, transPar, tranDur, EparInputs, OutFle+'/'+sub_dir)
    # print ("\n")

    # sub_dir = 'W96SpecShift-1'
    # Plot_Corner(OutFle+'/'+sub_dir)
    # Plot_LC(Time, cmcFlux, Flux_err, transPar, tranDur, EparInputs, OutFle+'/'+sub_dir)
    # print ("\n")

    # sub_dir = 'W96SpecShift-2'
    # Plot_Corner(OutFle+'/'+sub_dir)
    # Plot_LC(Time, cmcFlux, Flux_err, transPar, tranDur, EparInputs, OutFle+'/'+sub_dir)
    # print ("\n")