from scipy.stats import gamma, norm, beta, truncnorm
import numpy as np
import pickle
import batman
import os
import sys
import matplotlib.pyplot as plt
import itertools

def CheckPrev_Submits(previous_submits): #to create a list of jobs that we are waiting on 
    #how2handle_unsubmits = if wait, wait when a job is qued. if add, add it to the list of jobs that are being waited for
    running_jobs = []
    for rj in previous_submits:
        if os.path.isfile(rj+'.log'):
            last_line = ReadLastLine(rj+'.log')
            if last_line: #if last_line is not None, which occurs if there was an
                if not last_line[:37] == '/cm/local/apps/uge/var/spool/compute-':
                    running_jobs.append(rj) #then the job has NOT finished and have to wait for it
            else: #if can't read last line, then still waiting on this job to finish
                running_jobs.append(rj)
        else:
            running_jobs.append(rj) #if any of the previously submited job's logs don't exists. Then must still be in the que, so waiting on that job too
    return running_jobs
      
def load_pkl(fpath):
    with open(fpath, "rb") as f:
        data = pickle.load(f)
    return data

# Define transit-related functions:
def reverse_ld_coeffs(ld_law, q1, q2):
    if ld_law == "quadratic":
        coeff1 = 2.0 * np.sqrt(q1) * q2
        coeff2 = np.sqrt(q1) * (1.0 - 2.0 * q2)
    elif ld_law == "squareroot":
        coeff1 = np.sqrt(q1) * (1.0 - 2.0 * q2)
        coeff2 = 2.0 * np.sqrt(q1) * q2
    elif ld_law == "logarithmic":
        coeff1 = 1.0 - np.sqrt(q1) * q2
        coeff2 = 1.0 - np.sqrt(q1)
    elif ld_law == "linear":
        return q1, q2
    return coeff1, coeff2

#to read the last line of a large file
def ReadLastLine(path_n_file, printErr=False): #file must include the FULL path to the file
    try: #try to get the last line
        with open(path_n_file, 'rb') as f: #check here if end of file is finished
            f.seek(-2, os.SEEK_END)
            while f.read(1) != b'\n':
                f.seek(-2, os.SEEK_CUR)
            last_line = f.readline().decode()
        return last_line
    except: #but in some one cases, can't because log file is still being created.
        if printErr:
            print ("Unable to get last line!!!")
            print ("\n##########################")
            print ("Unable to get last line!!!\nf:", f, "\nos.SEEK_CUR:", os.SEEK_CUR)
            print ("##########################\n")
        return None

# READ OPTION FILE:
def _to_arr(idx_or_slc):
    # Converts str to 1d numpy array
    # or slice to numpy array of ints.
    # This format makes it easier for flattening multiple arrays in `_bad_idxs`
    # NOTE: bounds are inclusive like the good lord intended
    if ":" in idx_or_slc:
        lower, upper = map(int, idx_or_slc.split(":"))
        return np.arange(lower, upper + 1)
    else:
        return np.array([int(idx_or_slc)])


def _bad_idxs(s):
    if s == "[]":
        return []
    else:
        # Merges indices/slices specified in `s` into a single numpy array of
        # indices to omit
        s = s.strip("[]").split(",")
        bad_idxs = list(map(_to_arr, s))
        bad_idxs = np.concatenate(bad_idxs, axis=0)
        return bad_idxs


# TRANSFORMATION OF PRIORS:
transform_uniform = lambda x,a,b: a+(b-a)*x

transform_normal = lambda x,mu,sigma: norm.ppf(x, loc=mu, scale=sigma)

transform_beta = lambda x,a,b: beta.ppf(x, a, b)

transform_exponential = lambda x, a=1.0: gamma.ppf(x, a)

def transform_loguniform(x, a, b):
    la = np.log(a)
    lb = np.log(b)
    return np.exp(la + x * (lb - la))

def transform_truncated_normal(x, mu, sigma, a=0.0, b=1.0):
    ar, br = (a - mu) / sigma, (b - mu) / sigma
    return truncnorm.ppf(x, ar, br, loc=mu, scale=sigma)

def ChoosePrior(cube, param): #to select whichever prior is appropriate for a given paramaeter, based on the globally defined Params dictionary
    if param[1].lower() == 'uniform':
        return transform_uniform(cube, *param[2])
    elif param[1].lower() == 'normal' or param[1].lower() == 'gaussian':
        if type(param[2]) == int or type(param[2]) == float:
            return transform_normal(cube, param[0], param[2])
        else: #sometimes a float/int is passed, other times a 1d list/array
            return transform_normal(cube, param[0], param[2][0])
    elif param[1].lower() == 'beta':
        return transform_beta(cube, *param[2])
    elif param[1].lower() == 'exponential':
        return transform_exponential(cube, param[2][0])
    elif param[1].lower() == 'log-uniform':
        return transform_loguniform(cube, *param[2])
    elif param[1].lower() == 'trunc-normal':
        return transform_truncated_normal(cube, param[0], *param[2])
    else:
        sys.exit("prior function of '"+param[1].lower()+"' is unknown!")

# PCA TOOLS:
def get_sigma(x):
    """
    This function returns the MAD-based standard-deviation.
    """
    median = np.median(x)
    mad = np.median(np.abs(x - median))
    return 1.4826 * mad


def standarize_data(input_data):
    output_data = np.copy(input_data)
    averages = np.median(input_data, axis=1)
    for i in range(len(averages)):
        sigma = get_sigma(output_data[i, :])
        output_data[i, :] = output_data[i, :] - averages[i]
        output_data[i, :] = output_data[i, :] / sigma
    return output_data


def classic_PCA(Input_Data, standarize=True):
    """
    classic_PCA function

    Description

    This function performs the classic Principal Component Analysis on a given dataset.
    """
    if standarize:
        Data = standarize_data(Input_Data)
    else:
        Data = np.copy(Input_Data)
    eigenvectors_cols, eigenvalues, eigenvectors_rows = np.linalg.svd(
        np.cov(Data)
    )
    idx = eigenvalues.argsort()
    eigenvalues = eigenvalues[idx[::-1]]
    eigenvectors_cols = eigenvectors_cols[:, idx[::-1]]
    eigenvectors_rows = eigenvectors_rows[idx[::-1], :]
    # Return: V matrix, eigenvalues and the principal components.
    return eigenvectors_rows, eigenvalues, np.dot(eigenvectors_rows, Data)


# Post-processing tools:
def mag_to_flux(m, merr):
    """
    Convert magnitude to relative fluxes.
    """
    fluxes = np.zeros(len(m))
    fluxes_err = np.zeros(len(m))
    for i in range(len(m)):
        dist = 10 ** (-np.random.normal(m[i], merr[i], 1000) / 2.51)
        fluxes[i] = np.mean(dist)
        fluxes_err[i] = np.sqrt(np.var(dist))
    return fluxes, fluxes_err

def batman_model(x0,time):
    """Generate the Mandel & Agol (batman) transit model.
    Inputs:
    x0 - transit parameters:  P (orbital period [days] of planet), t0 (time of mid transit), Rp/Rs, a/Rs, inclination, u1, u2. This assumes quadratic limb darkening.
    period - orbital period of planet (in days)
    time - time array over which to calculate model

    Returns:
    modelled fluxes as a function of time array
    """
    parms = batman.TransitParams()
    parms.t0 = x0[1]                       #time of inferior conjunction
    parms.per = x0[0]                       #orbital period
    parms.rp = x0[2]                       #planet radius (in units of stellar radii)
    parms.a = x0[3]                        #semi-major axis (in units of stellar radii)
    parms.inc = x0[4]                      #orbital inclination (in degrees)
    parms.ecc = 0.                     #eccentricity
    parms.w = 90.                        #longitude of periastron (in degrees)
    parms.limb_dark = "quadratic"        #limb darkening model
    parms.u = [x0[5],x0[6]]      #limb darkening coefficients [u1, u2]
    m = batman.TransitModel(parms, time)
    return parms, m
    
# Define transit-related functions:
def reverse_ld_coeffs(ld_law, q1, q2):
    if ld_law == "quadratic":
        coeff1 = 2.0 * np.sqrt(q1) * q2
        coeff2 = np.sqrt(q1) * (1.0 - 2.0 * q2)
    elif ld_law == "squareroot":
        coeff1 = np.sqrt(q1) * (1.0 - 2.0 * q2)
        coeff2 = 2.0 * np.sqrt(q1) * q2
    elif ld_law == "logarithmic":
        coeff1 = 1.0 - np.sqrt(q1) * q2
        coeff2 = 1.0 - np.sqrt(q1)
    elif ld_law == "linear":
        return q1, q2
    return coeff1, coeff2

### other supporting code. Outside of class, because want to call these functions without having to make it's own instance
def LoadResultsTxt(file, deliminator='\t', noSpaces_tabs=False): 
    #noSpaces=assuming that the string names don't have any spaces or tabs and if there are some, it's extra spacing. This is ONLY for the title array (Info0)
    Names, final_Results = [], []
    txt = open(file, 'r')
    SameLength, cnt = True, 0 #to keep track of if each results array element has the same length, and thus can be converted to 
    for Line in txt: #to read first line of file, which contains the string titles of data
        if Line.startswith('#'):
            continue
        Info = Line.split(deliminator)
        Inf0 = Info[0]
        if noSpaces_tabs:
            Info_noSpace = Info[0].split(' ')[0] #to get rid of any lingering spaces
            Inf0 = Info_noSpace.split('\t')[0]  #to get rid of any lingering tabs
        Names.append(Inf0)
        results = []
        for I in range(1,len(Info)): 
            try:
                result = float(Info[I])  #converting to float automatically gets rid of extra spaces and tabs
            except: #if can't convert that specific index to a float, then assume it's a deliminator in the way or a string in replace of the float (e.g. 'FIXED PARAMETER')
                continue #Skip this index in that case
            results.append(result)
        if cnt == 0:
            Len = len(results)
        elif Len == len(results):
            pass
        else:
            SameLength = False
        final_Results.append(np.array(results))
        cnt +=1
    if SameLength:#only convert to array if the lists all have the same length
        return Names, np.array(final_Results)
    else:
        return Names, final_Results


def batman_transit_model(x0,time, q1q2_param=False): 
    """Generate the Mandel & Agol (batman) transit model.
    Inputs:
    x0 - transit parameters:  P (orbital period [days] of planet), t0 (time of mid transit), Rp/Rs, a/Rs, inclination, u1, u2. This assumes quadratic limb darkening.
    period - orbital period of planet (in days)
    time - time array over which to calculate model

    Returns:
    modelled fluxes as a function of time array
    """
    parms = batman.TransitParams()
    parms.t0 = x0[1]                       #time of inferior conjunction
    parms.per = x0[0]                       #orbital period
    parms.rp = x0[2]                       #planet radius (in units of stellar radii)
    parms.a = x0[3]                        #semi-major axis (in units of stellar radii)
    parms.inc = x0[4]                      #orbital inclination (in degrees)
    parms.ecc = 0.                     #eccentricity
    parms.w = 90.                        #longitude of periastron (in degrees)
    parms.limb_dark = "quadratic"        #limb darkening model
    if q1q2_param:
        u1, u2 = reverse_ld_coeffs("quadratic", x0[5],x0[6])
    else:
        u1, u2 = x0[5],x0[6]
    parms.u = [u1, u2]      #limb darkening coefficients [u1, u2]
    m = batman.TransitModel(parms, time)
    return m.light_curve(parms) #flux of model

def SimgaClipping(): #sigma clipping, to get rid of the outliers. If 'Nsig'-simga further than the surrounds 'Nneighbors' points ('Nneighbors'/2 bck, 'Nneighbors'/2 front)
    return 
    
def systematics_model(p0, sys_mod_params, sys_mod_poly_orders, array_len, normalise_inputs=True):
    """ Generate a systematics model which is fed any combination of external parameters (i.e. airmass, fwhm, sky-background, etc).

    Input:
    p0 -- the offset and coefficients of the model. The added offset must *always* be set at index of 0, i.e. p0[0].
    given_parameters = list of [Eparams,PolyOrders]. If None, then taking the global defined Eparams and PolyOrders. MUST BE a list
    # Eparams -- list of arrays, where each element in the list consits of the array values of the external parameter used
    # PolyOrders -- array of polynomial orders. This MUST be in the order of the Eparams list
    # array_len -- int, to keep count of how many frames there are in the data

    example of PolyOrders use:
    a cubic airmass polynomial, quadratic fwhm and xpos; time, ypos and sky not used: PolyOrders = np.array([0,3,2,2,0,0])

    Returns: the evaluated, combined systematics model as a numpy array."""

    # Ancillary data and PolyOrders are ALWAYS in the order:

    offset = p0[0] # offset added to model, which is at the start of p0
    systematics_model = 1 # initiate systematics offset as being unity so that susbequent models can be multiplied together
    current_index = 0 #to keep track of the already used coefficients
    if len(sys_mod_params) == 0: #just to pull out the offset for the case when running without external parameters
        return np.zeros(array_len)+offset

    for Ep in range(len(sys_mod_params)):
        # get coefficients from p0, have to add zero at the end so that we don't add another offset, as the offset for np.poly1d is at p0[-1]
        poly_coefficients_i = np.hstack((p0[1+current_index:1+current_index+sys_mod_poly_orders[Ep]],[0]))
        poly_i = np.poly1d(poly_coefficients_i) # construct polynomial

        # evaluate polynomial
        if normalise_inputs:
            poly_eval_i = poly_i((sys_mod_params[Ep]-sys_mod_params[Ep].mean())/sys_mod_params[Ep].std())
            # poly_eval_i = poly_i(-1.0 + 2.0*(sys_mod_params[Ep] - np.min(sys_mod_params[Ep]))/(np.max(sys_mod_params[Ep])-np.min(sys_mod_params[Ep])))
        else:
            poly_eval_i = poly_i(sys_mod_params[Ep])
        systematics_model *= poly_eval_i # multiply with current systematics model
        current_index += sys_mod_poly_orders[Ep] # keep running count of indices in p0
    return systematics_model + offset

def quantile(x, q, weights=None): #Compute sample quantiles with support for weighted samples. --- ripped from coner.py
    """
    Parameters
    ----------
    x : array_like[nsamples,] --- The samples.
    q : array_like[nquantiles,] --- The list of quantiles to compute. These should all be in the range ``[0, 1]``.
    weights : Optional[array_like[nsamples,]] ---  An optional weight corresponding to each sample. 
    NOTE: When ``weights`` is ``None``, this method simply calls numpy's percentile function with the values of ``q`` multiplied by 100.

    Returns
    -------
    quantiles : array_like[nquantiles,] --- The sample quantiles computed at ``q``.

    Raises
    ------
    ValueError: For invalid quantiles; ``q`` not in ``[0, 1]`` or dimension mismatch between ``x`` and ``weights``.
    """
    x = np.atleast_1d(x)
    q = np.atleast_1d(q)

    if np.any(q < 0.0) or np.any(q > 1.0):
        raise ValueError("Quantiles must be between 0 and 1")

    if weights is None:
        return np.percentile(x, list(100.0 * q))
    else:
        weights = np.atleast_1d(weights)
        if len(x) != len(weights):
            raise ValueError("Dimension mismatch: len(weights) != len(x)")
        idx = np.argsort(x)
        sw = weights[idx]
        cdf = np.cumsum(sw)[:-1]
        cdf /= cdf[-1]
        cdf = np.append(0, cdf)
        return np.interp(q, cdf, x[idx]).tolist()

def PlotLC(LC, poly_coeff, used_epars, used_Porders, Tduration, transit_coeff, FULLpath, plot_name='ModelFit', poly_model=None, save_dats=False): # to generate and save plots for a given fit
    #MAKE SURE TRANSIT TIME AND T0 ARE IN FULL JD!!!!
    #SaveAsNPY = if you want to save the data used for plotting. Can be either boolan, or a string consisting of the path name where the data should be saved. DON'T FORGET TO INCLUDE THE '/' at the end of the last path!!
    #To deliminate OOT and in transit times. Used only for plotting purposes
    #LC = 3xn array where the 3 different dimensions are times, relative flux, and flux error
    #transit_coeff = the batman transit lc parameters in order ['P', 't0', 'Rp/Rs', 'a/Rs', 'inc', 'u1', 'u2']
    Buff_days = 2.0/(24.0*60) #2 minutes to days
    Duration_days = Tduration/24.0 #hrs to days
    ingres, egres = transit_coeff[1]-Duration_days/2.0, transit_coeff[1]+Duration_days/2.0
    OOT1, OOT2 = np.where((LC[0]<=ingres-Buff_days))[0], np.where((LC[0]>=egres+Buff_days))[0]
    OOT = np.append(OOT1,OOT2)        
    normalization = np.mean(LC[1][OOT]) #to make sure oot flux is close to 1

    #To get the models to be plotted
    if type(poly_model) != np.ndarray: #then assuming the poly_model was not given and need to create it out of coefficients
        poly_model = systematics_model(poly_coeff, used_epars, used_Porders, len(LC[0]))
    transit_model = batman_transit_model(transit_coeff, LC[0])
    fitted_model = poly_model*transit_model
    JD_int = int(str(LC[0][0]).split('.')[0])
    FitStats = '$\sigma$ = %d [ppm]'%(np.std(LC[1]-fitted_model)*1e6)

    #Now to actually do the plotting
    fig = plt.figure(figsize=(20,10))
    wl_err = np.zeros(len(LC[0]))
    plt.title(plot_name+'_LCs')
    plt.subplot2grid((4, 1), (0, 0), rowspan=3, colspan=1)
    plt.errorbar(LC[0]-JD_int,LC[1]/normalization, [LC[2],LC[2]], [wl_err,wl_err], 'b.')
    plt.plot(LC[0]-JD_int,poly_model/normalization,'m.-', label = 'best polynomial model')
    plt.plot(LC[0]-JD_int,transit_model,'k-', label = 'best transit model\nRpRs='+str(np.round(transit_coeff[2], 5)))
    plt.plot(LC[0]-JD_int,fitted_model/poly_model,'g--', label = 'combined')
    plt.errorbar(LC[0]-JD_int,LC[1]/poly_model, [LC[2],LC[2]], [wl_err,wl_err], 'cd', label = 'corrected')
    plt.ylabel('Normalised flux')
    plt.legend()

    plt.subplot2grid((4, 1), (3, 0), rowspan=1, colspan=1)
    plt.errorbar(LC[0]-JD_int,LC[1]-fitted_model,[LC[2],LC[2]], [wl_err,wl_err],'r.', label = FitStats)
    plt.plot(LC[0]-JD_int,np.zeros(len(LC[0])),'k-')
    plt.ylabel('Residuals')
    plt.xlabel('JD date - '+str(JD_int)+' (days)')
    plt.legend()
    if not os.path.isdir(FULLpath): #if the folder doesn't exist, 
        os.makedirs(FULLpath, exist_ok=True) # create it
    plt.savefig('%s/%s.png'%(FULLpath,plot_name))
    print ("Saved '"+plot_name+".png' in '"+FULLpath+"'")
    plt.close()
    if save_dats: #To save all of the data in a .npy file for later use
        Additional = '' 
        if type(save_dats) == str: #if file type is a string, then it's the patH used to save the data
            folders = save_dats.split("/")
            patH = ''
            for f in range(len(folders)-1): #the last string in folders is additional naming for the file (could be an empty string)
                patH += folders[f]+'/' 
            Additional = folders[-1]
        else: #otherwise, save the .npy file where saving the figures
            patH = FULLpath
        if not os.path.isdir(patH): #if the folder doesn't exist, 
            os.makedirs(patH, exist_ok=True) # create it
        final_results = np.array([LC[0], poly_model/normalization, transit_model, LC[1]/normalization, LC[1]/poly_model]) #[time, normalized_polynomial_model, transit_model, original_CMCcorrected_normalized_data, final_corrected_LC]
        np.save(patH+Additional+plot_name, final_results)
        print ("Saved '"+Additional+plot_name+".npy' in '"+patH+"'")
    return None

def getVariants(Epars_names, Max_P_order): #to get all the different combinations of External parameters and polynomial fits for each variant
    all_possible_combinations = itertools.product([True,False],repeat=len(Epars_names))
    poly_orders = np.arange(1, Max_P_order+1) #+1 because arange last index is not inclusive

    Variants= []
    for combination in all_possible_combinations: 
        #list of all the external parameter names and array values used in this iteration
        epar_names_i = []
        for c in range(len(combination)):# switch on/off the detrend inputs
            if combination[c]: #because using the same epars list from the begining, always going to have the same order of epar names
                epar_names_i.append(Epars_names[c]) #For example, if epars is ['time', 'airmass', etc.] epars[0] will always be TIME, epars[1] will always be AIRMASS, etc. and we will run through all possible t/f options
        all_possible_polys = itertools.product(list(poly_orders),repeat=len(epar_names_i))
        for app in all_possible_polys: #to have each used external parameter fit a polynomial of each allowed order
            ply_orders_i = np.array(list(app))
            variant = ''
            if ply_orders_i.shape[0] == 0: #if there are no external parameters
                variant = 'Offset0' #then test the case when there is just an offset term
            for en in range(len(epar_names_i)):
                variant += epar_names_i[en]+'-'+str(ply_orders_i[en])
                if en != len(epar_names_i)-1:
                    variant += '_'
            Variants.append(variant)
    return Variants 

def RemoveLine(file, Del_lins): #to read in a .txt/.dat file, and remove specificed line. Have to do this by copying the whole file and rewriting all line excpet for the undesired ones
    #Del_lins  == list of lines that you want to delete. Index starts with ONE (not 0)!!! Doing this because sublime starts with 1
    KeepLines, lin_cnt = '', 1 #Index starts with ONE (not 0)!!!
    txt = open(file, 'r') #to make a list of all the lines that are wanted to be kept
    for Line in txt: #to read
        if lin_cnt in Del_lins: #if don't want that line, don't add it to your keep list
            pass
        else:
            KeepLines+=Line
        lin_cnt += 1 
    txt = open(file, 'w')
    txt.write(KeepLines)
    return None