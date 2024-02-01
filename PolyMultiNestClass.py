import matplotlib.pyplot as plt
import numpy as np
import batman
import pickle
import sys
import os
import pymultinest
from scipy.stats import gamma,norm,beta,truncnorm
import corner
from scipy import optimize
import utils

####################################################################################################
#### FUNCTIONS TO TEST ALL POSSIBLE COMBINATIONS OF DETRENDING FUNCTION BASED ON ANCILLARY DATA ####
####################################################################################################

###### Start of class consisting of all the important fitting routines. All functions in class have to be apart of their own instance for a given polynomial external parameter set up
class Parametric_MultiNest: #class for a specific model to be tested
    def __init__(self, LC_data, Eparams_values, Eparams_names, PolyOrders, transit_pars, Batman_init, TransDuration, Polynol_priors=.05, Buffer=2, Quantiles = [0.15865,0.50, 0.84135]): #Polynol_Bounds, 
        self.Eparams_values = Eparams_values #list of arrays, where each element in the list consits of the array values of the external parameter used
        self.Eparams_names = Eparams_names #list of arrays, where each element in the list consits of the string value of the name of the external parameter used
        self.PolyOrders = PolyOrders #array of polynomial orders applied to each axis. This MUST be in the order of Eparams_names and Eparams_values
        self.TransDuration = TransDuration
        self.Polynol_priors = Polynol_priors

        #LC_data = 3xn array where the 3 different dimensions are times, relative flux, and flux error
        self.LC_data = LC_data
        self.data, self.mean_sigma = LC_data[1], np.mean(LC_data[2])
        self.time = LC_data[0] #NOTE! make sure time is in JD starting with 2400000!

        #To combine the fitting parameter names 
        parameter_list = []
        self.trans_param_str = ['P', 't0', 'Rp/Rs', 'a/Rs', 'inc', 'u1', 'u2'] #1st 7 external parameters are these transit parameters, IF it's prior is not 0
        for tp in self.trans_param_str: #to keep track of the used transit parameters
            if transit_pars[tp][1]: #2nd element is the prior function. if None, then kept fixed and not fitting for
                parameter_list.append(tp)
        self.lenTransPars = len(parameter_list)
        print ("Fitting for the following ("+str(self.lenTransPars)+") transit parameters:", parameter_list)
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
        self.results_name = results_name
        self.ndim = len(parameter_list)
        self.lenCoeffs = 1+np.array(PolyOrders).sum() #number of polynomial coefficient. Plus 1, because of the offset term
        self.parameter_list = parameter_list
        # self.Polynol_Bounds = Polynol_Bounds
        self.params, self.m = Batman_init[0], Batman_init[1] #the parameters used to initizle the batman model (should just use the same initiation throughout)
        print (len(Eparams_names), "external parameters:", Eparams_names)
        print ("with the following polynomial orders for each:", PolyOrders)
        print ("yielding a total of",self.ndim, 'dimensions')
        print ("poly_coeffs:", self.parameter_list[self.lenTransPars:])
        self.Quants = Quantiles

        ####################################
        ############ DO Scipy.optimize to get priors ##############
        ####################################
        #To deliminate OOT and in transit times
        Buff_days = Buffer/(24.0*60) #minutes to day
        Duration_days = TransDuration/24.0 #hrs to days
        ingres, egres = transit_pars['t0'][0]-Duration_days/2.0, transit_pars['t0'][0]+Duration_days/2.0
        OOT1, OOT2 = np.where((LC_data[0]<=ingres-Buff_days))[0], np.where((LC_data[0]>=egres+Buff_days))[0]
        OOT = np.append(OOT1,OOT2)

        #set scipy.optimize transit bounds
        trans_bnds = []
        for tP in self.trans_param_str:
            if transit_pars[tP][1]: #then priors were given for this variable, having scipy fit for it
                if type(transit_pars[tP][2]) == list and len(transit_pars[tP][2]) == 1: #assuming the priors are gaussian, and making the scipy bounds 3 times the prior std
                    DnBnd, UpBnd = transit_pars[tP][0]-transit_pars[tP][2][0]*3, transit_pars[tP][0]+transit_pars[tP][2][0]*3
                elif type(transit_pars[tP][2]) == int or type(transit_pars[tP][2]) == float: #assuming the priors are gaussian, and making the scipy bounds 3 times the prior std
                    DnBnd, UpBnd = transit_pars[tP][0]-transit_pars[tP][2]*3, transit_pars[tP][0]+transit_pars[tP][2]*3
                elif len(transit_pars[tP][2]) == 2: #assuming the priors are uniform, and using that range for the scipy bounds
                    DnBnd, UpBnd = transit_pars[tP][2][0], transit_pars[tP][2][1]
                else:
                    sys.exit("ERROR!! didn't get reasonable bounds")
                trans_bnds.append((DnBnd, UpBnd))
            else: #if no prior given for this param, set the bounds to 0s
                trans_bnds.append((transit_pars[tP][0], transit_pars[tP][0]))

        #Fitting for just the systematic parameters, to get a good estimate for the systematic fit
        print ("trans_bnds:", trans_bnds)
        Poly_coeff_bnds=(-5,5)
        ARGs = (LC_data[1], LC_data[2], LC_data[0], Eparams_values, PolyOrders, OOT)
        PolyCoeffBnds = [Poly_coeff_bnds]*(self.lenCoeffs)
        m, bounds_sys = 'Powell', tuple(PolyCoeffBnds) 
        x0_sys = np.array([0]*self.lenCoeffs) #the initial polynomial coefficients. offset, then 1 coefficient for each order (i.e. order 3 has 3 coefficients [excluding constant], etc.)
        systematics_fit = optimize.minimize(self.chisq_for_optimize,x0_sys,args=ARGs,method=m,bounds=bounds_sys) #optimized systematic parameters

        #Now fit for both the transit parameters and the polynomial coefficients
        TransPars = [transit_pars['P'][0], transit_pars['t0'][0], transit_pars['Rp/Rs'][0], transit_pars['a/Rs'][0], transit_pars['inc'][0], transit_pars['u1'][0], transit_pars['u2'][0]]
        x0, bounds = np.hstack((TransPars,systematics_fit.x)), tuple(trans_bnds+PolyCoeffBnds)
        print ("\n x0:", x0)
        alldat = np.arange(len(LC_data[0])) 
        ARGs = (LC_data[1], LC_data[2], LC_data[0], Eparams_values, PolyOrders, alldat) #here if transit is on, actually fit for it. Thus, use in transit data too
        fit = optimize.minimize(self.chisq_for_optimize,x0,args=ARGs,method=m,bounds=bounds)
        print ("\n fit pars:", fit.x)

        Tp_cnt = 0
        for Tp in self.trans_param_str: #corresponds to the scipy fit values of fit.x[:7]
            if not transit_pars[Tp][1]:
                pass #don't do anything if it's a fixed value
            else: #need to change the mean of the prior, based on what scipy found. NOTE: keep the initial priors though
                transit_pars[Tp][0] = fit.x[Tp_cnt]
            Tp_cnt +=1
        self.transit_pars = transit_pars
        print ("transit_pars:", transit_pars)
        self.transit_vals = [transit_pars['P'][0], transit_pars['t0'][0], transit_pars['Rp/Rs'][0], transit_pars['a/Rs'][0], transit_pars['inc'][0], transit_pars['u1'][0], transit_pars['u2'][0]]

        self.Poly_pars = fit.x[7:] #these will be used as the mean value for the polycoeff priors

    #scipy.optimize functions

    def chisq_for_optimize(self, p0,flux,error,time,Eparams,poly_orders,OOT):
        """Evaluate the chi2 and generate the transit + systematics model. Needed for scipy.optimize.

        Inputs:
        p0 - [7:] is the offset and polynomial coefficients of the model. The added offset must *always* be set at index of 0, i.e. p0[0]. 
           - [:7] is Mandel & Agol transit parameters: P (orbital period [days] of planet), t0 (time of mid transit), Rp/Rs, a/Rs, inclination, u1, u2. This assumes quadratic limb darkening.
        flux - the array of fluxes 
        error - the array of errors on the fluxes
        time - the time array
        Eparams -- list of arrays, where each element in the list consits of the array values of the external parameter used
        poly_orders -- array of polynomial orders applied to each axis. This MUST be in the order of the Eparams list
            - e.g cubic in time, quadratic in airmass, no other polynomial used: np.array([3,2]) and Eparams would be [np.array(time), np.array(airmass)]
        OOT - array of the out of transit indecies. Needed because though fitting polynomials to all the epar data, can only compare the fit to the oot data.
            - if you want to use the full transit data, just set OOT to the whole array of lc data
        
        Returns:
        chi squared of generated model
        """

        #to determine if also fitting for the transit
        if len(p0) == 8+np.array(poly_orders).sum(): 
            transit = True
        elif len(p0) == 1+np.array(poly_orders).sum():
            transit = False
        else:
            print ("length of p0 is "+str(len(p0))+" but poly_orders is "+str(poly_orders))
            sys.exit("1+poly_orders.sum()="+str(1+poly_orders.sum())+", which is inconsitent with len(p0)!")

        if transit:
            transit_model = utils.batman_transit_model(p0[:7],time)
            sys_model = utils.systematics_model(p0[7:],Eparams,poly_orders, len(flux))
            model = transit_model*sys_model#[OOT] #when doing combined fit, also use in transit data to compare against combined model 
        else:
            model = utils.systematics_model(p0,Eparams,poly_orders, len(flux))[OOT] #can only compare the OOT data of the epars model, cause don't know true transit values
            flux,error,time = flux[OOT],error[OOT],time[OOT]
        resids = (model - flux)/error #can only compare the OOT data of the model and data
        return np.sum(resids*resids)

    def chisquared(self, model,flux,error,npars):
        """Evaluate the chi2 of a pre-generated model.

        Inputs:

        model - the model (transit + systematics)
        flux - array of fluxes
        error - array of errors on flux data points
        npars - number of free parameters in model. Needed to calculate reduced chi squared.

        Returns:

        (chi squared, reduced chi squared)"""

        resids = (model - flux)/error
        chi2 = np.sum(resids*resids)
        reduced_chi2 = chi2/(len(flux) - npars - 1)
        return chi2,reduced_chi2

    # Now define MultiNest priors:
    def prior(self,cube, ndim, nparams):
        cube_cnt = 0 #to keep track of which variables are actually allowed to change
        if self.transit_pars['P'][1]: # Prior on Period:
            cube[cube_cnt] = utils.ChoosePrior(cube[cube_cnt], self.transit_pars['P'])
            cube_cnt += 1

        if self.transit_pars['t0'][1]: # Prior on t0:
            cube[cube_cnt] = utils.ChoosePrior(cube[cube_cnt], self.transit_pars['t0'])
            cube_cnt += 1

        if self.transit_pars['Rp/Rs'][1]: # Prior on planet-to-star radius ratio:
            cube[cube_cnt] = utils.ChoosePrior(cube[cube_cnt], self.transit_pars['Rp/Rs'])
            cube_cnt += 1
        
        if self.transit_pars['a/Rs'][1]: # Prior on a/Rs:
            cube[cube_cnt] = utils.ChoosePrior(cube[cube_cnt], self.transit_pars['a/Rs'])
            cube_cnt += 1
        
        if self.transit_pars['inc'][1]: # Prior on inclination:
            cube[cube_cnt] = utils.ChoosePrior(cube[cube_cnt], self.transit_pars['inc'])
            cube_cnt += 1
        
        # Prior on first LD parms (u1 and u2):
        if self.transit_pars['u1'][1]:
            cube[cube_cnt] = utils.ChoosePrior(cube[cube_cnt], self.transit_pars['u1'])
            cube_cnt += 1
        if self.transit_pars['u2'][1]:
            cube[cube_cnt] = utils.ChoosePrior(cube[cube_cnt], self.transit_pars['u2'])
            cube_cnt += 1
        # Prior on polynomial coefficients:
        poly_coeffs = self.parameter_list[self.lenTransPars:] #polynomial coefficients after transit params
        for pcnt in range(len(poly_coeffs)):
            cube[cube_cnt+pcnt] = utils.transform_normal(cube[cube_cnt+pcnt], self.Poly_pars[pcnt], self.Polynol_priors) #Poly_pars SHOULD be in same order as written in systematics_model
    
    def loglike(self, cube, ndim, nparams):
        # To determine the log-likelihood. For this, assuming gaussian posterior i.e. use chi^2:
        """
        Inputs:
        cube - [7:] is the offset and polynomial coefficients of the model. The added offset must *always* be set at index of 0, i.e. p0[0]. 
           - [:7] is Mandel & Agol transit parameters: P (orbital period [days] of planet), t0 (time of mid transit), Rp/Rs, a/Rs, inclination, u1, u2. This assumes quadratic limb darkening.
        flux - the array of fluxes NOTE: make sure it's the same length as OOT!
        error - the array of errors on the fluxes NOTE: make sure it's the same length as OOT!
        time - the time array NOTE: make sure it's the same length as OOT!
        Eparams -- list of arrays, where each element in the list consits of the array values of the external parameter used
        PolyOrders -- array of polynomial orders applied to each axis. This MUST be in the order of the Eparams list
            - e.g cubic in time, quadratic in airmass, no other polynomial used: np.array([3,2]) and Eparams would be [np.array(time), np.array(airmass)]
        OOT - array of the out of transit indecies. Needed because though fitting polynomials to all the epar data, can only compare the fit to the oot data.
            - if you want to use the full transit data, just set OOT to the whole array of lc data
        
        Returns:
        chi squared of generated model
        """
        #first 7 parameters are transit parameters
        cube_cnt = 0 # keep track of prior indices in cube
        if self.transit_pars['P'][1]: #if a prior, then fitting for the variable. 
            self.params.per = cube[cube_cnt]
            cube_cnt += 1

        if self.transit_pars['t0'][1]: 
            self.params.t0 = cube[cube_cnt]
            cube_cnt += 1 

        if self.transit_pars['Rp/Rs'][1]:
            self.params.rp = cube[cube_cnt]
            cube_cnt += 1 

        if self.transit_pars['a/Rs'][1]:  
            self.params.a = cube[cube_cnt]
            cube_cnt += 1 

        if self.transit_pars['inc'][1]: 
            self.params.inc = cube[cube_cnt]
            cube_cnt += 1 

        if self.transit_pars['u1'][1]:  
            self.params.u[0] = cube[cube_cnt]
            cube_cnt += 1 

        if self.transit_pars['u2'][1]: 
            self.params.u[1] = cube[cube_cnt]
            cube_cnt += 1 

        # self.params.ecc = 0 #keep fixed to 0 for now
        # self.params.w = 90 #keep this fixed to 90 for now 
       
        lcmodel = self.m.light_curve(self.params)

        # Evaluate model:        
        # because can't just slice cube like a list, manually put each polynomial coefficient
        p0 = []
        poly_coeffs = self.parameter_list[self.lenTransPars:] #polynomial coefficients after transit params
        # print ("poly_coeffs:", poly_coeffs)
        for pcnt in range(len(poly_coeffs)):
            p0.append(cube[cube_cnt+pcnt])


        # print ("np.mean(Eparams_values[0]):", np.mean(self.Eparams_values[0]), "np.mean(Eparams_values[1]):", np.mean(self.Eparams_values[1]),"p0:", p0, "\n") 
        # normalisation
        norm = len(self.data)*(-np.log(2*np.pi) - np.log(self.mean_sigma*2)) #mean_sigma is mean of the error, defined in 'RunMultiNest()'
        sys_model = utils.systematics_model(p0, self.Eparams_values, self.PolyOrders, len(lcmodel))
        model = sys_model*lcmodel #simultaneously fit the systematic and transit models
        residuals = self.data-model #data=flux_input defined in 'RunMultiNest()'. Might have to defined this as a global variable
        chisq = np.sum(residuals**2)/(self.mean_sigma**2) 
        return 0.5*(norm - chisq)

    def Run(self, PATH, Nlive, plotCorner=False, plotFit=False):
        if PATH[-1] != '/':
            PATH += '/'
        sub_folder = PATH+self.results_name   
        if not os.path.exists(sub_folder):
            os.mkdir(sub_folder)
        elif os.path.isfile(sub_folder+'/fit_results.txt'):
            print ("results for '"+self.results_name+"' previously done... skipping\n")
            # if plotCorner and not os.path.isfile(sub_folder+'/corner.png'): #if the results are made, but no corner plot. Make the corner
            #     file = open(sub_folder+'/equal_weighted_posteriors.pkl','rb')
            #     posterior_samples = pickle.load(file)
            #     file.close()
            #     fig = corner.corner(posterior_samples, labels=self.parameter_list, quantiles=self.Quants)
            #     fig.savefig(sub_folder+'/corner.png')
            #     print ("Saved 'corner.png' in '"+sub_folder+"'")
            #     plt.close()
            # if plotFit and not os.path.isfile(sub_folder+'/corner.png'): #if the results are made, but no lc. Make the lc
            #     utils.PlotLC(self.LC_data, poly_params, self.Eparams_values, self.PolyOrders, TransDuration, trans_params, sub_folder)
            return None
        else: #if the directory was created, but the results file wasn't, assuming multinest didn't finish and start over
            pass
        sub_folder += '/'
        pymultinest.run(self.loglike,self.prior,self.ndim,n_live_points=Nlive,outputfiles_basename=sub_folder,resume=False,verbose=True)        
        output = pymultinest.Analyzer(outputfiles_basename=sub_folder, n_params=self.ndim)  # Get output
        posterior_samples = output.get_equal_weighted_posterior()[:,:-1] # Get out parameters: this matrix has (samples,n_params+1)
        pickle.dump(posterior_samples, open(sub_folder+"equal_weighted_posteriors.pkl", "wb")) #think this is already saved

        # Extract parameters:
        a_lnZ = output.get_stats()['global evidence']
        LnZs_tab = open(PATH+'Global_LnZs.txt','a')
        LnZs_tab.write(self.results_name+'\t\t\t\t'+str(a_lnZ)+'\t\t\t\t'+str(len(posterior_samples[:,0]))+'\n')
        LnZs_tab.close()

        results_tab = open(sub_folder+'fit_results.txt','w+')
        results_tab.write('#parameter   best_fit        upper_bnd       lower_bnd\n')
        poly_params, trans_params = [], self.transit_vals  #To keep track of the fitted and fixed parameters IF plotting
        cnt = 0
        for p in range(len(self.parameter_list)): #to keep track of the summary of the posterior results
            results = utils.quantile(posterior_samples[:,p],[self.Quants[1], self.Quants[2], self.Quants[0]])
            p_mean, p_upper, p_lower = np.round(results[0],7), np.round(results[1]-results[0],8), np.round(results[0]-results[2],8) 
            results_tab.write(self.parameter_list[p]+'                 \t'+str(p_mean)+'\t'+str(p_upper)+'\t'+str(p_lower)+'\n')
            if plotFit:
                if cnt < self.lenTransPars: #transit parameters 1st
                    fit_idx = self.trans_param_str.index(self.parameter_list[p])
                    trans_params[fit_idx] = results[0] #replace the orignial transit parameter with the fitted one
                else:
                    poly_params.append(results[0])
                cnt +=1
        results_tab.write('###\n#Global lnZ: '+str(a_lnZ)+"\t ndim: "+str(self.ndim)+"\t sample_number: "+str(len(posterior_samples[:,0]))) #to print all stats needed for the Global_LnZs.txt file incase it's not written in that file for some reason
        results_tab.close()
        if plotCorner:
            fig = corner.corner(posterior_samples, labels=self.parameter_list, quantiles=self.Quants)
            fig.savefig(sub_folder+'corner_'+self.results_name+'.png')
            print ("Saved 'corner.png' in '"+sub_folder+"'")
            plt.close()
        if plotFit:
            utils.PlotLC(self.LC_data, poly_params, self.Eparams_values, self.PolyOrders, self.TransDuration, trans_params, sub_folder, plot_name='ModelFit_'+self.results_name)
        print ("\n")
        return None #len(posterior_samples[:,0]) #to keep track number of samples for each iteration

###### End of class