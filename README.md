# TransSpecDetrend
Codes used to detrend ground-based transmission spectral data.

This package goes from the 2D spectral data extracted from a pipeline specific for the telescope and instrument to a final reduced transmission spectrum. It takes the 2D spectral data, which should be binned down by integrating the spectra in desired wavelength ranges and creates transit light curves from this. Then the systematics in each binned light curve are reduced using a combination of common-mode correction (CMC) and polynomial regression. This is done by first removing non-chromatic systematics with the CMC technique. Additional systematics unique to each spectroscopic bin is then reduced using polynomial regression, where each provided auxiliary parameter (e.g. airmass, FWHM, pixel trace) is used as regressors. Once each binned light curve has been corrected for systematics, the transit depth of each bin is compared against the associated wavelength of that bin. This is the transmission spectrum, which can also be plotted here.

A thorough description of the routine and reasoning behind its development can be seen in the paper that originally implemented it: McGruder et. el. 2022 (ADS link: https://ui.adsabs.harvard.edu/abs/2022AJ....164..134M/abstract).

This routine was used for reducing
1.	Magellan/IMACS data of HATS-29b and VLT/FORS2 data of WASP-124b in McGruder et. el. in prep;
2.	Magellan/IMACS data of WASP-25b and WASP-124b and NTT/FOSC2 data of WASP-25b in McGruder et. el. 2023 (ADS link: [https://ui.adsabs.harvard.edu/abs/2023AJ....166..120M/abstract](url));
3.	Magellan/IMACS and VLT/FORS2 data of WASP-96b in McGruder et. el. 2022 (ADS link: [https://ui.adsabs.harvard.edu/abs/2022AJ....164..134M/abstract](url))

Required packages (excluding anaconda standard packages):
1) corner - [https://github.com/dfm/corner.py](url)
2) batman - [https://lkreidberg.github.io/batman/docs/html/index.html](url)
3) pymultinest - [https://johannesbuchner.github.io/PyMultiNest/](url)

Scripts:
1) Run_BMA_Parameteric_mthread.py - This is the main script where one inputs the data and general settings. Additionally, this script does the common-mode correction (CMC) per each bin. The 'PolyMulitNestClass' class is then called here to do the additional chromatic detrending of the CM corrected data. The originally inputted data needs to be passed as a pickle file, where the file is of the following format:
```
t           An array of the times of observation in BJD (UTC).
wbins       An array of the wavelength ranges used for each bin in 'oLCw' and 'cLCw'.
oLCw        An array of light curves of the target star at different wavelengths. It has dimensions of (time x wavelength bin).
cNAmes      A dictionary of the names of each comparison star used for 'cLC' and 'cLCw' and which array element corresponds to which comparison star name.
cLCw        An array of light curves of the comparison stars at different wavelengths. It has dimensions of (time x comparison number x wavelength bin). The associated name to each of the comparison numbers is defined on the `cNAmes` key.
oLC         An array of the "white-light" light curve of the target star (i.e., the integrated flux in all the wavelength bins defined above).
cLC         Same for the comparison stars. It has dimensions of (time x comparison number). Where the associated name to each of the comparison numbers is defined on the `cNAmes` key.
```
2) PolyMulitNestClass.py - This is the main class of the package. It detrends the chromatic systematics using auxiliary parameters (i.e. airmass, fwhm, pixel trace), defined in 'Run_BMA_Parameteric_mthread.py,' as regressors to fit 1st and 2nd order polynomials on the systematics. This is done in 3 stages:
   A) First, it fits for just the polynomial systematic coefficients on the out-of-transit data using the “Powell” method of scipy.optimize.minimize. This is done to initially capture systematic trends that might be skewed by the transit features.
   B) Then, it uses the found coefficients as the initial start points of another scipy.optimize.minimize run, which includes the in-transit data and a transit model fit.
   C) Last, it uses pymultinest as the final posterior exploration, where the transit parameters found with scipy were used as priors on the mean value, while maintaining the prior bounds. The found polynomial coefficients are used as the mean values for normal prior distributions with a standard deviation of 0.05.

The class individually explores the posterior space of all possible combinations of systematic corrections with the auxiliary parameters. In other words, steps A - C are done when using just airmass as a regressor, airmass and fwhm, pixel trace and fwhm, etc. This means that, for each bin, 729 models will be fit if 6 auxiliary parameters are given and 243 models if 5 are given. The code then uses Bayesian Model Averaging (BMA) to combine the posteriors from all explored posterior spaces.

4) MakeTransSpec.py - Reads in the BMA posteriors for each wavelength bin. From that it determines the mean and 1 sigma uncertainties of the transit depth for each bin. This information is then put in a single .dat file, i.e. the transmission spectrum. The only input is the path to the original .pkl files and the path to the detrended data. 
5) FinalResults.py - Takes the transmission spectral data of each observation, produced by running 'MakeTransSpec.py' for each dataset, and makes a combined transmission spectrum. The combined transmission spectrum is also plotted in this script. Note: a different dataset is transmission spectral data of the same target from a different observing night.
6) utils.py - A module containing supporting functions that will be used by all other scripts.
