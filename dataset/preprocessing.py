from __future__ import annotations
from functools import partial
import numpy as np
import pandas as pd
from pywt import wavedec, waverec, swt, threshold
from scipy.signal import kaiserord, filtfilt, firwin, freqz, windows, cheby2, iirnotch, welch
from tqdm.auto import tqdm
from typing import Union, Optional, List, Any, Dict

class Preprocessing():
    """
    Class for preprocessing the data.

    Attributes
    ----------
    params : Dict[str, Dict[str, Any]]
        Dictionary defining which preprocessing steps to apply to the data.

    """
    def __init__(
            self,
            #params: Dict[str, Dict[str, ]] = None
            ):
        self.chan_names = ["PO8", "O2", "O1", "PO7", "PO3", "POZ", "PO4", "Pz"]
        self.fs = 250.0
    
    def __call__(
            self, 
            data: List[Union[pd.DataFrame, np.ndarray]],
            params: Dict[str, Dict[str, Any]] = None
            ) -> List[Union[pd.DataFrame, np.ndarray]]:
        """
        Allows to pass a nested dictionary that defines which preprocessing steps to apply to the data.
        Applies preprocessing steps in the order they are defined in the dictionary.

        Parameters
        ----------
        data : List[Union[pd.DataFrame, np.ndarray]]
            List of dataframes or arrays to preprocess.
        params : Dict[str, Dict[str, Any]], optional
            Dictionary defining which preprocessing steps to apply to the data, by default None

        Returns:
            Preprocessed data : List[Union[pd.DataFrame, np.ndarray]]
        """
        for func_name, func_params in params.items():
            # Check if method exists
            if hasattr(self, func_name):
                # Get method
                func = getattr(self, func_name)
                # Call method with parameters
                data = func(data, **func_params)
            else:
                raise ValueError(f'Unknown function: {func_name}')
        return data
    
    def normalization(
            self, 
            data: List[Union[pd.DataFrame, np.ndarray]],
            per_channel: bool = True,
            clamp: float = 20.0
            ) -> List[Union[pd.DataFrame, np.ndarray]]:
        """
        Normalizes the data by subtracting the mean and dividing by the standard deviation.

        Parameters
        ----------
        data : List[Union[pd.DataFrame, np.ndarray]]
            List of dataframes or arrays to normalize.
        per_channel : bool, optional
            Whether to normalize per channel or not, by default True
        clamp : float, optional (choose None for no clamp)
            Whether to clamp the data to a range to reduce impact of outliers, 
            by default clamps between -20:20 standard deviations

        Returns:
            Normalized data : List[Union[pd.DataFrame, np.ndarray]]
        """
        if per_channel:
            get_mean = partial(np.mean, axis=0, keepdims=True)
            get_std = partial(np.std, axis=0, keepdims=True)
        else:
            get_mean = np.mean
            get_std = np.std

        for i in tqdm(range(len(data)), total=len(data), desc="Normalization"):
            if not isinstance(data[i], (pd.DataFrame, np.ndarray)):
                raise TypeError('Data must be of type pd.DataFrame or np.ndarray')
            
            if isinstance(data[i], pd.DataFrame): #for dataframes
                channel_data = data[i][self.chan_names].values
                mean = get_mean(channel_data)
                std = get_std(channel_data)
                std[std == 0] = 1  # Avoid division by zero

                # Normalize data in-place
                np.subtract(channel_data, mean, out=channel_data)
                np.divide(channel_data, std, out=channel_data)

                if clamp:
                    np.clip(channel_data, a_min=-clamp, a_max=clamp, out=channel_data)
                # Replace original channel columns with normalized data
                data[i][self.chan_names] = channel_data

            else: #for arrays
                mean = get_mean(data[i])
                std = get_std(data[i])
                std[std == 0] = 1  # Avoid division by zero

                # Normalize data in-place
                np.subtract(data[i], mean, out=data[i])
                np.divide(data[i], std, out=data[i])
                if clamp:
                    np.clip(data[i], a_min=-clamp, a_max=clamp, out=data[i])
        return data
    
    def notch_filter(
            self, 
            data: List[Union[pd.DataFrame, np.ndarray]], 
            freqs: List[float]=[60.0]
            ) -> List[Union[pd.DataFrame, np.ndarray]]:
        """
        Applies a notch filter to the data at 60Hz to filter out line noise. 

        Parameters
        ----------
        freqs : Union[float, List[float]]
            Frequency to filter out, defaults to 50.0
            Can be a list of frequencies to filter out multiple frequencies or a single frequency, e.g. 50.0 or [50.0, 60.0]

        Returns:
            Normalized data : List[Union[pd.DataFrame, np.ndarray]]
        """
        if isinstance(freqs, float):
            freqs = [freqs]
        assert len(freqs) > 0, "frequency must be a list of floats"

        for i in tqdm(range(len(data)), total=len(data), desc="Notch filtering"):
            if not isinstance(data[i], (pd.DataFrame, np.ndarray)):
                raise TypeError('Data must be of type pd.DataFrame or np.ndarray')

            for f in freqs:
                b, a = iirnotch(f, 30.0, fs=self.fs)
                if isinstance(data[i], pd.DataFrame):
                    channel_data = data[i][self.chan_names]
                    channel_data = filtfilt(b, a, channel_data, axis = 0) 
                    data[i][self.chan_names] = channel_data
                else:
                    data[i] = filtfilt(b, a, data[i], axis = 0)
        return data
    
    def bandpass_filter(
            self,
            data: List[Union[pd.DataFrame, np.ndarray]], 
            low: Optional[float] = 1.0, 
            high: Optional[float] = 95.0,
            width_low: Optional[float] = 2.0, 
            width_high: Optional[float] = 6.0,
            ripple_low: Optional[float] = 72.0, 
            ripple_high: Optional[float] = 20.0
            ) -> List[Union[pd.DataFrame, np.ndarray]]:
        """
        Applies a band pass filter to the data
        
        Parameters
        ----------
        data : List[Union[pd.DataFrame, np.ndarray]]
            List of dataframes or arrays to filter.
        low : float, optional
            Low cutoff frequency, by default 1.0
        high : float, optional
            High cutoff frequency, by default 95.0
        width_low : float, optional
            Width of transition from pass at low cutoff (rel. to Nyq), by default 2.0
        width_high : float, optional
            Width of transition from pass at high cutoff (rel. to Nyq), by default 6.0
        ripple_low : float, optional
            Desired attenuation in the stop band at low cutoff (dB), by default 72.0
        ripple_high : float, optional
            Desired attenuation in the stop band at high cutoff (dB), by default 20.0
        
        Returns:
            Preprocessor: Preprocessor
        """
        nyq = self.fs/2.0

        # Compute order and Kaiser parameter for the FIR filter.
        N_high, beta_high = kaiserord(ripple_high, width_high/nyq)
        N_low, beta_low = kaiserord(ripple_low, width_low/nyq)

        # Compute coefficients for FIR filters
        taps_high = firwin(N_high, high/nyq, window=("kaiser", beta_high))
        taps_low = firwin(N_low, low/nyq, window=("kaiser", beta_low), pass_zero=False)

        for i in tqdm(range(len(data)), total=len(data), desc="Bandpass filtering"):
            if not isinstance(data[i], (pd.DataFrame, np.ndarray)):
                raise TypeError('Data must be of type pd.DataFrame or np.ndarray')
            
            if isinstance(data[i], pd.DataFrame): #for dataframes
                channel_data = data[i][self.chan_names]
                channel_data = filtfilt(taps_low, 1.0, channel_data, axis = 0) #highpass
                channel_data  = filtfilt(taps_high, 1.0, channel_data, axis = 0) #lowpass   
                data[i][self.chan_names] = channel_data
            else: #for arrays
                data[i] = filtfilt(taps_low, 1.0, data[i], axis = 0) #highpass
                data[i]  = filtfilt(taps_high, 1.0, data[i], axis = 0) #lowpass
        return data
    

    def dwt_denoising(
            self, 
            data: List[Union[pd.DataFrame, np.ndarray]],
            type="dwt", 
            wavelet="db8", 
            mode="sym",  
            level: int = 4,
            method: Union[str, float] = "soft",
        ) -> List[Union[pd.DataFrame, np.ndarray]]:
        """
        Applies a discrete wavelet decomposition or stationary wavelet decomposition of the data into 
        approximate and detailed coefficients containing low- and high-frequency components, respectively. 

        The obtained coefficients are used for denoising by thresholding the
        coefficients and inverting the filtered space back to the time domain. 

        Parameters
        ----------
        data : List[Union[pd.DataFrame, np.ndarray]]]
            Data to be denoised
        type : string
            Choose between discrete wavelet transformation ("dwt") and static wavelet transformation ("swt").
            SWT is a translation-invariance modification of the discrete wavelet transform (DWT).
            Default: "dwt"
        wavelet : string
            Wavelet type to use in decomposition 
            An exhaustive list of the wavelet families can be obtained by calling pywt.families()
            Default: "db8" - Daubechies-8 wavelet is used as suggested in Asaduzzaman etl al. (2010)
        mode : string
            Signal extension mode. Extrapolates the input data to extend the signal before computing the dwt.
            Depending on the extrapolation method, significant artifacts at the signal's borders can be introduced during that process.
            Default: "sym" - symmetric window 
            Check: https://pywavelets.readthedocs.io/en/latest/ref/signal-extension-modes.html#ref-modes for further explanation
        level : int
            Number of decompositions to be done (yields 1 approximate, n-1 detailed coefficients)
            If None, four-level decomposition is done as suggested in Aliyu & Lim (2023)
            Default: 4
        method : string or int
            Thresholding method to be used for denoising. If string, one of the following methods can be used:
                - "soft" - soft thresholding
                - "hard" - hard thresholding
                - "garrote" - garrote thresholding
                - "greater" - greater thresholding
                - "less" - less thresholding
            If int, Neigh-Block Denoising is used. The integer value is applied as sigma.
            Default: "soft" - soft thresholding

        Returns:
            Preprocessor: Preprocessor
                Adds the following columns to the metadata:
                    - approx_coeffs (array): approximate coefficients 
                    - detail_coeffs (list of arrays): detailled coefficients
                    Note: Contains level-1 arrays of detailled coefficients
        """
        types = ["dwt", "swt"]
        if type not in types:
            raise ValueError("Invalid type. Expected one of: %s" % types)
        assert method in ["soft", "hard", "garrote", "greater", "less"] or isinstance(method, float), "Invalid method. Expected one of: [soft, hard, garrote, greater, less] or of type float"
        for i in tqdm(range(len(data)), total=len(data), desc="DWT denoising"):
            if isinstance(data[i], pd.DataFrame): #if dataframe
                channel_data = data[i][self.chan_names].values
                denoised_data = self._denoise(channel_data, type, wavelet, mode, level, method)
                data[i][self.chan_names] = denoised_data
            elif isinstance(data[i], np.ndarray): #if ndarray
                denoised_data = self._denoise(data[i], type, wavelet, mode, level, method)
                data[i] = denoised_data
            else:
                raise TypeError('Data type not understood. Please provide a DataFrame or ndarray.')
        return data

    def _denoise(self, channel_data, type, wavelet, mode, level, method):
        """
        Helper function for dwt_denoising. Applies the actual denoising.
        """
        if type == "dwt":
            transform = partial(wavedec, wavelet = wavelet, level = level, axis = 0)
        else:
            transform = partial(swt, wavelet = wavelet, level = level, axis = 0, trim_approx = True)
        coeffs = transform(channel_data)

        # Simple Threshold Denoising
        if isinstance(method, str):
            thresholding = partial(threshold, mode=method, substitute=0)
            for j in range(len(coeffs[1:])):
                sig = np.median(np.abs(coeffs[j+1]), axis=0)/0.6745
                thresh = sig * np.sqrt(2*np.log(len(coeffs[j+1])))
                if np.all(thresh) == True:
                    thresholded_detail = thresholding(data=coeffs[j+1], value=thresh)
                    coeffs[j+1] = thresholded_detail
                    #res.append(thresholded_detail)
            channel_data = waverec(coeffs, wavelet = wavelet, axis=0)

        # Neigh-Block Denoising (To do: make more efficient with vectorization)
        else: #Method used as "sigma"
        # helper function to compute beta shrinkage
            def beta(method, L, detail, lmbd):
                S2 = np.sum(detail ** 2)
                beta = (1 - lmbd * L * method**2 / S2)
                return max(0, beta)

            #res = []
            L0 = int(np.log2(len(channel_data)) // 2)
            L1 = max(1, L0 // 2)
            L = L0 + 2 * L1
            lmbd = 4.50524  # explanation in Cai & Silverman (2001)
            for j, detail in enumerate(coeffs[1:]):
                d2 = detail.copy()
                # group wavelet into disjoint blocks of length L0
                for start_b in range(0, len(d2), L0):
                    end_b = min(len(d2), start_b + L0)  # if len(d2) < start_b + L0 -> last block
                    start_B = start_b - L1
                    end_B = start_B + L
                    if start_B < 0:
                        end_B -= start_B
                        start_B = 0
                    elif end_B > len(d2):
                        start_B -= end_B - len(d2)
                        end_B = len(d2)
                    assert end_B - start_B == L  # sanity check
                    d2[start_b:end_b] *= beta(method, L, d2[start_B:end_B], lmbd)
                #res.append(d2)
            channel_data = waverec(coeffs, wavelet = wavelet, axis = 0)
        return channel_data
            