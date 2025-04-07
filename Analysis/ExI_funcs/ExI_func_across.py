import os
import numpy as np
import sklearn
import scipy
from sklearn.metrics import auc
import pandas as pd
from ..general_funcs import CCEP_func, LL_funcs as LLf, freq_funcs as ff
import statsmodels

SleepStates_val = ['Wake', 'NREM', 'REM']
Fs = 500
t0 = 1

def get_AUC_MAX_Pearson(Int_values, LL_values):
    # Calculate Area Under the Curve (AUC), the maximum of the last three values, and Pearson correlation between intensity and LL values
    AUC = auc(Int_values, LL_values)  # AUC calculation based on intensity and LL
    MAX = np.mean(np.sort(LL_values)[-3:])  # Maximum value from the last 3 sorted LL values
    rho = scipy.stats.pearsonr(Int_values, LL_values)[0]  # Pearson correlation coefficient between intensity and LL values
    return AUC, MAX, rho  # Return the computed values


def get_AUC_surr(rc, con_trial, EEG_resp, mx_true, Int_selc, n_trial=200, n=10, w=0.25):
    """
        Calculate surrogate data for stimulation-intensity curve (AUC, max value, and Pearson correlation).
        This is done by generating surrogate data by taking the LL of the mean signal during non-stimulating time (0.5s range).

        Arguments:
            rc: The response channel number
            con_trial: Data frame containing trial information
            EEG_resp: EEG response data
            mx_true: True maximum values for comparison
            Int_selc: Selected intensity values
            n_trial: Number of trials to use for surrogate calculation (default 200)
            n: Number of repetitions to perform for surrogate data (default 10)
            w: Window size for LL calculation (default 0.25)

        Returns:
            AUC_surr: Array of surrogate AUC values
            max_surr: Array of surrogate max values
            p_surr: Array of surrogate Pearson correlation values
        """

    AUC_surr = np.zeros((n * 3,))  # Initialize array to store surrogate AUC values
    max_surr = np.zeros((n * 3,))  # Initialize array to store surrogate max values
    p_surr = np.zeros((n * 3,))  # Initialize array to store surrogate Pearson correlation values

    # Identify stimulation trials based on surrounding stimuli (previous, current, and next)
    stim_trials = np.unique(
        con_trial.loc[(con_trial.Stim == rc) | (con_trial.Stim == rc - 1) | (con_trial.Stim == rc + 1), 'Num'])

    # Identify all trials for the current response channel (excluding artefact trials)
    trials_all = np.unique(con_trial.loc[(con_trial.Chan == rc) & (con_trial.Artefact < 1), 'Num'])

    # Exclude stimulation trials from the selection of trials for surrogate calculation
    trials_all = np.array([i for i in trials_all if i not in stim_trials])
    trials_all = trials_all.astype('int')

    # Normalize the intensity values to the range [0, 1]
    Int_norm = (Int_selc - np.min(Int_selc)) / (np.max(Int_selc) - np.min(Int_selc))

    # Loop over the number of repetitions for generating surrogate data
    for rep in range(n):
        mx_all = np.zeros((len(Int_selc), 3))  # Initialize an array to store maximum values for each intensity level

        # Loop over each intensity value
        for i, intensity in enumerate(Int_selc):
            # Select a subset of trials randomly from the trials available (excluding stimulation trials)
            num_sel = np.unique(np.random.choice(trials_all, n_trial, replace=False))

            # Get the response signal by averaging over the selected trials, followed by a low-pass filter
            resp = ff.lp_filter(np.nanmean(EEG_resp[rc, num_sel, :], 0), 45, Fs)

            # Calculate the LL values for the filtered response signal
            LL_resp = LLf.get_LL_all(np.expand_dims(resp, [0, 1]), Fs, w)[0][0]

            # Extract the maximum values from different time windows in the LL signal
            mx_all[i, 0] = np.max(LL_resp[int((t0 - 0.5) * Fs):int((t0 - w / 2 - 0.01) * Fs)])  # Pre-stimulation window
            mx_all[i, 1] = np.max(LL_resp[int((w / 2) * Fs):int((t0 - 0.5) * Fs)])  # Baseline window
            mx_all[i, 2] = np.max(LL_resp[int((t0 + 2) * Fs):int((t0 + 25) * Fs)])  # Post-stimulation window

        # Loop over the 3 different time windows (pre-stim, baseline, post-stim)
        for i in range(3):
            # Normalize the maximum values in each window with respect to the minimum and maximum values
            normalized_mx_BL = (mx_all[:, i] - np.min(mx_all[:, i])) / (np.max(mx_true) - np.min(mx_all[:, i]))

            # Calculate the AUC, max value, and Pearson correlation for the surrogate data
            AUC_surr[int((i * n) + rep)], max_surr[int((i * n) + rep)], p_surr[
                int((i * n) + rep)] = get_AUC_MAX_Pearson(Int_norm, normalized_mx_BL)

    return AUC_surr, max_surr, p_surr


def get_pvalue(real_array, surr_array):
    count = np.sum(surr_array <= real_array[:, np.newaxis], axis=1)

    # Calculate the p-values
    p_values = np.array(count) / len(surr_array)

    return p_values


def get_AUC_real(sc, rc, con_trial, EEG_resp, get_surr = True, n=100, w=0.25):
    """
        Calculate AUC, MAX, and Pearson correlation (rho) for true data of a specific connection (sc x rc channels).
        General data across all rtials blinded to sleep state. used to calculate significant connections using surrogate testing

        Arguments:
            sc: The stimulation channel
            rc: The response channel
            con_trial: Data frame containing trial information
            EEG_resp: EEG response data
            get_surr: Flag to determine whether to calculate surrogate data (default True)
            n: Number of repetitions for surrogate data (default 100)
            w: Window size for LL calculation (default 0.25)

        Returns:
            AUC, MAX, rho: True data metrics
            AUC_p, MAX_p, rho_p: p-values for AUC, MAX, and rho (surrogate testing)
        """

    # Filter the trials for the specific stimulation channel (sc), response channel (rc), and non-artifact trials
    dat = con_trial[(con_trial['Stim'] == sc) & (con_trial['Chan'] == rc) & (con_trial['Artefact'] < 1)].reset_index(
        drop=True)

    # Get the unique intensities from the filtered data
    Int_selc = np.unique(dat['Int'])

    # Initialize variables to store the maximum values for each intensity
    n_trials = 200
    mx_all = np.zeros((len(Int_selc), 2))

    # Loop over each intensity level to calculate the response and maximum LL values
    for i, intensity in enumerate(Int_selc):
        # Filter the trials for the current intensity and non-artifact trials
        dati = dat[(dat['Int'] == intensity) & (dat['Artefact'] < 1)].reset_index(drop=True)

        # Compute the EEG response by averaging across selected trials and applying a low-pass filter
        resp = ff.lp_filter(np.nanmean(EEG_resp[rc, dati.Num.values.astype('int'), :], 0), 45, Fs)

        # Calculate the LL response for the filtered EEG signal
        LL_resp = LLf.get_LL_all(np.expand_dims(resp, [0, 1]), Fs, w)[0][0]

        # Extract the maximum LL value from the specified time window
        mx = np.max(LL_resp[int((t0 + w / 2) * Fs):int((t0 + 0.5 + w / 2) * Fs)])
        mx_all[i, 0] = mx

        # Update the number of trials for surrogate calculation
        n_trials = np.min([n_trials, len(dati.Num.values.astype('int'))])

    # Convert the max values into a numpy array
    mx_all = np.array(mx_all[:, 0])

    # Normalize the max values to the range [0, 1]
    mx_norm = (mx_all - np.min(mx_all)) / (np.max(mx_all) - np.min(mx_all))

    # Normalize the intensity values to the range [0, 1]
    Int_norm = (Int_selc - np.min(Int_selc)) / (np.max(Int_selc) - np.min(Int_selc))

    # Calculate AUC, MAX, and Pearson correlation for the true data using the helper function
    AUC, MAX, rho = get_AUC_MAX_Pearson(Int_norm, mx_norm)

    # Calculate surrogate data if the flag is set to True
    if get_surr:
        # Generate surrogate AUC, max, and Pearson values by calling the surrogate function
        AUC_surr, max_surr, rho_surr = get_AUC_surr(rc, con_trial, EEG_resp, mx_all, Int_selc, n_trial=n_trials, n=n,
                                                    w=0.25)

        # Compute p-values for AUC, MAX, and rho based on the surrogate data
        AUC_p = get_pvalue(np.array([AUC]), AUC_surr)[0]
        MAX_p = get_pvalue(np.array([MAX]), max_surr)[0]
        rho_p = get_pvalue(np.array([rho]), rho_surr)[0]
    else:
        # If surrogate testing is not required, set p-values to NaN
        AUC_p = np.nan
        MAX_p = np.nan
        rho_p = np.nan

    # Return the true data metrics and the p-values for the surrogate test
    return AUC, MAX, rho, AUC_p, MAX_p, rho_p


def get_AUC_real_mean(sc, rc, con_trial, EEG_resp, ss='Wake', w=0.25):
    """
    Calculate the maximum LL of the mean signal for each stimulation intensity for a specific connection (Sc x rc)
    and sleep state (Wake, NREM, REM).

    Arguments:
        sc: Stimulation channel
        rc: Response channel
        con_trial: DataFrame with trial information
        EEG_resp: EEG response data
        ss: Sleep state ('Wake', 'NREM', or 'REM') (default 'Wake')
        w: Window size for LL calculation (default 0.25)
        Fs: Sampling frequency (default 1000)
        t0: Time reference for calculating the LL response (default 1.0)

    Returns:
        df: DataFrame containing the AUC (LL) for each intensity, number of trials, and metadata
    """
    # Filter trials based on stimulation channel (sc), response channel (rc), artifact-free trials, and specified sleep state
    dat = con_trial[(con_trial['Stim'] == sc) &
                    (con_trial['Chan'] == rc) &
                    (con_trial['Artefact'] == 0) &
                    (con_trial['SleepState'] == ss)].reset_index(drop=True)

    if dat.empty:
        raise ValueError("No data found for the given conditions")

    # Get unique stimulation intensities
    Int_selc = np.unique(dat['Int'])

    # Initialize arrays to store results (max LL and number of trials for each intensity)
    mx_all = np.zeros(len(Int_selc))
    n_trials = np.zeros(len(Int_selc))

    # Loop over each intensity and calculate the mean LL response
    for i, intensity in enumerate(Int_selc):
        # Filter data for the current intensity
        dati = dat[dat['Int'] == intensity].reset_index(drop=True)

        # If no valid trials, skip this intensity
        if dati.empty:
            mx_all[i] = np.nan
            n_trials[i] = 0
            continue

        # Process EEG response: average across trials and apply low-pass filter
        resp = ff.lp_filter(np.nanmean(EEG_resp[rc, dati.Num.values.astype('int'), :], axis=0), 45, Fs)

        # Calculate the LL response using the filtered EEG signal
        LL_resp = LLf.get_LL_all(np.expand_dims(resp, axis=[0, 1]), Fs, w)[0][0]

        # Calculate the maximum LL response within the specified time window around t0
        mx_all[i] = np.max(LL_resp[int((t0 - w / 2) * Fs):int((t0 + 0.5 + w / 2) * Fs)])

        # Store the number of trials for the current intensity
        n_trials[i] = len(dati)

    # Create a DataFrame to store the results, including metadata (stimulation, channel, sleep state)
    df = pd.DataFrame({
        'Int': Int_selc,
        'LL': mx_all,
        'N_trial': n_trials
    })

    # Add metadata to the DataFrame
    df['Stim'] = sc
    df['Chan'] = rc
    df['SleepState'] = ss

    return df



def save_AUC_connection(con_trial, EEG_resp):
    """
        Calculate AUC, MAX, and rho values (including p-values) for each connection (stimulus channel and response channel)
        and store them in a DataFrame.

        Arguments:
            con_trial: DataFrame containing trial information (including stimulus channel, response channel, intensity, etc.)
            EEG_resp: EEG response data for each trial
            n: Number of iterations for surrogate testing (default 100)
            w: Window size for LL calculation (default 0.25)
            Fs: Sampling frequency (default 500 Hz)
            WOI: Window of interest for peak latency calculation (default 0.1)

        Returns:
            df: DataFrame with columns ["Stim", "Chan", "AUC", "MAX", "rho", "AUC_p", "MAX_p", "rho_p"]
        """
    stim_all = np.unique(con_trial['Stim'])
    chan_all = np.unique(con_trial['Chan'])
    data_rows = []  # Use for collecting rows of data
    for sc in stim_all.astype('int'):
        for rc in chan_all.astype('int'):
            dat = con_trial[
                (con_trial['Stim'] == sc) & (con_trial['Chan'] == rc) & (con_trial['Artefact'] < 1)].reset_index(
                drop=True)
            if len(dat) > 0:
                AUC, MAX, rho, AUC_p, MAX_p, rho_p = get_AUC_real(sc, rc, con_trial, EEG_resp, n=100, w=0.25)
                # get delay
                num = np.unique(
                    con_trial.loc[(con_trial.Stim == sc) & (con_trial.Chan == rc) & (con_trial.Artefact < 1), 'Num'])
                trials = EEG_resp[rc, num, :]
                WOI = 0.1
                peak_lat, _, _ = CCEP_func.peak_latency(trials, WOI, t0=1, Fs=500, w_LL=0.25)
                # Append each set of results as a new row in the data_rows list
                data_rows.append([sc, rc, AUC, MAX, rho, 1 - AUC_p, 1 - MAX_p, 1 - rho_p])

    # Create the DataFrame after the loop, using the collected rows
    df = pd.DataFrame(data_rows, columns=["Stim", "Chan", "AUC", "MAX", "rho", "AUC_p", "MAX_p", "rho_p"])
    return df


def save_AUC_connection_SS_mean(con_trial, EEG_resp,auc_summary):
    """
       For each significant connection (based on surrogate testing), calculate the AUC, MAX, and rho during different sleep states.

       Arguments:
           con_trial: DataFrame containing trial information (including stimulus, response channel, sleep state, etc.)
           EEG_resp: EEG response data for each trial
           auc_summary: DataFrame containing summary of AUC results, including significance
           n: Number of iterations for surrogate testing (default 40)
           w: Window size for LL calculation (default 0.25)

       Returns:
           df: DataFrame with columns ["Stim", "Chan", "SleepState", "AUC", "MAX", "rho"]
    """
    # List of all unique stimulus, response channels, and sleep states
    stim_all = np.unique(con_trial['Stim'])
    sleep_states = ['Wake', 'NREM', 'REM']

    # List to collect rows of data
    data_rows = []

    # Iterate over all stimulus channels
    for sc in stim_all.astype('int'):
        # Find significant response channels for the current stimulus
        respchans = np.unique(
            auc_summary.loc[(auc_summary.Stim == sc) & (auc_summary.sig_con == 1), 'Chan'])

        # Iterate over each significant response channel
        for rc in respchans:
            # Iterate over each sleep state
            for ss in sleep_states:
                # Filter data for the specific sleep state, stimulus, response channel, and artifact condition
                dat = con_trial[
                    (con_trial['SleepState'] == ss) &
                    (con_trial['Stim'] == sc) &
                    (con_trial['Chan'] == rc) &
                    (con_trial['Artefact'] < 1)
                    ].reset_index(drop=True)

                # Only proceed if there is data for the current combination
                if len(dat) > 0:
                    # Get AUC, MAX, and rho for the current connection
                    AUC, MAX, rho, _, _, _ = get_AUC_real(sc, rc, con_trial, EEG_resp, get_surr=False, n=n, w=w)

                    # Append the results as a new row in the data_rows list
                    data_rows.append([sc, rc, ss, AUC, MAX, rho])

    # Create a DataFrame from the collected data
    df = pd.DataFrame(data_rows, columns=["Stim", "Chan", "SleepState", "AUC", "MAX", "rho"])

    return df


def save_AUC_connection_SS(con_trial, EEG_resp,auc_summary):
    """
        Perform FDR correction on AUC p-values and calculate AUC, MAX, and rho for significant connections during different sleep states.

        Arguments:
            con_trial: DataFrame containing trial data including stimulus, response channel, sleep state, and other information.
            EEG_resp: EEG response data for each trial.
            auc_summary: DataFrame containing AUC summary, including p-values and significance.

        Returns:
            df_all: DataFrame containing the results for each significant connection during different sleep states.
        """

    # FDR correction on AUC p-values
    auc_summary['sig_con'] = 0
    p_values = auc_summary['AUC_p'].values
    _, p_corr = statsmodels.stats.multitest.fdrcorrection(p_values)
    auc_summary['AUC_p_FDR'] = np.array(p_corr)

    # Mark significant connections where both AUC and MAX p-values are less than 0.05
    auc_summary.loc[
        (auc_summary.AUC_p_FDR < 0.05) & (auc_summary.MAX_p < 0.05), 'sig_con'] = 1

    # Initialize the list to store results for each connection
    result_list = []

    # Iterate over all stimulus channels
    stim_all = np.unique(con_trial['Stim'])
    sleep_states = ['Wake', 'NREM', 'REM']

    for sc in stim_all.astype('int'):
        # Get the significant response channels for the current stimulus
        respchans = np.unique(
            auc_summary.loc[(auc_summary.Stim == sc) & (auc_summary.sig_con == 1), 'Chan'])

        # Iterate over each significant response channel
        for rc in respchans:
            # Iterate over each sleep state
            for ss in sleep_states:
                # Filter the data for the current combination of sleep state, stimulus, and response channel
                dat = con_trial[
                    (con_trial['SleepState'] == ss) &
                    (con_trial['Stim'] == sc) &
                    (con_trial['Chan'] == rc) &
                    (con_trial['Artefact'] < 1)
                    ].reset_index(drop=True)

                # Only proceed if there is data for the current combination
                if len(dat) > 0:
                    # Get the AUC, MAX, and rho for the current connection
                    df = get_AUC_real_mean(sc, rc, con_trial, EEG_resp, ss=ss, w=0.25)
                    result_list.append(df)

    # Concatenate all results into a single DataFrame
    df_all = pd.concat(result_list, ignore_index=True)

    return df_all


# Define a function to calculate normalized AUC for a group
def calculate_normalized_auc(group, max_overall, max_Int):
    if len(group) < 10:
        # Not enough data points to calculate AUC
        return np.nan
    else:
        # Normalize LL
        LL_norm = (group['LL'] - np.min(group['LL'])) / (max_overall - np.min(group['LL']))
        # Normalize Int
        Int_norm = group['Int'] / max_Int
        # Calculate AUC
        return sklearn.metrics.auc(Int_norm, LL_norm)

