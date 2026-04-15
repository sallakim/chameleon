import numpy as np
import multiprocess
import matplotlib.pyplot as plt
import glob
import h5py
import pandas as pd
import seaborn as sns
import time
import os
import shutil
from scipy.stats import pearsonr

# Import the utils module from the ocean package
from chameleon.utils import update_log, mad_outliers, mahalanobis_outliers

# Import cardiogrowth_py, needs to be added to search path in your Python environment
from monarch import Hatch


def run_forest_run(wave, input_file, sim_dirs, constants=None, growth=False, acute_key="_acute", posterior=False,
                   log_file=None, n_processes=multiprocess.cpu_count() - 1, file_hemo=None, m_outlier=5.0,
                   run_sims=True, remove_outliers=False, time_ticks=None, growth_type="transverse",
                   fig_name="sim_results", show_fig=False, v_ticks=(0, 50, 100, 150), p_ticks=(0, 50, 100, 150),
                   segment_lfw=0, segment_rfw=1, segment_sw=2, percentile=0.95, shortening_data=None,
                   show_hemo=False, show_rv_hemo=False, show_stretch=False, plot_only_all=False, print_log=None):
    """Wrapper function to run, analyze, and import model simulations"""

    # Specify directory to store simulations in
    file_path = wave.dir_sim
    fig_dir = wave.dir / "sim_results"

    print_log = wave.print_log if print_log is None else print_log

    # If running posterior simulations, pull according parameter sets
    if posterior:
        wave.x_sim = wave.x_posterior
        file_path = wave.dir.parent / wave.posterior_label
        fig_dir = wave.dir.parent / "Results"
        remove_outliers = True
        update_log(wave.log_file, "\n---------\nPosterior\n---------", print_log)

    # Run models
    if run_sims:

        # Remove prior sim dir if rerunning simulations
        if run_sims:
            shutil.rmtree(file_path, ignore_errors=True)

        # Create sim directory
        os.makedirs(file_path, exist_ok=True)

        run_models_par(wave.x_sim, wave.x_names, file_path, input_file, constants=constants, acute_key=acute_key,
                       log_file=log_file, n_processes=n_processes, growth=growth, file_hemo=file_hemo,
                       time_ticks=time_ticks, growth_type=growth_type, print_log=print_log)

    # Analyze model simulations
    if growth:
        analyze_model_growth(file_path, wave.x_names, wave.y_names, m_outlier=m_outlier, log_file=log_file,
                             remove_outliers=remove_outliers, percentile=percentile, print_log=print_log)
    else:
        analyze_model(file_path, wave.x_names, wave.y_names, m_outlier=m_outlier, log_file=log_file,
                      remove_outliers=remove_outliers, percentile=percentile, shortening_data=shortening_data,
                      print_log=print_log)

    # Import results of all model simulations ran for current and all previous waves
    sim_dirs.append(file_path)
    x_sim, y_sim = import_model_results(sim_dirs, wave.x_names, wave.y_names)

    if growth:
        # Plot growth results
        plot_sims_growth(file_path, fig_dir, wave.x_names, wave.y_names, wave.y_observed, wave.sigma_observed,
                         time_ticks=time_ticks, show_fig=show_fig, fig_name=fig_name, only_all=plot_only_all)
    else:
        # Plot model results
        plot_sims(file_path, fig_dir, x_labels=wave.x_names, fig_name=fig_name, show_fig=show_fig, show_hemo=show_hemo, show_rv_hemo=show_rv_hemo, # salla addition of rv
                  show_stretch=show_stretch, segment_lfw=segment_lfw, segment_rfw=segment_rfw, segment_sw=segment_sw,
                  v_ticks=v_ticks, p_ticks=p_ticks)

    return x_sim, y_sim, sim_dirs


def run_models_par(x_model, x_names, file_path, input_file, constants=None, acute_key="_acute",
                   n_processes=multiprocess.cpu_count() - 1, log_file=None, growth=False, file_hemo=None,
                   time_ticks=None, growth_type="transverse", print_log=True):
    """
    Run model for all input parameter sets in x_model using parallel computing to reduce computational time
    """

    if constants is None:
        constants = {}

    update_log(log_file, "Running " + str(x_model.shape[0]) + " model simulations...", print_log=print_log)

    # Number of simulations to be run
    n_sims = x_model.shape[0]

    # Time simulation time
    t0 = time.time()

    with multiprocess.Pool(processes=n_processes) as pool:
        if growth:
            pool.starmap(run_growth_par, list(zip(x_model, [x_names] * n_sims, np.arange(n_sims),
                                                  [file_path] * n_sims, [input_file] * n_sims,
                                                  [constants] * n_sims, [time_ticks] * n_sims,
                                                  [growth_type] * n_sims)))
        else:
            pool.starmap(run_model_par, list(zip(x_model, [x_names] * n_sims, np.arange(n_sims), [file_path] * n_sims,
                                                 [input_file] * n_sims, [constants] * n_sims)))

    t1 = time.time() - t0

    # Check if acute_key is a substring in any of x_names
    if growth:
        update_log(log_file, "%i" % n_sims + " growth simulations completed in %.2f seconds" % t1, print_log=print_log)
    else:
        if any([acute_key in x_name for x_name in x_names]):
            update_log(log_file, "%i" % n_sims + " simulation pairs (baseline + acute) completed in %.2f seconds" % t1,
                       print_log=print_log)
        else:
            update_log(log_file, "%i" % n_sims + " simulations completed in %.2f seconds" % t1, print_log=print_log)

    # Return total simulation time
    return t1


def run_model_par(x_model, x_names, i_x, file_path, input_file, constants={}, model_id0=0, acute_key="_acute"):
    """
    Run a single model for an input parameter set in x_model to calculate output parameters y: make sure they match
    with the user-defined input parameters in the main code. If acute_key is a substring in any of x_names, an acute
    simulation will be performed after the baseline simulation.
    """

    # Initialize CardioGrowth
    beat = Hatch(input_file)

    # Export file name
    file_name = f'{model_id0 + i_x:05d}'

    # Construct dictionary of model parameters
    pars = {x_names[i]: x_model[i] for i in range(len(x_names))}

    # Extract baseline parameters
    pars_baseline = {key: val for key, val in pars.items() if acute_key not in key}
    constants_baseline = {key: val for key, val in constants.items() if acute_key not in key}

    # Extract acute parameters, if any
    pars_acute = {key: val for key, val in pars.items() if acute_key in key}
    constants_acute = {key: val for key, val in constants.items() if acute_key in key}

    # Change model parameters
    beat.change_pars(pars_baseline)
    beat.change_pars(constants_baseline)

    # Run baseline model, do not use converged solution to be compatible with parallel computing
    try:
        beat.just_beat_it(print_solve=False, file_path=file_path, file_name=file_name, use_converged=False)
    except:
        return

    # Perform acute simulation if needed
    if (len(pars_acute) > 0) or (len(constants_acute) > 0):
        # Strip acute key from parameter names
        pars_acute = {key.replace(acute_key, ""): value for key, value in pars_acute.items()}
        constants_acute = {key.replace(acute_key, ""): value for key, value in constants_acute.items()}

        # Change model parameters
        beat.change_pars(pars_acute)
        beat.change_pars(constants_acute)

        # Run acute model, use converged solution from baseline simulation as starting point
        beat.circulation.k = beat.volumes[0, :] / beat.circulation.sbv
        beat.just_beat_it(print_solve=False, file_path=file_path, file_name=file_name + acute_key, use_converged=False)

    # Save real (unscaled) parameter values x
    np.save(file_path / file_name, x_model)


def run_growth_par(x_model, x_names, i_x, file_path, input_file, constants={}, time_ticks=None,
                   growth_type="transverse", model_id0=0, acute_key="_acute"):
    """
    Run a single growth model for an input parameter set in x_model to calculate output parameters y: make sure they
    match with the user-defined input parameters in the main code.
    """

    # Initialize CardioGrowth
    beat = Hatch(input_file)

    # Export file name
    file_name = "growth_" + f"{model_id0 + i_x:05d}"

    # Construct dictionary of model parameters
    pars = {x_names[i]: x_model[i] for i in range(len(x_names))}

    # Change baseline and acute parameters to a random fit within previous baseline/acute fit
    beat.change_pars(pars)
    beat.change_pars(constants)

    # Set growth hemodynamics using the acute parameters
    cg = set_ras_growth(beat, pars, acute_key=acute_key, time_ticks=time_ticks)

    # Run growth model, do not use converged solution to be compatible with parallel computing
    try:
        cg.let_it_grow(file_path, file_name, use_converged=False, print_solve=False)

        # Save real (unscaled) parameter values x
        np.save(file_path / file_name, x_model)
    except:
        pass

def set_ras_growth(beat, pars, ras_label="Ras", sbv_label="SBV", rmvb_label="Rmvb", rmvb_baseline=1e10,
                   acute_key="_acute", time_ticks=None, drmvb_label="dRmvb", taurmvb_label="tauRmvb",
                   dras_label="dRas", tauras_label="tauRas"):
    """Set Ras and SBV throughout growth based on csv file and parameter values"""

    t_growth = beat.growth.time

    # Set SBV to be constant from acute onwards
    if sbv_label in pars.keys():
        beat.growth.sbv[:] = pars[sbv_label]
    if sbv_label + acute_key in pars.keys():
        beat.growth.sbv[1:] = pars[sbv_label + acute_key]

    # Set baseline and acute Ras
    if ras_label in pars.keys():
        beat.growth.ras[:] = pars[ras_label]
    if ras_label + acute_key in pars.keys():
        beat.growth.ras[1:] = pars[ras_label + acute_key]

    # Set baseline and acute Rmvb
    if rmvb_label in pars.keys():
        beat.growth.rmvb[:] = pars[rmvb_label]
    if rmvb_label + acute_key in pars.keys():
        beat.growth.rmvb[1:] = pars[rmvb_label + acute_key]

    # Use exponential decay function to set Rmvb
    if drmvb_label in pars.keys() and taurmvb_label in pars.keys():
        beat.growth.rmvb[1:] = (pars[drmvb_label] * beat.growth.rmvb[1] - (pars[drmvb_label] - 1) * beat.growth.rmvb[1]
                                * np.exp(-beat.growth.time[1:] / pars[taurmvb_label]))

    # Use exponential decay function to set Ras
    if dras_label in pars.keys() and tauras_label in pars.keys():
        beat.growth.ras[1:] = (pars[dras_label] * beat.growth.ras[1] - (pars[dras_label] - 1) * beat.growth.ras[1]
                               * np.exp(-beat.growth.time[1:] / pars[tauras_label]))

    if rmvb_label + "_final" in pars.keys():
        beat.growth.rmvb[-1] = pars[rmvb_label + "_final"]
        # interpolate from acute to final value
        beat.growth.rmvb[2:] = np.interp(t_growth[2:], [t_growth[1], t_growth[-1]],
                                         [beat.growth.rmvb[1], beat.growth.rmvb[-1]])

    if ras_label + "_final" in pars.keys():
        beat.growth.ras[-1] = pars[ras_label + "_final"]
        # interpolate from acute to final value
        beat.growth.ras[2:] = np.interp(t_growth[2:], [t_growth[1], t_growth[-1]], [beat.growth.ras[1], beat.growth.ras[-1]])

    return beat


def get_model_files(file_path, acute_key="_acute", file_extension=".hdf5"):
    """Return list if converged model output files and their corresponding parameter files"""

    # Get cardiogrowth simulation output files and temporarily remove the file extension
    sim_files = sorted(glob.glob(str(file_path) + "/*" + file_extension))
    sim_files = [sim_file.split(file_extension)[0] for sim_file in sim_files]

    # Make separate list with acute simulations (if any) and exclude acute files from the original list
    sim_files_acute = [x for x in sim_files if acute_key in x]
    sim_files = [x for x in sim_files if acute_key not in x]

    # Cross-check and only keep simulations if both baseline and acute simulations converged
    if len(sim_files_acute) > 0:
        sim_files = [x for x in sim_files if x + acute_key in sim_files_acute]

    # Get corresponding parameter files and return original file extension
    par_files = [sim_file + ".npy" for sim_file in sim_files]
    sim_files = [sim_file + file_extension for sim_file in sim_files]

    return sim_files, par_files


def analyze_model(file_path, x_labels, y_labels, sim_results_name="sim_results", m_outlier=None, shortening_data=None,
                  acute_key="_acute", file_extension=".hdf5", log_file=None, remove_outliers=False, percentile=0.95,
                  print_log=True):
    """
    Obtains results y_sim from all simulations with input parameters x previously stored in file_path. Unconverged solutions
    (which have no output file) are skipped and assigned NaNs. When plot_results=True, LV and RV PV loops, strain
    curves, and geometry at end-diastole are stored as well, but this makes it more time-consuming. Outliers, i.e. with
    outputs with standard deviation * m_std_exclude (default=1.96, i.e. 95% confidence interval), are omitted from analysis.
    """

    # Find all converged simulations and corresponding parameter outputs
    sims, pars = get_model_files(file_path)
    update_log(log_file, str(len(sims)) + " Simulations reached convergence", print_log=print_log)

    # Pre-allocate arrays to store x and y.
    x_sims = np.empty((0, len(x_labels)))
    y_sims = np.empty((0, len(y_labels)))

    # Check for presence of acute simulations
    y_labels_acute = [y_label for y_label in y_labels if acute_key in y_label]

    # Check for presence of rho labels and put them in seperate list
    y_labels_rho = [y_label for y_label in y_labels if "rho" in y_label]

    # Remaining labels are baseline labels
    y_labels = [y_label for y_label in y_labels if acute_key not in y_label and "rho" not in y_label]

    for sim, par in zip(sims, pars):

        # Load parameters and add to the collection
        x_sims = np.vstack((x_sims, np.load(par)))

        # Load model readout
        with h5py.File(sim, "r", locking=False) as f:
            lab_f = f['lab_f'][:]
            outputs = f['outputs'][0]
            output_names = list(f.attrs['outputs_names'][:])

        # Convert output names to lower case to prevent typographic mistakes from causing errors
        output_names = [output_name.lower() for output_name in output_names]

        # Collect output results
        y_sim = []
        for y_label in y_labels:
            y_sim.append(outputs[output_names.index(y_label.lower())])

        if len(y_labels_rho) > 0:
            segments = [int(y_label.split("_s")[1]) for y_label in y_labels_rho]
            y_sim.extend(correlate_shortening(shortening_data, lab_f, int(outputs[output_names.index("ied")]), segments))

        # Load acute model readout, if present
        if len(y_labels_acute) > 0:
            with h5py.File(sim.split(file_extension)[0] + acute_key + file_extension, "r") as f:
                outputs = f['outputs'][0]
                output_names = list(f.attrs['outputs_names'])

            # Convert output names to lower case to prevent typographic mistakes from causing errors
            output_names = [output_name.lower() for output_name in output_names]

            for y_label in y_labels_acute:
                y_sim.append(outputs[output_names.index(y_label.split(acute_key)[0].lower())])

        # Add all baseline and acute outputs to stack
        y_sims = np.vstack((y_sims, y_sim))

    # Omit any simulations with nans
    nan_indices = np.unique(np.where(np.isnan(y_sims))[0])
    x_sims = np.delete(x_sims, nan_indices, axis=0)
    y_sims = np.delete(y_sims, nan_indices, axis=0)

    # Omit outliers, i.e. values outside m_std_outlier times the standard deviation from the median
    x_sims, y_sims = filter_outliers(x_sims, y_sims, m_outlier=m_outlier, percentile=percentile,
                                     remove=remove_outliers, sims=sims, pars=pars)

    # Export simulated x and y into csv file
    export_x_y(x_sims, y_sims, x_labels, y_labels + y_labels_rho + y_labels_acute, file_path, sim_results_name)

    update_log(log_file, str(x_sims.shape[0]) + " Simulations added to training data", print_log=print_log)

    return x_sims, y_sims


def analyze_model_growth(file_path, x_labels, y_labels, sim_results_name="sim_results", m_outlier=None,
                         time_turner="_d", log_file=None, remove_outliers=False, percentile=0.95, print_log=True):
    """
    Obtains results y_sim from all simulations with input parameters x previously stored in file_path. Unconverged solutions
    (which have no output file) are skipped and assigned NaNs. When plot_results=True, LV and RV PV loops, strain
    curves, and geometry at end-diastole are stored as well, but this makes it more time-consuming. Outliers, i.e. with
    outputs with standard deviation * m_std_exclude (default=1.96, i.e. 95% confidence interval), are omitted from analysis.
    """

    # Find all converged simulations and corresponding parameter outputs
    sims, pars = get_model_files(file_path)
    update_log(log_file, str(len(sims)) + " Simulations reached convergence", print_log=print_log)

    # Pre-allocate arrays to store x and y.
    x_sims = np.empty((0, len(x_labels)))
    y_sims = np.empty((0, len(y_labels)))

    for sim, par in zip(sims, pars):

        # Load parameters and add to the collection
        x_sims = np.vstack((x_sims, np.load(par)))

        # Load model readout
        with h5py.File(sim, "r", locking=False) as f:
            outputs = f['outputs'][:]
            output_names = list(f.attrs['outputs_columns'])
            output_index = list(f.attrs['outputs_rows'])

        # Convert output names to lower case to prevent typographic mistakes from causing errors
        output_names = [output_name.lower() for output_name in output_names]

        # Collect output results
        y_sim = []
        for y_label in y_labels:
            y_label_time = -1
            # Split label into name and time
            y_label_name = y_label.split(time_turner)[0]
            if "_acute" in y_label_name:
                y_label_name = y_label_name.split("_acute")[0]
                y_label_time = 0
            if time_turner in y_label:
                y_label_time = float(y_label.split(time_turner)[1])
            y_sim.append(outputs[output_index.index(y_label_time), output_names.index(y_label_name.lower())])

        # Add all baseline and acute outputs to stack
        y_sims = np.vstack((y_sims, y_sim))

    # Omit outliers, i.e. values outside m_std_outlier times the standard deviation from the median
    x_sims, y_sims = filter_outliers(x_sims, y_sims, m_outlier=m_outlier, remove=remove_outliers,
                                     sims=sims, pars=pars, percentile=percentile)

    # Export simulated x and y into csv file
    export_x_y(x_sims, y_sims, x_labels, y_labels, file_path, sim_results_name)

    update_log(log_file, str(x_sims.shape[0]) + " Simulations added to training data", print_log=print_log)

    return x_sims, y_sims


def correlate_shortening(data_shortening, lab_f, i_ed, segments):
    """Calculate Pearsson correlation coefficient between data and simulated shortening"""

    # Shift simulation results to start with ED and calculate shortening
    lab_f = np.roll(lab_f, -i_ed, axis=1)
    shortening = lab_f/lab_f[0, :] - 1

    # Correlate shortening curves for each segment
    rho = []
    for segment in segments:
        r, _ = pearsonr(data_shortening[:, segment], shortening[:, segment])
        rho.append(r)

    return rho

def export_x_y(x, y, x_labels, y_labels, file_path, file_name):
    """Export simulated or emulated x and into a csv file"""

    df = pd.DataFrame(np.concatenate((x, y), axis=1), columns=np.append(x_labels, y_labels))
    df.to_csv(file_path / str(file_name + ".csv"))


def import_model_results(sim_dirs, x_labels, y_labels, sim_results_name="sim_results"):
    """Import model results from all waves and scale y values to min and max of the simulated values"""

    # Find all simulation data, stored in .csv files
    sim_files = []
    for sim_dir in sim_dirs:
        sim_files.append(sim_dir / str(sim_results_name + '.csv'))

    df = pd.concat((pd.read_csv(f) for f in sim_files), ignore_index=True)

    x_sim = df[list(x_labels)].values
    y_sim = df[list(y_labels)].values

    return x_sim, y_sim


def filter_outliers(x, y, m_outlier=None, percentile=0.95, remove=False, sims=None, pars=None):
    """Omit outliers from simulation data using median absolute deviation and Mahalanobis distance"""

    # Initialize list of outliers indices
    outliers = []
    inliers = range(y.shape[0])

    # Step 1: MAD filtering
    if m_outlier is not None:
        if m_outlier > 0:
            outliers = mad_outliers(y, m_outlier=m_outlier)
            inliers = [i for i in range(y.shape[0]) if i not in outliers]

    # Step 2: Mahalanobis filtering, only analyze current inliers
    if percentile is not None:
        outliers_mahalanobis = mahalanobis_outliers(y[inliers, :], percentile=percentile)
    else:
        outliers_mahalanobis = []

    # Mahalanobis distance outliers are a subset of the MAD outliers, add to total set of outliers
    outliers.extend([inliers[i] for i in outliers_mahalanobis])

    # Remove outliers from x_sim and y_sim
    x = np.delete(x, outliers, axis=0)
    y = np.delete(y, outliers, axis=0)

    # Delete all outliers from simulation and parameter files if enabled
    if remove:
        if (sims is not None) and (pars is not None):
            for i in sorted(outliers, reverse=True):
                os.remove(sims[i])
                os.remove(pars[i])
                del sims[i]
                del pars[i]

    return x, y


def plot_sims(sim_dir, exp_dir, x_labels, fig_name="simresults", show_fig=False,
              color_main="#375441", color_space="#bfc4ac", segment_lfw=0, segment_sw=2, segment_rfw=1,
              acute_key="_acute", file_extension=".hdf5", show_hemo=False, show_rv_hemo=False, show_stretch=False,
              p_ticks=(0, 50, 100, 150, 200), v_ticks=(0, 100, 200, 300), lab_ticks=(0.8, 1.0, 1.2, 1.4),
              p_label="LV Pressure (mmHg)", v_label="Volume (mL)", lab_label="Stretch (-)", t_label="Time (s)"):
    """Plot predicted model outcome of all simulations in the directory sim_dir and saves plots to exp_dir"""

    # Find all simulation results
    sims, pars = get_model_files(sim_dir)

    # Check if acute simulations were used
    has_acute = any(acute_key in x for x in x_labels)

    # Load first simulation and parameters to find data dimensions. Has to be equal across all simulations
    times, volumes, pressures, rv_volumes, rv_pressures, stretches_lfw, \
        stretches_sw, stretches_rfw, x_sims = load_sim_data(sims, pars, segment_lfw, segment_sw, segment_rfw) # salla addition of RV

    if has_acute:
        sims_acute = [sim.split(file_extension)[0] + acute_key + file_extension for sim in sims]
        times_acute, volumes_acute , pressures_acute, rv_volumes_acute, rv_pressures_acute, stretches_lfw_acute, stretches_sw_acute, \
            stretches_rfw_acute, _ = load_sim_data(sims_acute, pars, segment_lfw, segment_sw, segment_rfw)

    # Find median simulation
    i_median = get_median_sim(x_sims)

    # Salla Change: added RV hemodynamics
    # Plot LV and RV hemodynamics
    fig = plt.figure(figsize=(15, 10), linewidth=1.0)
    sns.set_theme(style="white", palette=None)

    # Row 1: LV
    plot_xy(fig, 1, [2, 3], volumes, pressures, v_label, p_label, color_space, color_main, 
            x_ticks=v_ticks, y_ticks=p_ticks, title="LV PV Loop")
    plot_xy(fig, 2, [2, 3], times, volumes, t_label, v_label, color_space, color_main,
            title="LV Volume", x_ticks=(times[0, 0], times[-1, -1]), y_ticks=v_ticks)
    plot_xy(fig, 3, [2, 3], times, pressures, t_label, p_label, color_space, color_main,
            title="LV Pressure", x_ticks=(times[0, 0], times[-1, -1]), y_ticks=p_ticks)
    
    # Row 2: RV
    plot_xy(fig, 4, [2, 3], rv_volumes, rv_pressures, v_label, p_label, color_space, color_main, 
            x_ticks=v_ticks, y_ticks=p_ticks, title="RV PV Loop")
    plot_xy(fig, 5, [2, 3], times, rv_volumes, t_label, v_label, color_space, color_main,
            title="RV Volume", x_ticks=(times[0, 0], times[-1, -1]), y_ticks=v_ticks)
    plot_xy(fig, 6, [2, 3], times, rv_pressures, t_label, p_label, color_space, color_main,
            title="RV Pressure", x_ticks=(times[0, 0], times[-1, -1]), y_ticks=p_ticks)
    
    finish_plots(fig, exp_dir, str(fig_name + "_hemodynamics.pdf"), show_fig=show_fig or show_hemo or show_rv_hemo)
    # End Salla addition

    # Plot 2: Stretch
    if has_acute:
        plot_shape = [2, 3]
    else:
        plot_shape = [1, 3]
    
    fig = plt.figure(figsize=(10, 4 + has_acute * 4), linewidth=1.0)
    plot_xy(fig, 1, plot_shape, times, stretches_lfw, t_label, lab_label, color_space, color_main, y_ticks=lab_ticks,
            title="Left free wall", x_lim=(times[0, 0], times[-1, -1]))
    plot_xy(fig, 2, plot_shape, times, stretches_sw, t_label, lab_label, color_space, color_main, y_ticks=lab_ticks,
            title="Septal wall", x_lim=(times[0, 0], times[-1, -1]))
    plot_xy(fig, 3, plot_shape, times, stretches_rfw, t_label, lab_label, color_space, color_main, y_ticks=lab_ticks,
            title="Right free wall", x_lim=(times[0, 0], times[-1, -1]))
    if has_acute:
        plot_xy(fig, 4, plot_shape, times_acute, stretches_lfw_acute, t_label, lab_label, color_space, color_main,
                y_ticks=lab_ticks, title="Left free wall", x_lim=(times_acute[0, 0], times_acute[-1, -1]))
        plot_xy(fig, 5, plot_shape, times_acute, stretches_sw_acute, t_label, lab_label, color_space, color_main,
                y_ticks=lab_ticks, title="Septal wall", x_lim=(times_acute[0, 0], times_acute[-1, -1]))
        plot_xy(fig, 6, plot_shape, times_acute, stretches_rfw_acute, t_label, lab_label, color_space, color_main,
                y_ticks=lab_ticks, title="Right free wall", x_lim=(times_acute[0, 0], times_acute[-1, -1]))
    finish_plots(fig, exp_dir, str(fig_name + "_stretch.pdf"), show_fig=show_fig or show_stretch)


def plot_sims_growth(sim_dir, exp_dir, x_labels, y_labels, y_observed, y_observed_sigma, time_ticks=None,
                     time_turner="_d", color_main="#bc5090", color_space="#58508d", fig_name="simgrowth", legend=True,
                     acute_key="_acute", show_fig=False, only_all=False, analyze_growth=True, y_lims=None, y_ticks=None,
                     translations=None):
    """Plot predicted model outcome of all simulations in the directory sim_dir and saves plots to exp_dir"""

    exp_dir.mkdir(exist_ok=True, parents=True)

    # Get all unique y_labels without time_turner label or acute label
    y_just_labels_acute = [y_label.split(time_turner)[0] for y_label in y_labels]
    y_just_labels = [y_label.split(acute_key)[0] for y_label in y_just_labels_acute]
    y_just_labels = list(set(y_just_labels))

    y_time, y_data, y_std = [], [], []
    for y_label in y_just_labels:
        # Find all y_labels with current y_label
        y_labels_current = [label for label in y_labels if y_label in label]

        # Get time, mean, and sigma for all y_labels with current y_label
        y_time.append([float(label.split(time_turner)[1]) for label in y_labels_current])
        y_data.append([y_observed[i] for i, label in enumerate(y_labels) if y_label in label])
        y_std.append([y_observed_sigma[i] for i, label in enumerate(y_labels) if y_label in label])

    # Get all simulation results and parameters
    x_sims, y_sims = load_sim_data_growth(sim_dir, x_labels, y_just_labels, time_turner="_d")

    # Set default theme to override potential theme changes by Jupyter
    sns.set_theme(style="white", palette=None)

    # Convert names to plot format, if translations are provided, and update data headings
    y_just_labels = translate_names(y_just_labels, extra_translations=translations)

    # Change column names of y_sims to plot format
    y_sims.columns = translate_names(y_sims.columns, extra_translations=translations)

    for i in range(len(y_just_labels)):

        # Data + mean + CI + all sims
        plot_xy_growth(np.array(y_time[i]), np.array(y_data[i]), np.array(y_std[i]), y_sims, x_sims,
                       y_just_labels[i], color_space, color_main, exp_dir, fig_name, y_lims=y_lims, y_ticks=y_ticks,
                       subfix="_ci_all", x_ticks=time_ticks, ci=True, sim_space=True, show_fig=show_fig, legend=legend)

        if not only_all:
            # Data + Mean simulation
            plot_xy_growth(np.array(y_time[i]), np.array(y_data[i]), np.array(y_std[i]), y_sims, x_sims,
                           y_just_labels[i], color_space, color_main, exp_dir, fig_name, y_lims=y_lims, y_ticks=y_ticks,
                           subfix="_mean", x_ticks=time_ticks, ci=False, sim_space=False, show_fig=show_fig, legend=legend)

            # Data + mean + CI
            plot_xy_growth(np.array(y_time[i]), np.array(y_data[i]), np.array(y_std[i]), y_sims, x_sims,
                           y_just_labels[i], color_space, color_main, exp_dir, fig_name, y_lims=y_lims, y_ticks=y_ticks,
                           subfix="_ci", x_ticks=time_ticks, ci=True, sim_space=False, show_fig=show_fig, legend=legend)

            # Data + mean + all sims
            plot_xy_growth(np.array(y_time[i]), np.array(y_data[i]), np.array(y_std[i]), y_sims, x_sims,
                           y_just_labels[i], color_space, color_main, exp_dir, fig_name, y_lims=y_lims, y_ticks=y_ticks,
                           subfix="all", x_ticks=time_ticks, ci=False, sim_space=True, show_fig=show_fig, legend=legend)
            # Plot geometry
            # plot_sims_geo(exp_dir, fig_name, y_just_labels[i], y_sims, x_sims, color_space, color_main, show_fig=show_fig)
    if analyze_growth:
        plot_growth(sim_dir, exp_dir, show_fig=show_fig)

#
# def plot_sims_geo(sim_dir, exp_dir, fig_name, y_label, y_sims, x_sims, color_space, color_main, save_fig=True, show_fig=show_fig):
#     """Plot geometry of all simulations in the directory sim_dir and saves plots to exp_dir"""
#
#     # Get all simulation results and parameters
#     x_sims, y_sims = load_sim_data_growth(sim_dir, time_turner="_d")
#     for i in y_sims:
#         time_frame = y_sims[i]["IED"][:]
#         x_m = y_sims[i]["x_m"][time_frame, :]
#         r_m = y_sims[i]["r_m"][time_frame, :]
#         wall_thickness = np.array([])
#
#         fig, ax = plt.subplots()
#
#         plot_geometry(x_m, r_m, wall_thickness, ax)
#
#     if save_fig:
#         plt.savefig(os.path.join("geometry.pdf"), bbox_inches='tight')
#     if show_fig:
#         plt.show()
#     else:
#         plt.close()

def plot_growth(sim_dir, exp_dir, show_fig=False, fig_name="sim_growth.pdf"):
    """PLot growth and stimulus functions of all simulations in the directory sim_dir and saves plots to exp_dir"""

    # Get all simulation files
    sims, pars = get_model_files(sim_dir)

    fig, ax = plt.subplots(2, 4, figsize=(10, 4))
    cmap = sns.color_palette("Set2", len(sims), as_cmap=False)
    # Load all posterior simulations
    for sim in sims:
        with h5py.File(sim, "r") as f:
            tg = f.attrs['outputs_rows'][:]
            f_g = f["f_g"][:]
            s_l = f["s_l"][:]
            s_r = f["s_r"][:]
            s_l_set = f["s_l_set"][:]
            s_r_set = f["s_r_set"][:]
            lab_f_max = f["lab_f_max"][:]
            ax[0, 0].plot(tg, f_g[:, 0, 0], color=cmap[sims.index(sim)])
            ax[1, 0].plot(tg, f_g[:, 2, 0], color=cmap[sims.index(sim)])
            ax[0, 1].plot(tg, lab_f_max[:, 0], color=cmap[sims.index(sim)])
            ax[1, 1].plot(tg, 1/lab_f_max[:, 0]**2, color=cmap[sims.index(sim)])
            ax[0, 2].plot(tg[2:], s_l[2:, 0], color=cmap[sims.index(sim)])
            ax[1, 2].plot(tg[2:], s_r[2:, 0], color=cmap[sims.index(sim)])
            ax[0, 3].plot(tg[2:], s_l_set[2:, 0], color=cmap[sims.index(sim)])
            ax[1, 3].plot(tg[2:], s_r_set[2:, 0], color=cmap[sims.index(sim)])

    ax[0, 0].set_ylabel(r"$F_{g,ff}$ (-)")
    ax[1, 0].set_ylabel(r"$F_{g,rr}$ (-)")
    ax[0, 1].set_ylabel(r"$\lambda_{e,ff}$ (-)")
    ax[1, 1].set_ylabel(r"$\lambda_{e,rr}$ (-)")
    ax[0, 2].set_ylabel(r"$s_f$ (-)")
    ax[1, 2].set_ylabel(r"$s_r$ (-)")
    ax[0, 3].set_ylabel(r"$s_{f,set}$ (-)")
    ax[1, 3].set_ylabel(r"$s_{r,set}$ (-)")

    for i, axi in enumerate(ax):
        for axij in axi:
            if i == 1:
                axij.set_xlabel("Time (s)")
            else:
                axij.set_xticks([])

            axij.set_xlim([tg[0], tg[-1]])
            # Make plot look nicer
            axij.spines['top'].set_visible(False)
            axij.spines['right'].set_visible(False)
            axij.spines['bottom'].set_visible(False)
            # axij.spines['left'].set_visible(False)
            axij.tick_params(axis='both', which='both', length=0)
            axij.set_axisbelow(True)

            for axis in ['top', 'bottom', 'left', 'right']:
                axij.spines[axis].set_linewidth(1.5)

    plt.tight_layout()
    if exp_dir is not None:
        fig.savefig(exp_dir / fig_name, bbox_inches='tight')
    if show_fig:
        plt.show()
    else:
        plt.close()


def load_sim_data_growth(sim_dir, x_labels, y_labels, time_turner="_d"):
    # Find all converged simulations and corresponding parameter outputs
    sims, pars = get_model_files(sim_dir)

    # Pre-allocate arrays to store x
    x_sims = np.empty((0, len(x_labels)))

    # Get all simulation results
    for i_sim, sim, par in zip(range(len(sims)), sims, pars):

        # Load parameters and add to the collection
        x_sims = np.vstack((x_sims, np.load(par)))

        # Load model readout
        with h5py.File(sim, "r", locking=False) as f:
            outputs = f['outputs'][:]
            output_names = list(f.attrs['outputs_columns'])
            time_g = f.attrs['outputs_rows'][:]

        # Convert output names to lower case to prevent typographic mistakes from causing errors
        output_names = [output_name.lower() for output_name in output_names]

        # Collect output results
        y_sim = np.zeros((time_g.size, len(y_labels)))
        for i, y_label in enumerate(y_labels):
            y_sim[:, i] = outputs[:, output_names.index(y_label.lower())]

        df = pd.DataFrame(data=y_sim, columns=y_labels)
        df["sim_id"] = i_sim
        df["time"] = time_g

        # Add df to df_sim
        if i_sim == 0:
            df_sims = df
        else:
            df_sims = pd.concat([df_sims, df])

    return x_sims, df_sims


def load_sim_data(sims, pars, segment_lfw, segment_sw, segment_rfw):
    n_sims = len(sims)

    with h5py.File(sims[0], "r") as f:
        n_inc = f["time"][:].size
    n_x = np.load(pars[0]).size

    times = np.zeros((n_sims, n_inc))
    volumes = np.zeros((n_sims, n_inc))
    pressures = np.zeros((n_sims, n_inc))
    rv_volumes = np.zeros((n_sims, n_inc))  # salla addition
    rv_pressures = np.zeros((n_sims, n_inc))  # salla
    stretches_lfw = np.zeros((n_sims, n_inc))
    stretches_sw = np.zeros((n_sims, n_inc))
    stretches_rfw = np.zeros((n_sims, n_inc))
    x_sims = np.zeros((n_sims, n_x))

    for i_sim, (sim, par) in enumerate(zip(sims, pars)):
        # Baseline
        with h5py.File(sim, "r", locking=False) as f:
            time = f["time"][:] * 1e3
            volume = f["volumes"][:, 2] 
            pressure = f["pressures"][:, 2] 
            rv_volume = f["volumes"][:, 6] # salla addition, add RV index 6
            rv_pressure = f["pressures"][:, 6] # salla addition, add RV index 6
            stretch = f["lab_f"][:]

        # Store simulated pressures and volumes and model parameters
        times[i_sim, :] = time
        volumes[i_sim, :] = volume
        pressures[i_sim, :] = pressure
        rv_volumes[i_sim, :] = rv_volume # salla addition
        rv_pressures[i_sim, :] = rv_pressure #salla addition
        stretches_lfw[i_sim, :] = stretch[:, segment_lfw]
        stretches_sw[i_sim, :] = stretch[:, segment_sw]
        stretches_rfw[i_sim, :] = stretch[:, segment_rfw]
        x_sims[i_sim, :] = np.load(par)

    return times, volumes, pressures, rv_volumes, rv_pressures, stretches_lfw, stretches_sw, stretches_rfw, x_sims # salla addition


def get_median_sim(x_sims):
    """Select "median simulation outcome": Select parameter set with the lowest euclidean distance to the multidimensional
    median of the parameter space. Normalize the distances to prevent parameters with higher orders of magnitude to
    dominate the distance calculation"""
    x_median = np.median(x_sims, axis=0)
    dist = []
    for x_sim in x_sims:
        dist.append(np.linalg.norm((x_sim - x_median) / x_median))
    return np.argmin(dist)


def estimate_amref_range(pars, data_mean, data_names, lab_lim=(0.8, 1.5)):
    """Estimate range for am_ref based on EDV and wall volume"""

    edv = np.array([data_mean[data_names.index("LVEDV")], data_mean[data_names.index("RVEDV")]])

    # Midwall area at ED: EDV plus half the wall volume of the LV, RV is thin enough to not make a significant difference
    par_lvwv = next(item for item in pars if item["name"] == "LVWV")
    lvwv = 0.5 * (par_lvwv["limits"][0] + par_lvwv["limits"][1])
    v_m_edv = edv * 1e3 + 0.5 * np.array([lvwv, 0])
    am_edv = np.pi ** (1 / 3) * (6 * v_m_edv) ** (2 / 3)

    # Split up LV into Lfw and Sw assuming 11/5 ratio based on AHA segment ratio
    am_edv = np.array([am_edv[0] * 11 / 16, am_edv[1], am_edv[0] * 5 / 16])

    # Estimate minimum and maximum possible Am_refs based on potential wall stretch
    am_ref_min = np.floor(am_edv * lab_lim[1] ** (-2) / 100) * 100
    am_ref_max = np.ceil(am_edv * lab_lim[0] ** (-2) / 100) * 100

    for par in pars:
        if par["name"] == "AmRefLfw":
            par["limits"] = [am_ref_min[0], am_ref_max[0]]
        if par["name"] == "AmRefRfw":
            par["limits"] = [am_ref_min[1], am_ref_max[1]]
        if par["name"] == "AmRefSw":
            par["limits"] = [am_ref_min[2], am_ref_max[2]]

    return pars


def plot_xy(fig, i_sub, plot_shape, x, y, x_label, y_label, color_space, color_main,
            x_ticks=None, y_ticks=None, i_main=None, title=None, x_lim=None):
    """Plot x and y data of all simulations"""

    # Plot all simulations
    ax = fig.add_subplot(plot_shape[0], plot_shape[1], i_sub)
    ax.plot(x.T, y.T, linewidth=1.0, alpha=0.1, color=color_space)

    # Plot main trend, either average of all simulations or a specified simulation
    if i_main is None:
        ax.plot(np.mean(x, axis=0), np.mean(y, axis=0), linewidth=4.0, color=color_main)
    # elif i_main is "auto":
    #     # Find the curve that is closest to the median of all simulations
    #     y_median = np.mean(y, axis=0)
    #     dist = []
    #     for y_sim in x:
    #         dist.append(np.linalg.norm(y_sim - y_median))
    #     i_main = np.argmin(dist)
    #     ax.plot(x[i_main, :], y[i_main, :], linewidth=4, color=color_main)
    else:
        ax.plot(x[i_main, :], y[i_main, :], linewidth=4, color=color_main)

    # Set labels and ticks (if specified)
    ax.set(xlabel=x_label, ylabel=y_label)
    if x_ticks is not None:
        ax.set(xticks=x_ticks, xlim=(x_ticks[0], x_ticks[-1]))
    if y_ticks is not None:
        ax.set(yticks=y_ticks, ylim=(y_ticks[0], y_ticks[-1]))
    if x_lim is not None:
        ax.set(xlim=x_lim)

    if title is not None:
        ax.set_title(title)


def plot_xy_growth(y_time, y_data, y_std, y_sims, x_sims, y_label, color_space, color_main, save_dir, save_name,
                   x_ticks=None, y_lims=None, y_ticks=None, legend=False,
                   ci=True, sim_space=True, subfix="", show_fig=False):
    """Plot predicted model outcome of all simulations in the directory sim_dir and saves plots to exp_dir"""

    # Get limits and ticks if specified
    y_lim, y_tick = None, None
    if y_lims is not None:
        if y_label in y_lims.keys():
            y_lim = y_lims[y_label]
    if y_ticks is not None:
        if y_label in y_ticks.keys():
            y_tick = y_ticks[y_label]

    if legend == True:
        sim_label = "Simulated"
        obs_label = "Observed"
        legend_title = "95% CI"
    else:
        sim_label = None
        obs_label = None
        legend_title = None

    fig, ax = plt.subplots(figsize=(0.8*6, 0.8*4), linewidth=2.0)

    # Plot all simulations
    if sim_space:
        sns.lineplot(ax=ax, data=y_sims, x="time", y=y_label, hue="sim_id", estimator=None, alpha=0.1,
                     palette=sns.color_palette([color_space] * x_sims.shape[0]), linewidth=0.6, legend=False)

    # Plot data, add nan line to add label to the legend
    t = y_time[:len(y_data)]
    ax.errorbar(x=t, y=y_data, yerr=2 * abs(y_std), fmt='o', color="black", linewidth=3.0,
                markerfacecolor="white", markeredgewidth=2.0, capsize=5.0, capthick=2.0)
    plt.plot(np.nan, np.nan, color="black", label=obs_label, linewidth=3.0)

    # Plot mean simulation with or without confidence interval
    if ci:
        sns.lineplot(ax=ax, data=y_sims, x="time", y=y_label, errorbar=('sd', 2),
                     linewidth=3.0, color=color_main, label=sim_label)
    else:
        sns.lineplot(ax=ax, data=y_sims, x="time", y=y_label, errorbar=None,
                     linewidth=3.0, color=color_main, label=sim_label)

    # Set axes
    ax.set(xlabel="Time (days)")
    if x_ticks is not None:
        ax.set(xticks=x_ticks)
    if y_lim is not None:
        ax.set(ylim=y_lim)
    if y_tick is not None:
        ax.set(yticks=y_tick)
    finish_plots(fig, save_dir, str(save_name + "_" + y_label + subfix + ".pdf"), box_aspect=0.8,
                 legend_title=legend_title, show_fig=show_fig)


def finish_plots(fig, exp_dir, fig_name, box_aspect=1.0, legend_title=None, show_fig=False):
    """Finish plots by setting box aspect, removing frame, and exporting to exp_dir"""""
    # for rectangular plots, set box_aspect to 4/5
    ax = fig.get_axes()

    # Set things in all subplots
    for axi in ax:
        if legend_title is not None:
            axi.legend(frameon=False, title=legend_title)
        axi.set_box_aspect(box_aspect)
        for axis in ['top', 'bottom', 'left', 'right']:
            axi.spines[axis].set_linewidth(1.5)
        axi.tick_params(width=1.5)
    fig.tight_layout()

    # Create export directory if it does not exist
    exp_dir.mkdir(exist_ok=True, parents=True)
    fig.savefig(exp_dir / fig_name, bbox_inches='tight')
    if show_fig:
        plt.show()
    plt.close()


def get_aha_shortening(sim_dir):
    """Plot predicted model outcome of all simulations in the directory sim_dir and saves plots to exp_dir"""

    # Find all simulations in given directory
    sims, pars = get_model_files(sim_dir)
    n_sims = len(sims)

    # Load the first simulation to find the number of time increments
    with h5py.File(sims[0], "r") as f:
        n_inc = f["time"][:].size
        n_segments = f["lab_f"][:].shape[1]

    if n_segments < 16:
        raise ValueError("Number of segments is lower than 16")

    # Now load all files and store the strain data
    shortening = np.zeros((n_sims, n_inc, 16))
    for i_sim, sim in enumerate(sims):
        with h5py.File(sim, "r", locking=False) as f:
            lab = f["lab_f"][:, :16]
            outputs_names = f.attrs['outputs_names'][:]
            outputs = f['outputs'][:]
            i_ed = int(outputs[0, outputs_names == "IED"][0])

            # Change order so that i_ed is the first element
            lab = np.roll(lab, -i_ed, axis=0)

            # Calculate shortening
            shortening[i_sim, :, :] = (lab / lab[0, :] - 1) * 100

    return shortening


def translate_names(names, extra_translations=None, reverse=False):
    """Replace variable names with names that are better for plotting, use LaTeX, or are more descriptive"""

    translations = {# Inputs
                    "Ras": r"$R_\mathrm{as}$", "Rvs": r"$R_\mathrm{vs}$", "Rap": r"$R_\mathrm{ap}$",
                    "Rvp": r"$R_\mathrm{vp}$", "Rmvb": r"$R_\mathrm{mvb}$", "Rcp": r"$R_\mathrm{cp}$",
                    "Rcs": r"$R_\mathrm{cs}$", "Rav": r"$R_\mathrm{av}$", "Rtvb": r"$R_\mathrm{tvb}$",
                    "Rmvb_acute": r"$R_\mathrm{mvb}$",
                    "Cas": r"$C_\mathrm{as}$", "Cvs": r"$C_\mathrm{vs}$",
                    "Cap": r"$C_\mathrm{ap}$", "Cvp": r"$C_\mathrm{vp}$",
                    "AmRefLfw": r"$A_\mathrm{m,ref,lfw}$", "AmRefSw": r"$A_\mathrm{m,ref,rw}$",
                    "AmRefRfw": r"$A_\mathrm{m,ref,rfw}$", "AmRefLA": r"$A_\mathrm{m,ref,la}$",
                    "AmRefRA": r"$A_\mathrm{m,ref,ra}$",
                    "VLfw": r"$V_\mathrm{lfw}$", "VSw": r"$V_\mathrm{sw}$", "VRfw": r"$V_\mathrm{rfw}$",
                    "VLA": r"$V_\mathrm{la}$", "VRA": r"$V_\mathrm{ra}$",
                    "TAct": r"$T_\mathrm{act}$", "tad": r"$t_\mathrm{ad}$", "tr": r"$\tau_\mathrm{r}$",
                    "td": r"$\tau_\mathrm{d}$",
                    "avd": r"$AVD$", "ivd_lv": r"$IVD_\mathrm{LV}$", "ivd_rv": r"$IVD_\mathrm{RV}$",
                    "c1": r"$c_1$", "c3": r"$c_3$", "c4": r"$c_4$",
                    "SAct": r"$S_\mathrm{act}$", "SfAct": r"$S_\mathrm{f,act}$",
                    "SBV": r"$SBV$",
                    "HR": r"$HR$",
                    "Tau": r"$\tau$", "MaxP": r"$P_\mathrm{max}$", "MaxdP": r"$\mathrm{d}P/\mathrm{d}t_\mathrm{max}$",
                    "c1_a": r"$c_{1,a}$", "c3_a": r"$c_{3,a}$", "c4_a": r"$c_{4,a}$",
                    "SfAct_a": r"$S_\mathrm{f,act,a}$",
                    "tad_a": r"$t_\mathrm{ad,a}$", "tr_a": r"$\tau_\mathrm{r,a}$", "td_a": r"$\tau_\mathrm{d,a}$",
                    "c1_p": r"$c_{1,p}$", "c3_p": r"$c_{3,p}$", "c4_p": r"$c_{4,p}$",
                    "WTh_p": r"$Wth_\mathrm{p}$",

                    # Outputs
                    "LVMaxP": r"$\mathrm{LVP}_\mathrm{max}$", "RVMaxP": r"$\mathrm{RVP}_\mathrm{max}$",
                    "LVMaxdP": r"$\mathrm{LVdP}_\mathrm{max}$", "RVMaxdP": r"$\mathrm{RVdP}_\mathrm{max}$",
                    "LVMindP": r"$\mathrm{LVdP}_\mathrm{min}$", "RVMindP": r"$\mathrm{RVdP}_\mathrm{min}$",
                    "EDWthLfw": r"$\mathrm{EDWth}_\mathrm{lfw}$", "EDWthSw": r"$\mathrm{EDWth}_\mathrm{sw}$",
                    "EDWthRfw": r"$\mathrm{EDWth}_\mathrm{rfw}$", "EDWthLA": r"$\mathrm{EDWth}_\mathrm{la}$",
                    "EDWthRA": r"$\mathrm{EDWth}_\mathrm{ra}$",
                    "ESWthLfw": r"$\mathrm{ESWth}_\mathrm{lfw}$", "ESWthSw": r"$\mathrm{ESWth}_\mathrm{sw}$",
                    "ESWthRfw": r"$\mathrm{ESWth}_\mathrm{rfw}$", "ESWthLA": r"$\mathrm{ESWth}_\mathrm{la}$",
                    "ESWthRA": r"$\mathrm{ESWth}_\mathrm{ra}$",
                    "Dlvsw": r"$D_\mathrm{lfw,sw}$", "Drvsw": r"$D_\mathrm{rfw,sw}$", "Drvi": r"$D_\mathrm{rvi}$",
                    "EDStretchLfw": r"$ED\lambda_\mathrm{lfw}$", "EDStretchSw": r"$ED\lambda_\mathrm{sw}$",
                    "EDStretchRfw": r"$ED\lambda_\mathrm{rfw}$", "EDStretchLA": r"$ED\lambda_\mathrm{la}$",
                    "EDStretchRA": r"$ED\lambda_\mathrm{ra}$",

                    }
    if extra_translations is not None:
        translations.update(extra_translations)

    if reverse:
        translations = {v: k for k, v in translations.items()}

    return [translations[name] if name in translations else name for name in names]


def set_pars(par_names):
    """Get the ranges of all parameters"""

    # Loop through parameters to get their default ranges
    pars = {}

    # Inputs
    for par_name in par_names:
        if par_name == "Ras":
            pars[par_name] = {"limits": [0.25, 5.0]}
        elif par_name == "SBV":
            pars[par_name] = {"limits": [1000, 4000]}
        elif par_name == "tad":
            pars[par_name] = {"limits": [100, 500]}
        elif par_name == "tr":
            pars[par_name] = {"limits": [0.1, 0.5]}
        elif par_name == "td":
            pars[par_name] = {"limits": [0.1, 0.5]}
        elif par_name == "SfAct":
            pars[par_name] = {"limits": [0.05, 0.30]}
        elif par_name == "AmRefLfw":
            pars[par_name] = {"limits": [5e3, 20e3]}
        elif par_name == "AmRefSw":
            pars[par_name] = {"limits": [2e3, 20e3]}
        elif par_name == "AmRefRfw":
            pars[par_name] = {"limits": [5e3, 20e3]}
        elif par_name == "VLfw":
            pars[par_name] = {"limits": [50e3, 300e3]}
        elif par_name == "VSw":
            pars[par_name] = {"limits": [10e3, 100e3]}
        elif par_name == "VRfw":
            pars[par_name] = {"limits": [10e3, 100e3]}
        else:
            raise ValueError(f"Parameter {par_name} not found")

    return pars
