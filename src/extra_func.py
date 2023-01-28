import pandas as pd
import numpy as np
import os

def make_dummy_data(dir = "/Users/valenetjong/Bayesian-Optimization-Ferroelectrics/data/",
    src_file="KHM005_KHM006_quartz_HZO_samples.csv", 
    output_file_name='../data/KHM005_KHM006_quartz_HZO_samples2.csv'): 
    """
    make_dummy_data() reads in [src_file], containing Thickness (nm) and 
    Flash Time (msec) data, as well as fills in dummy Duty Cycle and 
    Num Pulses data to a pandas dataframe, which is returned. 
    A csv of this training data, with the four aforementioned input parameters, 
    is outputted to [output_file_name].
    """
    # Load data
    fe_data = pd.read_csv(dir + src_file, index_col=0)
    fe_data_len = len(fe_data['Thickness (nm)'])

    # Add duty cycle data
    duty_cycle_list = np.array([0.45, 0.55, 0.65])
    duty_cycles = np.random.choice(duty_cycle_list, size=fe_data_len)
    fe_data['Duty Cycle'] = duty_cycles

    # Add num pulses data
    num_pulses_list = np.array([15, 25])
    num_pulses = np.random.choice(num_pulses_list, size=fe_data_len)
    fe_data['Num Pulses'] = num_pulses

    # Rearrange columns
    cols = list(fe_data.columns.values) 
    idx = cols.index("Flash time (msec)")
    cols = cols[:idx+1] + cols[-2:] + cols[idx+1:-2]
    fe_data = fe_data[cols]

    # Write data back to csv
    os.makedirs('../data', exist_ok=True) 
    fe_data.to_csv('../data/KHM005_KHM006_quartz_HZO_samples2.csv') 

    # Load manipulated data
    fe_data = pd.read_csv(output_file_name, index_col=0)
    return fe_data

