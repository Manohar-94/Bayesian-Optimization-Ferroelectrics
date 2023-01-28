import sys, glob, os, re, shutil
import xlwings as xw
import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter1d as gf
from scipy.interpolate import UnivariateSpline as us
from plotter import prettyplot, vis_iv, vis_pv, vis_pund
import matplotlib.pyplot as plt

def df_try_except(success, param, *exceptions):
    try: return success(param)
    except exceptions or Exception: return pd.DataFrame()

def get_devicelen(device): 
    idx = [x.isdigit() for x in device].index(True)
    return int(device[idx:device.find('um')])

def scale_pr(device, pr): return pr/(get_devicelen(device)**2*10**(-14))

def calc_metric(qsw, u, d): return 2*qsw/(u + np.abs(d))

def get_dev(type, file):
    """
    get_dev(type, file) finds the device names given [file] and type of 
    file [type].
    """
    file = file.split(type,1)[1]
    file = file[:file.rfind('.')]
    idx = [x.isdigit() for x in file].index(True)
    file = file[idx:]
    if file.rfind('_endurance') != -1: file = file[:file.rfind('_endurance')]
    if file.rfind('_after') != -1: file = file[:file.rfind('_after')]
    return file

def process_PV(df, file, print_flag_IV=False, print_flag_PV=False):
    """
    process_PV(df, file, print_flag_IV, print_flag_PV) calculates the 
    Pr, Vc, and Imprint values for the P-V and I-V loops for a given [file] 
    and returns an updated dataframe; if print_flag = True, 
    runs plot production code.
    """
    device = get_dev("PV_", file)  
    dev_row = df[df["device"] == device].index.to_numpy()[0]
    
    try: 
        PV_df = pd.read_excel(file, sheet_name='Append2', usecols=['Vforce','Charge'])
        IV_df = pd.read_excel(file, sheet_name='Append2', usecols=['Vforce','Imeas'])
    except:
        print("No Append2")
        return df
    pv_data, iv_data = np.array(PV_df), np.array(IV_df)

    # Unit conversion
    pv_data[:,1] = scale_pr(device, pv_data[:,1])
    
    # Shift data
    max, min = np.amax(pv_data,axis=0)[1],np.amin(pv_data,axis=0)[1]
    pv_data[:,1] -= np.mean((max, min))

    # Find Pr, Vc and imprint for P-V
    dir_Δ = 300 # 300 marks data point of direction change
    Pr = pv_data[np.argwhere(pv_data[dir_Δ:,0] < 0)[0]+dir_Δ,1][0]
    Vc_neg = pv_data[np.argwhere(pv_data[dir_Δ:,1] < 0)[0]+dir_Δ,0][0]
    Vc_pos = pv_data[np.argwhere(pv_data[50:,1] > 0)[0]+50,0][0]
    Vc = np.mean((Vc_pos, np.abs(Vc_neg)))
    Imprint = np.mean((Vc_pos,Vc_neg))

    df.at[dev_row,"Pr (mC/cm^2)"] = Pr
    df.at[dev_row,"Vc (P-V)"] = Vc
    df.at[dev_row,"Imprint (P-V)"] = Imprint

    # Find Vc and imprint for I-V
    iv_filt = gf(iv_data[:,1],2)
    arg_vc_pos, arg_vc_neg = np.argmax(iv_filt), np.argmin(iv_filt)
    Vc_pos_iv, Imeas_Vc_pos = iv_data[:,0][arg_vc_pos], iv_filt[arg_vc_pos]
    Vc_neg_iv, Imeas_Vc_neg = iv_data[:,0][arg_vc_neg], iv_filt[arg_vc_neg]
    df.at[dev_row, "Vc (I-V)"] = np.mean((Vc_pos_iv, np.abs(Vc_neg_iv)))
    df.at[dev_row, "Imprint (I-V)"] = np.mean((Vc_pos_iv, Vc_neg_iv))

    # Produce P-V plots
    if print_flag_PV:
        pv_neg = pv_data[np.argwhere(pv_data[:,0]==Vc_neg)[0][0]][1]
        pv_neg_tup = (Vc_neg, pv_neg)
        pv_pos = pv_data[np.argwhere(pv_data[:,0]==Vc_pos)[0][0]][1]
        pv_pos_tup = (Vc_pos, pv_pos)
        vis_pv(pv_data, pv_neg_tup, pv_pos_tup, device)

    # Produce I-V plots
    if print_flag_IV: 
        pos_tup, neg_tup = (Vc_pos_iv, Imeas_Vc_pos), (Vc_neg_iv, Imeas_Vc_neg)
        vis_iv(iv_data, iv_filt, pos_tup, neg_tup, device)  
    return df

def process_PUND(df, file, sheet_name, print_flag_PUND=False):  
    """ 
    process_PUND(df, file, sheet_name, print_flag_PUND) finds the Vc and Imprint 
    for max, 10^6, 10^7 cycles. If print_flag_PUND=True, PUND plots are generated.
    """
    device = get_dev("PUND_", file)  
    try: 
        dev_row = df[df["device"] == device].index.to_numpy()[0]
        PUND_df = pd.read_excel(file, sheet_name=sheet_name, usecols=['t','V'])
        IV_df = pd.read_excel(file, sheet_name=sheet_name, usecols=['t','I'])
        Pr_df = pd.read_excel(file, sheet_name=sheet_name, usecols=['U','D','Qsw'])
    except:
        print("No Data")
        return df

    PUND_df, IV_df = PUND_df[['t','V']], IV_df[['t','I']]
    pund_data, iv_data = np.array(PUND_df), np.array(IV_df)
    pr_dat = np.array(Pr_df)[0]
    u,d,qsw = pr_dat[0], pr_dat[1], pr_dat[-1]

    # Scale by ramp rate (alpha) + ratios of maxes for easy viewing
    t_data = pund_data[:,0] 
    spline_fun = us(t_data,pund_data[:,1],s=0,k=4)
    spline_1d = spline_fun.derivative(n=1)
    filt_iv, filt_spline = gf(iv_data[:,1],3), gf(spline_1d(t_data), 3)
    alpha = np.polyfit(filt_spline, filt_iv, deg=1)[1]
    iv_data[:,1] = filt_iv/np.abs(alpha)
    
    if print_flag_PUND:
        pund_max, iv_max = np.amax(pund_data[:,1]), np.amax(iv_data[:,1])
        iv_data[:,1]*=(pund_max/iv_max)
        vis_pund(pund_data, iv_data, device, sheet_name)
    
    P_idx = np.argmax(iv_data[:,1][:int(len(iv_data)/4)])
    N_idx = np.argmin(iv_data[:,1][int(len(iv_data)/2):int(3*len(iv_data)/4)])

    # Corresponds to 3 cycle files
    if sheet_name == 'Data':
        df.at[dev_row, "2 Qsw/(U+|D|)"] = calc_metric(qsw, u, d)

    # Corresponds to 10^6 cycle files
    elif sheet_name == 'Append1':
        PUND_helper("10^6", df, pund_data, P_idx, N_idx, dev_row, device, qsw, u, d)
    
    # Corresponds to 10^7 cycle files
    elif sheet_name == 'Append2':
        PUND_helper("10^7", df, pund_data, P_idx, N_idx, dev_row, device, qsw, u, d)
    return df

def PUND_helper(iter, df, pund_data, P_idx, N_idx, dev_row, device, qsw, u, d):
    """
    PUND_helper(iter, df, pund_data, P_idx, N_idx, dev_row, device, qsw, u, d)
    is a helper function for process_PUND that carries out df writing for shared
    rows for different [iter]'s.
    """
    Vc_pos, Vc_neg = pund_data[:,1][P_idx], pund_data[:,1][N_idx]
    df.at[dev_row, iter+" Vc"] = np.mean((Vc_pos, np.abs(Vc_neg)))
    df.at[dev_row, iter+" Imprint"] = np.mean((Vc_pos, Vc_neg))
    df.at[dev_row, iter+" Pr (mC/cm^2)"] = 0.5 * scale_pr(device, qsw)
    df.at[dev_row, iter+" 2 Qsw/(U+|D|)"] = calc_metric(qsw, u, d)

def process_endurance(df, file):
    """
    process_endurance(df, file,sheet_name) finds the endurance, max Pr, 
    10^6 Pr, of a given endurance [file] and returns an updated dataframe.
    """
    device = get_dev("endurance_", file)
    dev_row = df[df["device"] == device].index.to_numpy()[0]
    read_x = lambda sheet_name: pd.read_excel(file, sheet_name=sheet_name, 
                                usecols=['iteration','P','Qsw'])
    PV_df, i, sheet_names = pd.DataFrame(), 0, ["Append4", "Append3", "Append2", "Append1", "Data"]
    while PV_df.empty: 
        PV_df = df_try_except(read_x, sheet_names[i+1], ValueError)
        i+=1
    data = np.array(PV_df)

    # Find iteration before occurrence of breakdown
    breakdown = find_breakdown(data)

    # Write endurance based on breakdown
    df.at[dev_row,"endurance"] = data[breakdown -1][0] if breakdown!=0 else 0

    # Find max Pr (before break)
    write_max_Pr(data, df, device, breakdown, dev_row)
    return df

def find_breakdown(data): return np.argmax(data[:,1]>=1e-9)

def write_max_Pr(data, df, device, breakdown, dev_row):
    """
    write_max_Pr(data, df, device, breakdown, dev_row) finds and writes max_Pr
    value to [df].
    """
    dat_bef_brk = data[:breakdown] if breakdown!=0 else data
    Q_sw = np.amax(dat_bef_brk, axis=0)[2]
    Pr_max = 0.5 * Q_sw
    col_name = "max Pr (mC/cm^2)"
    df.at[dev_row,col_name] = scale_pr(device, Pr_max)

def init_df(files_exp):
    """
    init_df(files_exp) initializes a dataframe with all device names in current
    subdirectory whose filenames are in [files_exp].  
    """
    end_files = [file for file in files_exp if "endurance" in file and "PV" not in file]
    PV_files = [file for file in files_exp if "PV" in file]

    # If missing endurance files, use PV files to extract device names
    if len(end_files) < len(PV_files):
        row_names = [file.split("PV_",1)[1] for file in PV_files]
        row_names = [file[:file.rfind('.')] for file in row_names]
        row_names = [file[re.search(r"\d", file).start():] for file in row_names]
        row_names = [file[:file.rfind('_after')] if "after" in file else 
        file for file in row_names]
        n_devices = len(PV_files)
    else:
        n_devices = len(end_files)
        row_names = [file.split("endurance_",1)[1] for file in end_files]
        row_names = [file[:file.rfind('.')] for file in row_names]
    n_rows = 17
    dat = np.zeros((n_devices, n_rows))
    col_names = ["device", "Pr (mC/cm^2)", "2 Qsw/(U+|D|)", "Vc (P-V)", 
    "Imprint (P-V)", "Vc (I-V)", "Imprint (I-V)", "endurance", 
    "10^6 Pr (mC/cm^2)", "10^6 2 Qsw/(U+|D|)", "10^6 Vc", "10^6 Imprint", 
    "10^7 Pr (mC/cm^2)",  "10^7 2 Qsw/(U+|D|)", "10^7 Vc", "10^7 Imprint",
    "max Pr (mC/cm^2)"] 
    df = pd.DataFrame(dat, columns=col_names)
    df['device'] = row_names
    return df

def read_file(dir, idx):
    """
    read_file(dir, idx) reads all files in subdirectory specified by [dir]
    and [idx] and returns relevant information, written to a pandas df.
    """
    subdir = dir + str(idx)
    files = subdir + '/*.xls'
    files_exp = glob.glob(files, recursive = True)
    df = init_df(files_exp)

    for filename in files_exp:
        if 'PUND' in filename:
            if '1e6' in filename:
                sheet_name = 'Append1'
                df = process_PUND(df, filename, sheet_name)
            if '1e7' in filename:
                sheet_name = 'Append2'
                df = process_PUND(df, filename, sheet_name)
            df = process_PUND(df, filename, 'Data')
        elif 'PV' in filename:
            df = process_PV(df, filename)
        elif 'endurance' in filename:
            df = process_endurance(df, filename)
    return df

def file_combine(dir, idx):
    """
    file_combine(files_exp) combines files with same device and different cycles
    into single file with multiple sheets:
        i.e. KHM0ID_subID_PUND_devspecs_3cycles_1e6_1e7cycles.xls
    """
    subdir = dir + str(idx)
    files = subdir + '/*.xls'
    files_exp = glob.glob(files, recursive = True)

    end_files = [file for file in files_exp if "endurance" in file and "PV" not in file]
    PV_files = [file for file in files_exp if "PV" in file]
    PUND_files = [file for file in files_exp if "PUND" in file]
    files_col = [PV_files, PUND_files, end_files]
    
    devs = set([get_dev("PV_", PV_file)for PV_file in PV_files])
    make_dict = lambda: dict.fromkeys(devs, [])
    PV_dict, PUND_dict, end_dict = make_dict(), make_dict(), make_dict()
    file_dicts = [PV_dict, PUND_dict, end_dict]
    
    for f_d, fs in list(zip(file_dicts, files_col)): file_dict_maker(f_d, fs, devs)
    get_iter = lambda fn:fn[fn.rfind('_'):]
    cut_ext = lambda fn, cut_word:fn[:fn.rfind(cut_word)]

    app = xw.App(visible=False)
    for f_d in [PV_dict, end_dict]: 
        for dev, fn_list in f_d.items():
            fn_to = fn_list[0]
            fn_temp = cut_ext(fn_to, '.') + "_after_3cycles.xls"
            os.rename(fn_to, fn_temp)
            fn_to = fn_temp
            wb_to = xw.Book(fn_to)
            for fn_idx in range(1, len(fn_list)):
                fn_from = fn_list[fn_idx]
                wb_from = xw.Book(fn_from)
                ws_from = wb_from.sheets['Data']
                ws_from.copy(after=wb_to.sheets[-1])
                wb_to.sheets[-1].name = 'Append' + str(2+fn_idx)
                wb_from.close()
                os.remove(fn_from)
                wb_to.save()
                wb_to.close()
                fn_temp = cut_ext(fn_to, 'cycles')+ get_iter(fn_from)
                os.rename(fn_to, fn_temp)
                fn_to = fn_temp
        
    for f_d in [PUND_dict]:
        for dev, fn_list in f_d.items():
            fn_to = fn_list[-1]
            for fn_idx in range(len(fn_list)-1):
                wb_to = xw.Book(fn_to)
                fn_from = fn_list[fn_idx]
                wb_from = xw.Book(fn_from)
                ws_from = wb_from.sheets['Data']
                ws_from.copy(after=wb_to.sheets[-1])
                wb_to.sheets[-1].name = 'Append' + str(1+fn_idx)
                wb_from.close()
                os.remove(fn_from)
                wb_to.save()
                wb_to.close()
                fn_temp = cut_ext(fn_to, 'cycles')+ get_iter(fn_from)
                os.rename(fn_to, fn_temp)
                fn_to = fn_temp
    app.quit()

def file_dict_maker(file_dict, files, devs):
    for dev in devs:
        temp = []
        for file in files:
            if dev in file: temp.append(file)
        if len(temp)>=2: file_dict[dev] = sorted(temp)
        elif len(temp)<2: file_dict.pop(dev) 

def main(dir, samp_ID, subdir_arr):
    """
    main(dir, subID, subdir_arr) operates as the main caller function to 
    read in all raw data in various subdirectories and write out processed df data.
    """
    subdir = range(1, subdir_arr[0]+1) if len(subdir_arr) == 1 else subdir_arr
    outdir = "/processed" + samp_ID
    out_dir = dir[:dir.rfind("/")] + outdir 
    if not os.path.exists(out_dir): os.mkdir(out_dir)

    for idx in subdir:
        # file_combine(dir, idx)
        df = read_file(dir, idx)
        df.to_csv(out_dir +  "/" + str(idx)+ ".csv") 

# Update with file path on your local device 
prettyplot()
dir = '/Users/valenetjong/Bayesian-Optimization-Ferroelectrics/data/KHM006_'
main(dir, "006", [6,7,8,11,13,14,15,17,18,19,20,21])