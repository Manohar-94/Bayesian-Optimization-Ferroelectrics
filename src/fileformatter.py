import glob
import os
import sys
import xlwings as xw
import pandas as pd

def get_dev(type, file):
    """
    get_dev(type, file) finds the device names given [file] and type of 
    file [type].
    """
    file = file.split(type,1)[1]
    file = file[:file.rfind('.')]
    idx = [x.isdigit() for x in file].index(True)
    file = file[idx:]
    trim_idx = file.rfind("_after")
    file = file[:trim_idx] if trim_idx!=-1 else file
    return file

def file_combine(dir):
    """
    file_combine(dir) combines files with same device and different cycles
    into single file with multiple sheets:
        i.e. KHM0ID_subID_PUND_devspecs_3cycles_1e6_1e7cycles.xls
    """
    files = dir + '/*.xls'
    files_exp = glob.glob(files, recursive = True)

    # Set-up conditions for file list iteration
    cond_end = lambda file: "endurance" in file and "PV" not in file
    cond0_PV = lambda file: "PV" in file
    cond1_PV = lambda file: cond0_PV(file) and "after_endurance" not in file 
    cond2_PV = lambda file: cond0_PV(file) and "1PUND" not in file
    cond_PV = lambda file: cond1_PV(file) and cond2_PV(file)
    cond_PUND = lambda file: "PUND" in file and "PV" not in file

    # Make lists of all files of endurance, PV, and PUND 
    end_files = [file for file in files_exp if cond_end(file)]
    PV_files = [file for file in files_exp if cond_PV(file)]
    PUND_files = [file for file in files_exp if cond_PUND(file)]

    # Rename PV files with endurance as 1e7 
    for file in PV_files: 
        if "endurance" in file:
            sub_end = file[:file.rfind("endurance")]
            sub_1e7 = sub_end + "1e7cycles.xls"
            os.rename(file, sub_1e7)
            PV_files.append(sub_1e7)
            PV_files.remove(file)

    files_col = [PV_files, PUND_files, end_files]
    devs = set([get_dev("PV_", PV_file)for PV_file in PV_files])
    make_dict = lambda: dict.fromkeys(devs, [])
    PV_dict, PUND_dict, end_dict = make_dict(), make_dict(), make_dict()
    file_dicts = [PV_dict, PUND_dict, end_dict]

    for f_d, fs in list(zip(file_dicts, files_col)): file_dict_maker(f_d, fs, devs)

    # File interation and combine
    app = xw.App(visible=False)
    for type, f_d in [('PV', PV_dict), ('end', end_dict), ('PUND', PUND_dict)]: 
        file_combiner(type, f_d)
    app.quit()

def file_combiner(type, f_d):
    """
    file_combiner(type, f_d) performs actual combining of files with same device 
    as helper for file_combine(dir, idx). 
    """
    get_iter = lambda fn:fn[fn.rfind('_'):]
    cut_ext = lambda fn, cut_word:fn[:fn.rfind(cut_word)]
    for dev, fn_list in f_d.items():
        fn_to = fn_list[-1]
        iter = range(len(fn_list)-1)
        if type != "PUND":
            fn_to = fn_list[0]
            fn_temp = cut_ext(fn_to, '.') + "_after_3cycles.xls"
            os.rename(fn_to, fn_temp)
            fn_to = fn_temp
            iter = range(1, len(fn_list))
        for fn_idx in iter:
            fn_from = fn_list[fn_idx]
            wb_to = xw.Book(fn_to)
            wb_from = xw.Book(fn_from)
            ws_from = wb_from.sheets['Data']
            ws_from.copy(after=wb_to.sheets[-1])
            idx = 2+fn_idx if type != "PUND" else 1+fn_idx
            wb_to.sheets[-1].name = 'Append' + str(idx) 
            wb_from.close()
            os.remove(fn_from)
            wb_to.save()
            wb_to.close()
            fn_temp = cut_ext(fn_to, 'cycles')+ get_iter(fn_from)
            os.rename(fn_to, fn_temp)
            fn_to = fn_temp

def file_dict_maker(file_dict, files, devs):
    """
    file_dict_maker(file_dict, files, devs) fills in file_dicts with [devs] as keys
    and [files] with the appropriate device as values. 
    """
    for dev in devs:
        temp = []
        for file in files:
            if dev in file: 
                subfile = file[file.rindex(dev)+len(dev):]
                if not subfile[0].isdigit():
                    temp.append(file)
        if len(temp)>=2: file_dict[dev] = sorted(temp)
        elif len(temp)<2: file_dict.pop(dev) 

def main(dir, sampID):
    """
    main(dir, subID operates as the main caller function to read in all raw data 
    in various subdirectories and write out processed df data.
    """
    subfolders = [f.path for f in os.scandir(dir) if f.is_dir() and sampID in f.path]
    for subfolder in subfolders: file_combine(subfolder)

# Update with file path on your local device 
dir = '/Users/valenetjong/Bayesian-Optimization-Ferroelectrics/data/'
main(dir, "KHM005")