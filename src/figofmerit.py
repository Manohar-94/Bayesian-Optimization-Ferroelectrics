import glob
import pandas as pd
import xlwings as xw

def read_joulemeter(in_file="data/MKS_Ophir_joulemeter_readings.xlsx", 
                    out_file="data/Bolometer_readings_PulseForge.xlsx"):
    """
    [read_joulemeter(in_file, out_file)] reads in_file and out_file and writes
    energy density values from in_file to respective column in out_file. 
    """
    df = pd.read_excel(in_file, sheet_name="Sheet2")
    df2 = pd.read_excel(out_file, sheet_name="Combined")
    cond1 = lambda i,j: sheet.range((i,3)).value == df.iloc[j-2, 0] # voltage
    cond2 = lambda i,j: sheet.range((i,4)).value == df.iloc[j-2, 1] # time
    book = xw.Book(out_file) 
    sheet = book.sheets[1]              

    # Write energy density  
    for i in range(2, len(df2)): 
        for j in range(2, len(df)):     
            if cond1 and cond2: sheet.range((i,7)).value = df.iloc[j-2, 2]  

def write_fig_of_merit(ID, out_file="Bolometer_readings_PulseForge.xlsx"):
    """
    [write_fig_of_merit(ID, out_file)] writes figure of merit for each [ID] in 
    the respective column.
    """
    dir = '/Users/valenetjong/Bayesian-Optimization-Ferroelectrics/data/'
    subdir = "processed" + ID 
    files = dir + subdir + '/*.csv'
    file_lst = glob.glob(files, recursive = True)

    df = pd.read_excel(dir+out_file, sheet_name="Combined")

    for file in file_lst:
        sub_ID = file[file.rfind('/')+1:file.rfind('.')]
        df = pd.read_csv(file)
        fig_merit_1e6 = df['10^6 2 Qsw/(U+|D|)'].max()
        row_max = df['10^6 2 Qsw/(U+|D|)'].idxmax()
        device = df['device'][row_max]
        fig_merit_3 = df['2 Qsw/(U+|D|)'][row_max]

        # Initialize 
        book = xw.Book(out_file) 
        sheet = book.sheets[1]  

        # Check sample + sub_ID
        cond = lambda i: sheet.range((i,1)).value == "KHM" + ID + "_" + sub_ID

        # Write energy density  
        for i in range(2, 79): 
            if cond(i):                 
                sheet.range((i,2)).value = device
                sheet.range((i,8)).value = fig_merit_3
                sheet.range((i,9)).value = fig_merit_1e6

write_fig_of_merit('010')

