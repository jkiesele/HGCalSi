#!/usr/bin/env python3

from argparse import ArgumentParser
import sklearn
from fitting import  readDepletionData, DepletionFitter
from tools import getDepletionVoltage, getDiodeAndTime
import matplotlib.pyplot as plt
from plotting import curvePlotter
import pandas as pd
import math
import numpy as np


import styles

styles.setstyles(16)
import os
import pickle 

parser = ArgumentParser('''
Script to determine the depletion voltage for a frequency scan input file and works almost the same way as fitDepletion.py.
It takes an input file (*.cf) and fits the depletion voltage (as in fitDepletion.py) 
 - for each frequency point
 - for parallel and serial mode (the best choice can also depend on frequency *and* fluence
The output is saved in the output directory (will be put under the usual processed data output: {dataout}).
The output format is a pickled pandas dataframe for easy handling afterwards.
'''.format(dataout=os.getenv("DATAOUTPATH")))
parser.add_argument('inputFile')
parser.add_argument('outputDir')
parser.add_argument('--var', type=float, default=10., help="variation in percent that is applied to the fit ranges")
args = parser.parse_args()

def invert(a):
    if a is not None:
        return -a
    return a
           
'''
credit for this part to Oliwia!
'''

#convertToCs(y,k,f)
def search_for_begin(datapath):
    with open(datapath, 'r') as infile:
        lines = infile.readlines()
        for l in lines:
            if l.find('BEGIN') != -1:
                ind = lines.index(l)
                break
        return ind + 1


# reading raw data from the file and putting them into a dataframe
def read_data_create_df(data_file, skip_header, skip_footer):
    data = np.genfromtxt(fname=data_file, skip_header=skip_header, skip_footer=skip_footer)
    new_df = pd.DataFrame(data, columns=['V_detector', 'Frequency', 'Capacitance', 'Conductivity', 'Bias', 'Current'])
    return new_df


# function for selecting CV data for one, selected, frequency
def select_CV_from_freq_scan(dataframe, frequency_selected):
    selected_CV = dataframe.loc[dataframe['Frequency'] == frequency_selected]  # choosing the rows with one frequency
    selected_CV.reset_index(drop=True, inplace=True)  # resetting the index of the dataframe
    return selected_CV


# function for converting parallel mode to serial mode. A new column with serial mode is added to the dataframe
def convert_cp_to_cs(single_CV, frequency):
    Gp = single_CV['Conductivity']
    omega = 2 * math.pi * frequency
    Cp = single_CV['Capacitance']
    Cs = (Gp ** 2 + omega ** 2 * Cp ** 2) / (omega ** 2 * Cp)
    single_CVs = single_CV.copy()
    single_CVs.loc[:, 'Serial_Capacitance'] = Cs
    return single_CVs






'''
fitting part
'''          

'''
Use two approaches:
- linear fit of two curves
- smoothed inflection point of two skewed curves
'''


globalpath=os.getenv("DATAPATH")+'/'
outpath=os.getenv("DATAOUTPATH")+'/'
os.system('mkdir -p '+outpath+args.outputDir)
outpath = outpath+args.outputDir+"/"

# read the data


begin_row = search_for_begin(args.inputFile)
meas_df = read_data_create_df(args.inputFile, skip_header=begin_row, skip_footer=1)
freq = meas_df.Frequency.unique()
variation = float(args.var)


#not implement the fit in this loop and save as data frame    
    
    

# done reading in

def find_ranges(plot_to_show):

    print('fit rising edge from to (positive)')
    print('fit constant from to')
    print('constant value (if any)')
    
    print('fit rising edge from to (positive)')
    print('fit constant from to')
    print('constant value (if any)')
    plot_to_show.show()
    high_end= - float(input())
    high_start= - float(input())
    low_end = - float(input())
    low_start = - float(input())
    
    
    
    const_cap=input()
    if not const_cap == "":
        const_cap=float(const_cap)
    else:
        const_cap=None
        
    return high_end, high_start, low_end, low_start, const_cap

def do_one_fit(singleCVs, capacitance : str , plotfilename : str):
    
    
    fig, ax = plt.subplots()
    
    def onpick(event):
        thisline = event.artist
        xdata = thisline.get_xdata()
        ydata = thisline.get_ydata()
        ind = event.ind
        print ('onpick points:', zip(xdata[ind], ydata[ind]))
        
    data = []    
    def onclick(event):
        if event.dblclick:
            if event.button == 1:
                print('picked',event.xdata, event.ydata)
                data.append(-event.xdata)
    
    x, y = singleCVs['V_detector'], 1 / (singleCVs[capacitance] ** 2)
    


    ax.plot(-x, y, '-')
    #plt.yscale('log')
    plt.title('Single CV for f = ' + str(f) + ' Hz')
    
    fig.canvas.mpl_connect('button_press_event', onclick)
    fig.canvas.mpl_connect('pick_event', onpick)
    
    plt.show()
    
    try:
        if len(data) != 4:
            high_end, high_start, low_end, low_start, const_cap = find_ranges(plt)
        else:
            high_end, high_start, low_end, low_start, const_cap = *data, None
    
        print(high_end, high_start, low_end, low_start, const_cap)
        df = DepletionFitter(x=np.array(x), 
                             y=np.array(y), 
                     const_cap=const_cap,
                     rising=1, constant=8,
                     debugfile=plotfilename+'_debug',
                     low_start  =low_start,
                            low_end = low_end,
                            high_start=high_start,
                            high_end =high_end,
                            varcut=variation,
                            interactive=True)
        
        v = df.getDepletionVoltage(debugplot=True,plotprefix=plotfilename, return_all_data=True)
        
        ret = input('are you happy? (y(Y)/n(N)')
        if not (ret == "y" or ret == "Y"):
            return None
        
        out = { k:v[k] for k in ['depletion_nominal','depletion_up','depletion_down']}
        out.update({
            'high_end':high_end, 
            'high_start':high_start, 
            'low_end':low_end, 
            'low_start':low_start, 
            'const_cap':const_cap})
        print(out)
        
        return out
    
    except Exception as e:
        print(e)
        print('an error occured, try again or quit with ctrl-c')
        return None
    
outdict = {
    'depletion_nominal': [],
    'depletion_up': [],
    'depletion_down': [],
    
    
    'high_end': [],
    'high_start': [],
    'low_end': [],
    'low_start': [],
    'const_cap': [],
    
    'frequency': [],
    'mode': []
    }


    
for f in freq:
    
    singleCV = select_CV_from_freq_scan(meas_df, f)
    print('CV data for f = ', f, ' Hz')
    singleCVs = convert_cp_to_cs(singleCV, f)
    #print(singleCVs['Serial_Capacitance'])
    
    v = None
    while v is None:
        v = do_one_fit(singleCVs, 'Serial_Capacitance',  outpath + '/cvfit_cs_'+str(f)+'.pdf')
    
    for k in v.keys():
        outdict[k].append(v[k])
    outdict['mode'].append("cs")
    outdict['frequency'].append(f)
    
    
    v = None
    while v is None:
        v = do_one_fit(singleCVs, 'Capacitance',  outpath + '/cvfit_cp_'+str(f)+'.pdf')
    
    for k in v.keys():
        outdict[k].append(v[k])
    outdict['mode'].append("cp")
    outdict['frequency'].append(f)
    
    df = pd.DataFrame.from_dict(outdict)
    df.to_pickle(outpath+'depl_dataframe.pkl')

print(df,'saved data in',outpath+'depl_dataframe.pkl')


