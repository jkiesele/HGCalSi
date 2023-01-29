#!/usr/bin/env python3

from argparse import ArgumentParser

import pandas as pd

import styles

styles.setstyles(16)
import os
import pickle 

import matplotlib.pyplot as plt


globalpath=os.getenv("DATAPATH")+'/'
processed_data=os.getenv("DATAOUTPATH")+'/'

#get the data

def getDF(dirname):
    with open(processed_data+dirname+'/depl_dataframe.pkl', 'rb') as f:
        return pickle.load(f)
    
    
    
df = getDF('3008_LL_2D_diode_big_no_ann')
df_ser = df[df['mode'] == 'cs']
df_par = df[df['mode'] == 'cp']
#make plot
plt.plot(df_ser['frequency'],-df_ser['depletion_nominal'], label = 'serial')
plt.plot(df_par['frequency'],-df_par['depletion_nominal'], label = 'parallel')
plt.legend()
plt.xlabel("Frequenzy [Hz]")
plt.ylabel("Depletion voltage [V]")
plt.xscale('log')
plt.tight_layout()
plt.show()