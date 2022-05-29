#!/usr/bin/python3

from argparse import ArgumentParser
from fileIO import fileReader

parser = ArgumentParser()
parser.add_argument('inputDir')
args = parser.parse_args()

import glob

fileReader=fileReader(mode="CV",return_freq=True)

for infile in glob.glob(args.inputDir+'/*cv'):
    _,_,_, freq = fileReader.read(infile)
    print(infile, 'freq', freq)