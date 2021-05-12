#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 16:07:56 2021

@author: tianmu
"""

import numpy as np

class Input:
    variables = {}
    scaned = np.zeros(0)
    
    def __init__(self,fn):
        print("Reading input file: ",fn)
        with open(fn) as inputfile:
            for line in inputfile:
                temp = line.split()
                self.variables[temp[0]] = np.array([float(temp[i+1]) for i in range(len(temp)-1)])
    def print_out(self):
        for i in self.variables:
            print(i," : ",self.variables[i])
    def print_out_scanned(self):
        for i in range(len(self.scaned)):
            print(self.variables[self.scaned[i]])
    
    
    def wait_for_choice(self):
        print("Please select the parameter that is going to be scanned:")
        index = 0
        for i in self.variables:
            print(index, ". ",i)
            index+=1
        print(index, ". ", "Next")
        cmd = ""
        while cmd!=str(index):
            cmd=input("Your choice: ")
            if float(cmd)<(index):
                np.append(self.scaned,float(cmd))
                