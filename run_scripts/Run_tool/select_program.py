#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 14:18:37 2021

@author: tianmu
"""

import os
import subprocess

class Program_to_run:
    home_dir0 = os.getcwd()
    home_dir = os.getcwd()
    abs_dir = os.getcwd()
    
    def __init__(self,address):
        self.abs_dir = address
    
    def run(self):
        print(self.abs_dir)
        popen = subprocess.Popen(self.abs_dir, stdout=subprocess.PIPE)
        print("Simulation started...")
        err = popen.wait()
        output = popen.stdout.read()
        print(output.decode("utf-8"))