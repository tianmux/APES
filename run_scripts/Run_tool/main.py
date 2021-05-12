#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 14:23:25 2021

@author: tianmu
"""

import select_program as sp
import input_class
import os



if __name__ == '__main__':
    cwd = os.getcwd()
    fn_input = 'input.txt'
    
    address = "/home/tianmu/Documents/Code/APES_dev/APESAVX2"
    
    prog_to_run = sp.Program_to_run(address)
    input1 = input_class.Input(fn_input)
    input1.print_out()
    
    # ask for the parameters to scan:
    input1.wait_for_choice()
    input1.print_out_scanned()
    
        
    #prog_to_run.run()
