# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 14:49:03 2019

@author: alexaandrews16
"""

import utils

def main():
    table = utils.read_table("auto-data-clean.txt")
    for row in table:
        del row[-2]
    
    utils.write_table("auto-data-no-names.txt", table)
    
    
main()