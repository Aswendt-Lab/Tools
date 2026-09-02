
"""
Created on Mon Nov 22 17:28:14 2021
@author: kalantaria
Description: This code will parse through every possible folder after a defined initial path,
looking for MR data files of any type. Then it will extract the wanted files 
and eliminiate the duplicates.
"""
import argparse
import os
import glob
#%% Command line interface
if __name__ == "__main__":
        
    parser = argparse.ArgumentParser(description='Helper script to upload data to gin')
    parser.add_argument('-i','--initial_path', help='initial path of the dataset')
    args = parser.parse_args()
    initial_path = args.initial_path
    
    print("Hello!")
    print('------------------------------------------------------------')
    print('Thank you for using our Code. For questions please contact us over:')
    print('aref.kalantari-sarcheshmeh@uk-koeln.de or markus.aswendt@uk-koeln.de')
    print('Lab: AG Neuroimaging and neuroengineering of experimental stroke University Hospital Cologne')
    print('Web: https://neurologie.uk-koeln.de/forschung/ag-neuroimaging-neuroengineering/')
    print('------------------------------------------------------------')

    #%% Parsing
    os.chdir(initial_path)
    dd = 0
    
    PathALL    = os.path.join(initial_path, "inputs/mri/proc_data/*/*")

    text_files = glob.glob(PathALL, recursive = True)
    kall = len(text_files)
    print(( 'Total number of '+ str(kall) + ' files/folders were found:'+'Parsing finished! '.upper()).upper())
    
    for PP in text_files:
        print('datalad save ' + PP + ' -m "inital save"')
        os.system('datalad save ' + PP + ' -m "inital save"')
        print('datalad push ' + PP + ' --to gin')
        os.system('datalad push ' + PP + ' --to gin')
        print('datalad drop ' + PP)
        os.system('datalad drop ' + PP)
        
    
    print('---------------------FINISHED---------------------------------------')
    print('Thank you for using our Code. For questions please contact us over:')
    print('aref.kalantari-sarcheshmeh@uk-koeln.de or markus.aswendt@uk-koeln.de')
    print('Lab: AG Neuroimaging and neuroengineering of experimental stroke University Hospital Cologne')
    print('------------------------------------------------------------')   