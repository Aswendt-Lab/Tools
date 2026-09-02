#!/usr/bin/env python
# coding: utf-8

"""
Created on 07/12/2022

@author: Franziska Ziolkowski
AG Aswendt: Neuroimaging & Neuroengineering
Department of Neurology
University Hospital Cologne
"""

import glob
import os
import shutil
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--idfile", dest="idfile", default="", help="Path to ID file")
parser.add_argument("-t", "--idtype", dest="idtype", default="mouse", choices=["mouse", "video"], help="ID types mouse or video")
parser.add_argument("-s", "--start", dest="startstring", default="", help="Start path where files are currently located")
parser.add_argument("-d", "--destination", dest="destination", default="", help="Path to where files should be copied")
parser.add_argument("-l", "--lists", dest="savelists", default="", help="Path to where files containing copy information should be saved, default will be same as copy location")
parser.add_argument("-k", "--key", dest="key", default="Behavior", choices=["Behavior", "iwxdata"], help="Key of data type")
parser.add_argument("-e", "--end", dest="end", default="", help="File ending")
parser.add_argument("-p", "--time", dest="time", default="all", help="Gives option to copy only data for a specific time point (e.g. 'BL','P3',...)")
params = parser.parse_args()

#################################################################################
# check parameters

if not os.path.exists(params.idfile):
    sys.exit("Cannot find ID file.")
else: ID_file = params.idfile

if not params.idtype=='mouse' or params.idtype=='video':
    sys.exit("Specify ID type as 'mouse' or 'video'.")
else: ID_type = params.idtype

if not os.path.exists(params.startstring):
    sys.exit("Cannot find start path.")
else: start_string = params.startstring

if not os.path.exists(params.destination):
    sys.exit("Cannot find copy destination.")
else: out_dir = params.destination

if params.savelists == "":
    save_lists = params.destination
elif not os.path.exists(params.savelists):
    sys.exit("Cannot find directory where to save lists.")
else:
    save_lists = params.savelists

key = params.key # Bahavior

if params.key == "Behavior" and params.end=="":
    endings = ['avi', 'mp4']
elif params.key == "iwxdata" and params.end=="":
    endings = ['iwxdata', 'mat']
else:
    endings = [params.end]



###########################################################


print("Looking for files, this can take a while...")

ID_list = []
copy_list = []
not_found = []
new_names = []

# make list of all IDs to search for
with open(ID_file, "r") as infile:
    for line in infile:
        if ID_type == "mouse":
            line = line.strip()
            ID_list.append(line)
        if ID_type == "video":
            splitID = line.split("_")
            ID = splitID[0]+"_"+splitID[1]+"_"+splitID[2]+"_"+splitID[3]
            ID = ID.rstrip("_croppedDLC")
            ID_list.append(ID)
            ID_list.append(line)

# make list of all available file paths with given ending
all_video_files = []
if key == "Behavior" and params.time == "all":
    for end in endings:
        all_video_files.extend(glob.glob(start_string + "/**/Behavior/**/Baseline/**/*." + end, recursive=True))
        all_video_files.extend(glob.glob(start_string + "/**/Behavior/**/P**/**/*." + end, recursive=True))
elif key == "Behavior" and params.time != "all":
    for end in endings:
        all_video_files.extend(glob.glob(start_string + "/**/Behavior/**/" + params.time + "/**/*." + end, recursive=True))
elif key == "iwxdata" and params.time == "all":
    for end in endings:
        all_video_files.extend(glob.glob(start_string + "/**/EMG/**/*." + end, recursive=True))
elif key == "iwxdata" and params.time != "all":
    for end in endings:
        all_video_files.extend(glob.glob(start_string + "/**/EMG/**/"+params.time+"/**/*." + end, recursive=True))

print('A total of {} files were found with the correct ending.'.format(len(all_video_files)))

# make list of file paths for given IDs
for num, ID in enumerate(ID_list):
    found = False
    for video_path in all_video_files:
        if ID in video_path:
            copy_list.append(video_path)
            found = True
    if not found:
        not_found.append(ID) # IDs that weren't found are copied into this list

# check which files were found and which weren't
print('{} matching files were found.'.format(len(copy_list)))
print('{}/{} IDs were found.'.format(len(ID_list)-len(not_found), len(ID_list)))

# save lists containing files to copy, IDs that werent found, new file names, and all files found
outfile = os.path.join(save_lists, "copyfiles.txt")
with open(outfile, 'w') as out:
    for item in copy_list:
        out.write(item+'\n')

outfile = os.path.join(save_lists, "errorfiles.txt")
with open(outfile, 'w') as out:
    for item in not_found:
        out.write(item+'\n')


def splitall(rpath):
    allparts = []
    while 1:
        parts = os.path.split(rpath)
        if parts[0] == rpath:  # sentinel for absolute paths
            allparts.insert(0, parts[0])
            break
        elif parts[1] == rpath: # sentinel for relative paths
            allparts.insert(0, parts[1])
            break
        else:
            rpath = parts[0]
            allparts.insert(0, parts[1])
    return allparts

size=0
outfile = os.path.join(save_lists, "newnames.txt")
with open(outfile, 'w') as out:
    for copy_path in copy_list:
        split_path = splitall(copy_path)
        for end in endings:
            if split_path[-1].endswith(end):
                file_name = split_path[-1] [:-(len(end)+1)]
                suffix = "."+end

        new = file_name + '-'+ split_path[-5]+ '-'+ split_path[-4]+ '-'+ split_path[-3]+ '-'+ split_path[-2] + suffix
        new_names.append(new)
        out.write(new+'\n')
        size += os.path.getsize(copy_path)

outfile = os.path.join(save_lists, "allfiles.txt")
with open(outfile, 'w') as out:
    for item in all_video_files:
        out.write(item+'\n')

while True:
    print("Check out the lists saved in {}. \n {} files will be copied from \n {} \n to \n {}".format(save_lists, len(copy_list),  start_string, out_dir))
    print("{} GB of space are reqiured.".format(size/1000000000))
    check = input("If you want to proceed to copy the files type 'yes', to terminate type 'no': ")
    if check == 'yes' or check == 'Yes' or check == 'YES':
        # copy in new directory
        print("Copying files...")
        running_space = 0
        check5=False
        check10=False
        check20=False
        check30=False
        check40=False
        check50=False
        check60=False
        check70=False
        check80=False
        check90=False
        check100=False
        for (i, path) in enumerate(copy_list):
            # command to copy files
            shutil.copyfile(path, os.path.join(out_dir, new_names[i]))

            # progress information
            running_space += os.path.getsize(path)
            if running_space/size >= 0.05 and check5==False:
                print("5 % of data, {}/{} files have been copied".format(i, len(copy_list)))
                check5 = True
            elif running_space/size >= 0.1 and check10==False:
                print("10 % of data, {}/{} files have been copied".format(i, len(copy_list)))
                check10 = True
            elif running_space/size >= 0.2 and check20==False:
                print("20 % of data, {}/{} files have been copied".format(i, len(copy_list)))
                check20 = True
            elif running_space/size >= 0.3 and check30==False:
                print("30 % of data, {}/{} files have been copied".format(i, len(copy_list)))
                check30 = True
            elif running_space/size >= 0.4 and check40==False:
                print("40 % of data, {}/{} files have been copied".format(i, len(copy_list)))
                check40 = True
            elif running_space/size >= 0.5 and check50==False:
                print("50 % of data, {}/{} files have been copied".format(i, len(copy_list)))
                check50 = True
            elif running_space/size >= 0.6 and check60==False:
                print("60 % of data, {}/{} files have been copied".format(i, len(copy_list)))
                check60 = True
            elif running_space/size >= 0.7 and check70==False:
                print("70 % of data, {}/{} files have been copied".format(i, len(copy_list)))
                check70 = True
            elif running_space/size >= 0.8 and check80==False:
                print("80 % of data, {}/{} files have been copied".format(i, len(copy_list)))
                check80 = True
            elif running_space/size >= 0.9 and check90==False:
                print("90 % of data, {}/{} files have been copied".format(i, len(copy_list)))
                check90 = True
            elif running_space/size >= 1 and check100==False:
                print("100 % of data, {}/{} files have been copied".format(i, len(copy_list)))
                check100 = True
        break
    else:
        check2 = input("Are you sure you want to terminate before copying the files?: ")
        if check2 == 'yes' or check2 == 'Yes' or check2 == 'YES':
            sys.exit()

print("Done: {} files have been copied to {}.".format(len(copy_list), out_dir))