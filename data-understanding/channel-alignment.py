import matplotlib.pyplot as plt
import matplotlib
import mne
import numpy as np
import re
import os
import utility.my_utils as deniz
import pandas as pd
import os
import shutil
import importlib
import warnings
warnings.filterwarnings("ignore")

"""
End-to-end data curation pipeline for the CHB-MIT dataset.
The goal:
- Take the raw, messy dataset
- Produce a clean, standardized folder (seizured_files) 
- Containing only the files that have seizures and compatible channel configurations.
"""
importlib.reload(deniz)
dataset_path = 'data-understanding/data/chb-mit'
subject_folders = deniz.get_folder_names(dataset_path)
edf_paths = deniz.get_edf_paths(dataset_path, subject_folders)
# channels_dict = deniz.get_channels_as_dict(dataset_path, subject_folders, json_name='channels', save=True)
channels_dict = deniz.get_channels_as_dict(dataset_path, subject_folders, json_name='channels', save=False)
cleaned_dict = deniz.normalize_channel_dict(channels_dict)

cleaned_dict.values()

# TARGET_CHANNELS = [
#     'FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1',
#     'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
#     'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2',
#     'FP2-F8', 'F8-T8', 'T8-P8', 'P8-O2',
#     'FZ-CZ', 'CZ-PZ'
# ]

# list of unique values of channels
a_list = [item for sublist in cleaned_dict.values() for item in sublist]

set(TARGET_CHANNELS) <= set(a_list)

print(a_list)

print(set(a_list))

type(channels_dict)

# Loads all channel names from all files into a dictionary
for key, old_list in channels_dict.items():
    new_list = cleaned_dict.get(key, [])

    # By taking the cluster difference, we find the ones that are in the old list but not in the new list
    removed = set(old_list) - set(new_list)
    added = set(new_list) - set(old_list)

    if removed or added:
        print(f"--- Directory: {key} ---")
        if removed: print(f"Removed/Changed: {removed}")
        if added:   print(f"Added/Modified: {added}")

# Looking for each patient
TARGET_CHANNELS = [
    'FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1',
    'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
    'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2',
    'FP2-F8', 'F8-T8', 'T8-P8-0', 'T8-P8-1', 'P8-O2', 'T8-P8',
    'FZ-CZ', 'CZ-PZ'
    ]

# Use of set to normalize channels (for search speed)
target_set = set(TARGET_CHANNELS)
missing_files = []
dataset_path = 'data-understanding/data/chb-mit'
subject_folders = deniz.get_folder_names(dataset_path)
CHB01_paths = deniz.get_edf_paths(dataset_path, subject_folders[0])
CHB02_paths = deniz.get_edf_paths(dataset_path, subject_folders[1])
CHB03_paths = deniz.get_edf_paths(dataset_path, subject_folders[2])
CHB04_paths = deniz.get_edf_paths(dataset_path, subject_folders[3])
CHB05_paths = deniz.get_edf_paths(dataset_path, subject_folders[4])
CHB06_paths = deniz.get_edf_paths(dataset_path, subject_folders[5])
CHB07_paths = deniz.get_edf_paths(dataset_path, subject_folders[6])
CHB08_paths = deniz.get_edf_paths(dataset_path, subject_folders[7])
CHB09_paths = deniz.get_edf_paths(dataset_path, subject_folders[8])
CHB10_paths = deniz.get_edf_paths(dataset_path, subject_folders[9])
CHB11_paths = deniz.get_edf_paths(dataset_path, subject_folders[10])
CHB12_paths = deniz.get_edf_paths(dataset_path, subject_folders[11])
CHB13_paths = deniz.get_edf_paths(dataset_path, subject_folders[12])
CHB14_paths = deniz.get_edf_paths(dataset_path, subject_folders[13])
CHB15_paths = deniz.get_edf_paths(dataset_path, subject_folders[14])
CHB16_paths = deniz.get_edf_paths(dataset_path, subject_folders[15])
CHB17_paths = deniz.get_edf_paths(dataset_path, subject_folders[16])
CHB18_paths = deniz.get_edf_paths(dataset_path, subject_folders[17])
CHB19_paths = deniz.get_edf_paths(dataset_path, subject_folders[18])
CHB20_paths = deniz.get_edf_paths(dataset_path, subject_folders[19])
CHB21_paths = deniz.get_edf_paths(dataset_path, subject_folders[20])
CHB22_paths = deniz.get_edf_paths(dataset_path, subject_folders[21])
CHB23_paths = deniz.get_edf_paths(dataset_path, subject_folders[22])
CHB24_paths = deniz.get_edf_paths(dataset_path, subject_folders[23])

# CHB01 * T8-P8 is in all
deniz.check_is_every_edf_containts_target_chs(edf_paths=CHB01_paths, target_set=target_set, missing_files=missing_files)
# CHB02 * T8-P8 is in all
deniz.check_is_every_edf_containts_target_chs(edf_paths=CHB02_paths, target_set=target_set, missing_files=missing_files)
# CHB03 * T8-P8 is in all
deniz.check_is_every_edf_containts_target_chs(edf_paths=CHB03_paths, target_set=target_set, missing_files=missing_files)
# CHB04 * 'T8-P8' is in all
deniz.check_is_every_edf_containts_target_chs(edf_paths=CHB04_paths, target_set=target_set, missing_files=missing_files)
# CHB05 * 'T8-P8' is in all
deniz.check_is_every_edf_containts_target_chs(edf_paths=CHB05_paths, target_set=target_set, missing_files=missing_files)
# CHB06 * 'T8-P8' is in all
deniz.check_is_every_edf_containts_target_chs(edf_paths=CHB06_paths, target_set=target_set, missing_files=missing_files)
# CHB07 * 'T8-P8' is in all
deniz.check_is_every_edf_containts_target_chs(edf_paths=CHB07_paths, target_set=target_set, missing_files=missing_files)
# CHB08 * 'T8-P8' is in all
deniz.check_is_every_edf_containts_target_chs(edf_paths=CHB08_paths, target_set=target_set, missing_files=missing_files)
# CHB09 * 'T8-P8' is in all
deniz.check_is_every_edf_containts_target_chs(edf_paths=CHB09_paths, target_set=target_set, missing_files=missing_files)
# CHB10 * 'T8-P8' is in all
deniz.check_is_every_edf_containts_target_chs(edf_paths=CHB10_paths, target_set=target_set, missing_files=missing_files)
# CHB11 * 'T8-P8' is in all
deniz.check_is_every_edf_containts_target_chs(edf_paths=CHB11_paths, target_set=target_set, missing_files=missing_files)

# CHB14 * 'T8-P8' is in all
deniz.check_is_every_edf_containts_target_chs(edf_paths=CHB14_paths, target_set=target_set, missing_files=missing_files)
# CHB15 * Only on the first one {'T8-P8-0', 'T8-P8-1'} the rest of 'T8-P8' is missing.
deniz.check_is_every_edf_containts_target_chs(edf_paths=CHB15_paths, target_set=target_set, missing_files=missing_files)
# CHB16 * Last two  {'T8-P8-0', 'T8-P8-1'} the rest of {'T8-P8'} is missing.
deniz.check_is_every_edf_containts_target_chs(edf_paths=CHB16_paths, target_set=target_set, missing_files=missing_files)
# CHB17 * Last {'T8-P8-0', 'T8-P8-1'} the rest of {'T8-P8'} is missing.
deniz.check_is_every_edf_containts_target_chs(edf_paths=CHB17_paths, target_set=target_set, missing_files=missing_files)
# CHB18 * first {'T8-P8-0', 'T8-P8-1'} the rest of {'T8-P8'} is missing.
deniz.check_is_every_edf_containts_target_chs(edf_paths=CHB18_paths, target_set=target_set, missing_files=missing_files)
# CHB19 * first {'T8-P8-0', 'T8-P8-1'} the rest of {'T8-P8'}...
deniz.check_is_every_edf_containts_target_chs(edf_paths=CHB19_paths, target_set=target_set, missing_files=missing_files)
# CHB20 * 'T8-P8' is in all
deniz.check_is_every_edf_containts_target_chs(edf_paths=CHB20_paths, target_set=target_set, missing_files=missing_files)
# CHB21 * 'T8-P8' is in all
deniz.check_is_every_edf_containts_target_chs(edf_paths=CHB21_paths, target_set=target_set, missing_files=missing_files)
# CHB22 * 'T8-P8' is in all
deniz.check_is_every_edf_containts_target_chs(edf_paths=CHB22_paths, target_set=target_set, missing_files=missing_files)
# CHB23 'T8-P8' is in all
deniz.check_is_every_edf_containts_target_chs(edf_paths=CHB23_paths, target_set=target_set, missing_files=missing_files)
# CHB24 * 'T8-P8' is in all
deniz.check_is_every_edf_containts_target_chs(edf_paths=CHB24_paths, target_set=target_set, missing_files=missing_files)

# Patients like CHB12, CHB15, and CHB16 have missing or inconsistent channels, and noted it down:
# CHB12
deniz.check_is_every_edf_containts_target_chs(edf_paths=CHB12_paths, target_set=target_set, missing_files=missing_files)

# CHB13 * in some {'T8-P8-0', 'T8-P8-1'} in some 'T8-P8' is missing
deniz.check_is_every_edf_containts_target_chs(edf_paths=CHB13_paths, target_set=target_set, missing_files=missing_files)

# CHB15 * in some {'T8-P8-0', 'T8-P8-1'} in some 'T8-P8' is missing
deniz.check_is_every_edf_containts_target_chs(edf_paths=CHB15_paths, target_set=target_set, missing_files=missing_files)

# CHB16 * in some {'T8-P8-0', 'T8-P8-1'} in some 'T8-P8' is missing
deniz.check_is_every_edf_containts_target_chs(edf_paths=CHB16_paths, target_set=target_set, missing_files=missing_files)


#############################
dataset_path = 'data-understanding/data/chb-mit'
subject_folders = deniz.get_folder_names(dataset_path)
edf_paths = deniz.get_edf_paths(dataset_path, subject_folders[0])

raw = mne.io.read_raw_edf(CHB12_paths[11], preload=False)

print(raw.ch_names)

###########################33
dataset_path = 'data-understanding/data/chb-mit'
###Get only seizured files
df = deniz.build_seizure_dataframe(dataset_path)
# df.to_csv('only_seizure_dataframe.csv')
df['path'] = df['folder'] + '/' + df['file']
seizured_file_list = df['file'].unique()
df.to_csv('only_seizure_dataframe.csv')
##################################################################################################################
TARGET_CHANNELS = [
    'FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1',
    'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
    'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2',
    'FP2-F8', 'F8-T8', 'T8-P8-0', 'T8-P8-1', 'P8-O2', 'T8-P8',
    'FZ-CZ', 'CZ-PZ'
]

target_set = set(TARGET_CHANNELS)

seizured_file_list = []
for i in df['path']:
    seizured_file_list.append(i)
seizured_file_list = set(seizured_file_list)
dataset_path = 'data-understanding/data/chb-mit'
seizured_file_list = [dataset_path + '/' + item for item in seizured_file_list]

missing_files = []
deniz.check_is_every_edf_containts_target_chs(seizured_file_list, target_set=target_set, missing_files=missing_files)

seizured_file_list = set(seizured_file_list)

# Seizure Identification & Filtering
to_remove = {
    'data-understanding/data/chb-mit/chb12/chb12_29.edf',
    'data-understanding/data/chb-mit/chb12/chb12_27.edf',
    'data-understanding/data/chb-mit/chb12/chb12_28.edf'
}
seizured_file_list.difference_update(to_remove)
# CHB12 is having completely different channel montages that are incompatible with the other patients.
print("data-understanding/data/chb-mit/chb12/chb12_29.edf" in seizured_file_list)


seizured_file_list = list(seizured_file_list)
seizured_file_list = sorted(seizured_file_list)


# def analyze_only_seizures(dataset_path, df, target_channels, seizured_file_list,missing_files):
#     target_channels = set(target_channels)
#
#     deniz.check_is_every_edf_containts_target_chs(edf_paths=edf_paths, target_set=target_channels)


deniz.copy_files_to_seizured_folder(seizured_file_list)

filtered_df = df[df['folder'] != 'chb12']
total_seizure_elapse = (filtered_df['seizure_end_sec'] - filtered_df['seizure_start_sec']).sum()
print(f"Total Seizure duration: {total_seizure_elapse}")
total_time = len(seizured_file_list) * 3600
percent = (total_seizure_elapse / total_time) * 100
print('ratio of the duration of the duration of the duration of the total duration of edf files containing Seizure: %', percent)

# ## start preprocessing
# seizure_file_path = 'seizured_files'
# seizure_paths = deniz.get_edf_paths(seizure_file_path, '')
# seizure_paths = sorted(seizure_paths)
#
# # Channel adjustment
#
# FINAL_CHANNELS = [
#     'FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1',
#     'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
#     'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2',
#     'FP2-F8', 'F8-T8', 'T8-P8', 'P8-O2',
#     'FZ-CZ', 'CZ-PZ'
# ]
# processed_raws = []
# for file_path in seizure_paths:
#     data = deniz.fix_eeg_channels(file_path, FINAL_CHANNELS)
#     if data:
#         processed_raws.append(data)
#
# something = processed_raws[60]
# something.ch_names

######################3
#
#
# def preprocess_pipe(seizure_paths, final_channels,output_folder):
#
#     channel_raw = []
#     for file_path in seizure_paths:
#         data = deniz.fix_eeg_channels(file_path, final_channels)
#         if data:
#             channel_raw.append(data)
#     print('')
#     processed_raws=[]
#     for i in channel_raw:
#         # load edf
#         raw = i
#         # map channels
#
#         # down_sample
#         raw = deniz.downsample(raw)
#
#         # filtering
#         raw = deniz.apply_filtering(raw)
#
#         processed_raws.append(raw)
#         #save
#         deniz.save_processed_edfs(processed_raws, output_folder=output_folder)
#     return processed_raws
#
#
# FINAL_CHANNELS = [
#     'FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1',
#     'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
#     'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2',
#     'FP2-F8', 'F8-T8', 'T8-P8', 'P8-O2',
#     'FZ-CZ', 'CZ-PZ'
# ]
# processed_raws= preprocess_pipe(seizure_paths,final_channels=FINAL_CHANNELS)
#
# zero = processed_raws[0]
#
# zero.filenames[0].stem
#
#
#
# zero.info
#
#
# save_processed_edfs(processed_raws)