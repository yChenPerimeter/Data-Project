"The failure analysis process is 4 codes, 2 codes are in the imageassit branch and the other 2 
"developed by me. This code write all the false negative images into an excel file with the sublcasses"


import pandas as pd
import numpy as np
import os

fn = np.array(pd.read_csv('/mnt/Data4/Summer2024/RNarasimha/All_Model_Outputs/model_output_original_imgassist2/failure analysis /False_Negatives.csv', header=None)[0])
#fp = np.array(pd.read_csv('home/ragini/false_negatives.csv', header=None)[0])
test_dir = '/mnt/Data4/Summer2024/RNarasimha/ALL data+csv /Testing_multiclass/'

# Check count in folders
counter = 0
files_names = []
for root, dirs, files in os.walk(test_dir):
    print(root, len(files))
    # if root.split('/')[-1] not in ['Adipose']:
        # counter += len(files)
        # files_names += [root+'/'+f for f in files]
# counter

# Create file dictionary
file_dict = {}
for root, dirs, files in os.walk(test_dir):
    for file in files:
        file_dict[file] = root
# len(np.intersect1d(fp, list(file_dict.keys())))

# Create true class list
original_class = os.listdir('/mnt/Data4/Summer2024/RNarasimha/ALL data+csv /Testing_sus_nonsus/Suspicious')
                            #/mnt/Data4/Summer2024/520-00069_DataSet/Testing/Suspicious/')

print(len(original_class), original_class[0])

location = {}
for fname in fn:
    try:
        location[fname] = file_dict[fname].split('/')[-1]
    except:
        location[fname] = 'Not Found'

location
# Turn location dictionary into a dataframe with headers, file name, location
# False Negative means that it should be positive but was classified as negative
df = pd.DataFrame(list(location.items()), columns=['File Name', 'True Class'])
#df['Inference Label'] = df['File Name'].apply(lambda x: 'Non-Suspicious' if x in original_class else 'Suspicious')

df['Inference Label'] = df['File Name'].apply(
    lambda x: 'Suspicious' if x in original_class else 'Non-Suspicious')

df

df['True Class'].value_counts()
df.to_csv('/mnt/Data4/Summer2024/RNarasimha/All_Model_Outputs/model_output_original_imgassist2/failure analysis /FN_subclass.csv', index=False)
