import nibabel as nib
import os 

path_to_files = "/home/magata/data/braintumor_data/VIGO_01/original"
dir = os.listdir(path_to_files)

for file in dir:
    filepath = os.path.join(path_to_files, file)
    nii = nib.load(filepath)
    new_filepath = filepath + ".gz"
    print(new_filepath)
    nib.save(nii, new_filepath)
