# Report

The report is in the REPORT directory.

# Installation

Start the terminal and you can install the virtual environment as follow:

```bash
# Go to assignment folder
cd ML395_NN

# Change permissions to execute bash script
chmod 755 install_lab_env.sh

# Create virtual environment 'env' and install requirements
# This can take several minutes to finish
./install_lab_env.sh
```

# How to run src/test.py for Q5 and 6

Here is the example of how to run `src/test.py`.

```
model_path_q5 = "PATH/TO/THIS/REPO/pkl/final_model.pkl"
img_folder_q5 = "PATH/TO/IMAGE"

# Call test_fer_model() for Q5
test_fer_model(img_folder, model_path)


model_path_q6 = "PATH/TO/THIS/REPO/src/question6/models/vgg_netvgg.hdf5"
img_folder_q6 = "PATH/TO/IMAGE"

# Call test_deep_fer_model() for Q6
test_deep_fer_model(img_folder, model_path)
```

`train_fcnet_optimise_params.py` is used to train the optimised model for question 5. The model can be trained directly from the JPEG data on bitbucket, however this is slow to load. Also provided is a function to load the data from a pickle object, which loads significantly faster.

`train_fcnet_optimise_params_gridsearch` and `train_fcnet_optimise_params_gridsearch_2dims` are used only to search for parameters and were not used to train the final model.

# Instruction

395 Machine Learning: Assignment 2
=====================================

Please notice that this year, we have an intermediate and an advanced version
for the second assignment. Identify if your group consists of:

1. Master's Computing students: You **must** complete this advanced assignment
(`manuals/assignment2_advanced.md`, pdf format is also available).
2. Non Master's Computing students
(Bachelor's Computing/non-Computing/external/exchange): We recommend you
complete the the intermediate assignment which is available on the group web
site for download. Optionally: You **can** also choose this advanced assignment,
 it is up to you.
3. Mixed of Master's Computing and non Master's Computing students: You
 **must** complete the advanced assignment (`manuals/assignment2_advanced.md`,
 pdf format is also available).
