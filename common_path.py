import os
'''Things that we can change'''
###################################################
siam_model_ = 'siamrpn_r50_l234_dwxcorr'
# siam_model_ = 'siamrpn_r50_l234_dwxcorr_otb'
###################################################
# dataset_name_ = 'GOT-10k'
# dataset_name_ = 'OTB100'
# dataset_name_ = 'TrackingNet'
# dataset_name_ = 'VOT2018
dataset_name_ = 'UAV123'
# dataset_name_ = 'LaSOT'
##################################################
# video_name_ = 'CarScale' # worser(inaccurate scale estimation)
# video_name_ = 'Bolt' # fail earlier(distractor)
# video_name_ = 'Doll' # unstable
# video_name_ = 'ants1'
# video_name_ = 'airplane-1'
video_name_ = ''
#########################################################################################
'''change to yours'''
project_path_ = '/data_B/renjie//CSA'
dataset_root_ = '/data_B/renjie/CSA/pysot/testing_dataset'# make links for used datasets
train_set_path_ = '/data_B/renjie/CSA/pysot/testing_dataset/GOT10K_reproduce' #

