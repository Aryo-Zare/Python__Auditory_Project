
# env_17
# pipe _ sort
# G:\Aryo\analysis\sort

################
#####################

# create : a destination directory on windows explorer (not here), to save the results ( sorting objects ) : corresponding to dest_dir.
# change : the following n+1 directories !:
#       n : source recording directories
#       1 : destination directory.
# adjust :
#       recording objects (rd_n).
#       rd (total).
# select the correpsponding probe : for Lucy there is a linear selection of contact points with no overlap.
# select the corresponding stream_id : left or right hemisphere for Lucy.
# you do not need to run any program from pre_req_n.py.
# copy the changeable lines below to  Dell \ D:\address \ file_sort  .docx


# attention : if you're copy-pasting the directory from windows, don't forget to change the backslash to forward-slashes.
# another reason for error reading a file is with bilateral recordings in which 1 probe fails to be written : check the contents of the recording folder.
dr_1 = r'/home/azare/groups/PrimNeu/Aryo/copy_data/Elfie_final_exp_202303/p7/2/2023-03-21_19-12-10'
dr_2 = r'/home/azare/groups/PrimNeu/Aryo/copy_data/Elfie_final_exp_202303/p7/3/2023-03-21_19-15-13'
dr_3 = r'/home/azare/groups/PrimNeu/Aryo/copy_data/Elfie_final_exp_202303/p7/4/2023-03-21_19-47-12'
dr_4 = r'/home/azare/groups/PrimNeu/Aryo/copy_data/Elfie_final_exp_202303/p7/5/2023-03-21_19-56-10'
dr_5 = r'/home/azare/groups/PrimNeu/Aryo/copy_data/Elfie_final_exp_202303/p7/6/2023-03-21_20-01-43'
dr_6 = r'/home/azare/groups/PrimNeu/Aryo/copy_data/Elfie_final_exp_202303/p7/8/2023-03-21_20-20-48'
dr_7 = r'/home/azare/groups/PrimNeu/Aryo/copy_data/Elfie_final_exp_202303/p7/9/2023-03-21_20-49-03'
dr_8 = r'/home/azare/groups/PrimNeu/Aryo/copy_data/Elfie_final_exp_202303/p7/10/2023-03-21_21-24-35'
# dr_9 = r'/home/azare/groups/PrimNeu/Aryo/copy_data/Elfie_final_exp_202303/p7/9/2023-03-21_17-59-53'
# dr_10 = r'/home/azare/groups/PrimNeu/Aryo/copy_data/Elfie_final_exp_202303/p6/10/2023-03-21_18-45-07'
# dr_11 = r'/home/azare/groups/PrimNeu/Aryo/copy_data/Elfie_final_exp_202303/p6/11/2023-03-21_18-50-55'
# dr_10 = r'/home/azare/groups/PrimNeu/Aryo/copy_data/Elfie_final_exp_202303/p5/11/2023-03-21_11-49-49'
# dr_11 = r'/home/azare/groups/PrimNeu/Aryo/copy_data/Elfie_final_exp_202303/p5/11/2023-03-21_11-49-49'

# G:\Aryo\copy_data\Elfie_final_exp_202303/p1/7/2023-03-20_19-47-09


# recording objects.
# sample : if there are multiple experiments in 1 recording (multi-block), use 'block_index' to extract each one separately.
    # rd_6 = se.read_openephys(dr_6 , stream_id='0' , block_index=0)    
    # this happens during data acquisition when in open-ephys for a new recording a new destination folder is not set (by mistake).
    # partially because of open_ephys gui's bug that you need to delete & re-bring the record node plug-in.
rd_1 = se.read_openephys(dr_1 , stream_id='0')
rd_2 = se.read_openephys(dr_1 , stream_id='0')  # change the dr_# number.
rd_3 = se.read_openephys(dr_3 , stream_id='0')
rd_4 = se.read_openephys(dr_4 , stream_id='0')
rd_5 = se.read_openephys(dr_5 , stream_id='0')
rd_6 = se.read_openephys(dr_6 , stream_id='0')
rd_7 = se.read_openephys(dr_7 , stream_id='0')
rd_8 = se.read_openephys(dr_8 , stream_id='0')
# rd_9 = se.read_openephys(dr_9 , stream_id='0')
# rd_10 = se.read_openephys(dr_9 , stream_id='0')
# rd_11 = se.read_openephys(dr_9 , stream_id='0')

# combined recording.
rd = si.concatenate_recordings( [ 
    rd_1 , 
    rd_2 , 
    rd_3 , 
    rd_4 , 
    rd_5 , 
    rd_6 ,
    rd_7 ,
    rd_8 ,
    # rd_9 ,
    # rd_10 ,
    # rd_11
] )


# destination directory
# this must already be created.
dest_dir = r'/home/azare/groups/PrimNeu/Aryo/analysis/sort/Elfie/p7'

###############
###############
###############
###############


prb = read_probeinterface(r'/home/azare/groups/PrimNeu/Aryo/copy_data/Lucy_20221219/Lucy.json')
#prb = read_probeinterface(r'/home/azare/groups/PrimNeu/Aryo/analysis/sort/overlap_linear-2.json')

rd_prb = rd.set_probegroup(prb)

rd_prb_f = bandpass_filter(rd_prb , freq_min=300, freq_max=6000)

rd_prb_f_cmr = common_reference(rd_prb_f , reference='global', operator='median')

# pps : preprocessed
rd_pps = rd_prb_f_cmr.save(folder= dest_dir + '/pps' ,  format='binary' , n_jobs=-1, chunk_size=30000  )


#################
#################

# drift correction.

job_kwargs = dict(chunk_duration='1s', n_jobs=-1, progress_bar=True)

# the output will be numpy_compact, even_though this is not mentioned as a kwarg here.
peaks = detect_peaks(
    rd_pps, 
    method='by_channel',
    peak_sign='both',
    detect_threshold=5,
    exclude_sweep_ms=0.2,
    noise_levels=None,
    random_chunk_kwargs={},
    **job_kwargs
)


peak_locations = localize_peaks(
    rd_pps, 
    peaks, 
    method='center_of_mass',
    local_radius_um=70 , 
    ms_before=0.3, 
    ms_after=0.6,
    **job_kwargs
)


motion, temporal_bins, spatial_bins, extra_check = estimate_motion(
    rd_pps, 
    peaks, 
    peak_locations=peak_locations,
    direction='y', 
    bin_duration_s=10., 
    bin_um=10., 
    margin_um=0.,
    method='decentralized',
    rigid=False, 
    win_shape='gaussian', 
    win_step_um=50., 
    win_sigma_um=150.,
    output_extra_check=True ,
    progress_bar=True 
)



recording_corrected = CorrectMotionRecording(
    rd_pps, 
    motion, 
    temporal_bins, 
    spatial_bins,
    spatial_interpolation_method='kriging',
    border_mode='force_zeros'
)


# d : drift corrected.
rd_pps_d = recording_corrected.save(folder= dest_dir + '/drift' ,  format='binary' , n_jobs=-1, chunk_size=30000  )

############

# saving the variables.

np.save( dest_dir + '/peaks.npy' , peaks)
np.save( dest_dir + '/peak_locations.npy' , peak_locations)


np.save( dest_dir + '/motion.npy' , motion)
np.save( dest_dir + '/temporal_bins.npy' , temporal_bins)
np.save( dest_dir + '/spatial_bins.npy' , spatial_bins)

# pp_w : pickle protocol _ write
# pp_r : pickle protocol _ read
# extra_check is a dictionary & can not be converted to pandas (better not to be).
with open( dest_dir + '/extra_check.pickle' , 'wb' ) as pp_w :
    pickle.dump( extra_check , pp_w )

# to load it again ;
# instead of 'path' write the path of the saved pickled file (here : = the above dest_dir).
# with open( path + '/extra_check.pickle' , 'rb' ) as pp_r :
#     extra_check = pickle.load(pp_r)

###########
# plot

# sw.plot_drift_over_time(rd_pps , peaks, mode='scatter' )
# plt.plot(motion[ : , :10])

## these are seemmingly overlapped on the probe map.
# sw.plot_peak_activity_map(rd_pps , peaks)     # static
# sw.plot_peak_activity_map(rd_pps , peaks , bin_duration_s=60)     # animated (per-bin-duration) : has a problem.

# plt.savefig(r'/home/azare/groups/PrimNeu/Aryo/analysis/sort/drift/motion_6.pdf')

##################
##################


srt = sst.run_sorter( 
    sorter_name='spykingcircus' , 
    recording=rd_pps_d , 
    detect_sign=0 , 
    detect_threshold=8 , 
    template_width_ms=2 ,  
    filter=False , 
    auto_merge=0.5 , 
    num_workers=20 , 
    verbose=True , 
    output_folder = dest_dir + '/srt'
)


wfe = si.extract_waveforms(
    rd_pps_d , 
    srt , 
    ms_before=1, 
    ms_after=1 , 
    n_jobs=-1, 
    chunk_size=30000 ,  
    verbose=True , 
    folder= dest_dir + '/wfe'
)



###########
###########

# parameters of quality metrics.
# qm.get_default_qm_params()

# set : isi_threshold_ms
qm_p = {
    'presence_ratio': {'bin_duration_s': 60},
    'snr': {'peak_sign': 'both',
     'peak_mode': 'extremum',
     'random_chunk_kwargs_dict': None},
    'isi_violation': {'isi_threshold_ms': 1, 'min_isi_ms': 0},
    'rp_violation': {'refractory_period_ms': 1.0, 'censored_period_ms': 0.0},
    'sliding_rp_violation': {'bin_size_ms': 0.25,
     'window_size_s': 1,
     'exclude_ref_period_below_ms': 0.5,
     'max_ref_period_ms': 10,
     'contamination_values': None},
    'amplitude_cutoff': {'peak_sign': 'neg',
     'num_histogram_bins': 100,
     'histogram_smoothing_value': 3,
     'amplitudes_bins_min_ratio': 5},
    'amplitude_median': {'peak_sign': 'neg'},
    'drift': {'interval_s': 60,
     'min_spikes_per_interval': 100,
     'direction': 'y',
     'min_num_bins': 2},
    'nearest_neighbor': {'max_spikes': 10000, 'n_neighbors': 5},
    'nn_isolation': {'max_spikes': 10000,
     'min_spikes': 10,
     'n_neighbors': 4,
     'n_components': 10,
     'radius_um': 100,
     'peak_sign': 'neg'},
    'nn_noise_overlap': {'max_spikes': 10000,
     'min_spikes': 10,
     'n_neighbors': 4,
     'n_components': 10,
     'radius_um': 100,
     'peak_sign': 'neg'}}


# quality metrics.
metrics = compute_quality_metrics(wfe, 
                                  metric_names=['num_spikes',
                                   'firing_rate',
                                   'presence_ratio',
                                   'snr',
                                   'isi_violation', # this will automatically compute : 'isi_violations_ratio' & 'isi_violations_count'.
                                   #'rp_violation',     # requires numba
                                   #'sliding_rp_violation',    #  requires numba
                                   'amplitude_cutoff',
                                   'amplitude_median',
                                   'drift'],
                                  qm_params=qm_p
                                  )

# a : all (not curated).
metrics.to_pickle( dest_dir + '/qm_a.pkl')
metrics.to_hdf( dest_dir + '/qm_a.h5', key='qm_a', mode='w')


# True , False  
keep = \
    (metrics['snr'] > 2) & \
    (metrics['isi_violations_ratio'] < 0.3 ) 


# c : curated.
# qm : quality metrics.
# reasons on saving a datframe as pickle  =>   server // unit_fit_11.py
metrics_c = metrics[keep]

metrics_c.to_pickle( dest_dir + '/qm_c.pkl')
metrics_c.to_hdf( dest_dir + '/qm_c.h5', key='qm_c', mode='w')

# the result is a numpy array.
keep_uids = keep[keep].index.values
# keep_uids.size

# c : curated 
# this may not be needed. possibly only cuts the data volume short in case of storage (or RAM ?)  concerns.
wfe_c = wfe.select_units(keep_uids , dest_dir + '/wfe_c')

###############

# tm : template metrics
tm = post.compute_template_metrics(wfe_c)

tm.to_pickle( dest_dir + '/tm.pkl')
tm.to_hdf( dest_dir + '/tm.h5', key='tm', mode='w')

#########

# skt : spike-trains
# dm : dimensionalized 
skt_all = srt.get_all_spike_trains()
skt_dm = np.hstack(skt_all)

np.save( dest_dir + '/skt_dm.npy' , skt_dm.astype(int) )

###########

np.save( dest_dir + '/unit_id.npy' , keep_uids )

# this is to avoid confusion.
unit_id = keep_uids



# rd_pps = si.load_extractor(r'/home/azare/groups/PrimNeu/Aryo/analysis/sort/Lucy/p15/3_prev_2/pps' )
# rd_pps_d = si.load_extractor(r'/home/azare/groups/PrimNeu/Aryo/analysis/sort/Lucy/p15/3/drift' )
# motion =np.load(r'/home/azare/groups/PrimNeu/Aryo/analysis/sort/Lucy/p15/3_prev_2/motion.npy')

###############
###############

# packages

# import spikeinterface as si  
# import spikeinterface.extractors as se
# import spikeinterface.sorters as sst    
# import spikeinterface.widgets as sw
# import spikeinterface.exporters as exp

# from spikeinterface.preprocessing import (bandpass_filter, common_reference)
# import spikeinterface.postprocessing as post


# import spikeinterface.qualitymetrics as qm
# from spikeinterface.qualitymetrics import compute_quality_metrics

# from probeinterface import read_probeinterface


# ##############

# import spikeinterface.curation as cur

# from spikeinterface.sortingcomponents.peak_detection import detect_peaks
# from spikeinterface.sortingcomponents.peak_localization import localize_peaks
# from spikeinterface.sortingcomponents.motion_estimation import estimate_motion
# from spikeinterface.sortingcomponents.motion_correction import CorrectMotionRecording

# #################

# import pickle

#####################
###############

# wfe_c = si.WaveformExtractor.load_from_folder( source_dir + '/wfe_c')
