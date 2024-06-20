
# note ; if you analyze not all channels, but with leaps, this program should be modified to provide the index of the dataframe.
    # =>    below : "index : if you analyze ... " 

# %%

# env_2
# immediate precursor to :
    # mu_fit.py in this folder.
    # server / Aryo / analysis / mu_tc / mu_tc_extract_spikes.py
# extracting a common sd & mean from the drift corrected (combined) recording, for mu (multiunit) analysis.

# policy :
    # silence_segment (index of trigger)   =>    trg_re_total (index of sample)   =>  silence trace


# %%
######

rd_pps_d = si.load_extractor(r'/home/azare/groups/PrimNeu/Aryo/analysis/sort/Elfie/p21/drift' )

# for trigger
#   change thess directories for each penetration.
# for any penetration , any measurement's analysis : all of that penetration's directories should be given.
# don't forget to place ',' between the direcotry strings. Otherwise you would get the following error :
        # OSError: No available data format detected.
dr = [
r'/home/azare/groups/PrimNeu/Aryo/copy_data/Elfie_final_exp_202303/p21/1/2023-03-23_14-10-15',
r'/home/azare/groups/PrimNeu/Aryo/copy_data/Elfie_final_exp_202303/p21/2/2023-03-23_14-14-11',
r'/home/azare/groups/PrimNeu/Aryo/copy_data/Elfie_final_exp_202303/p21/3/2023-03-23_14-51-44',
r'/home/azare/groups/PrimNeu/Aryo/copy_data/Elfie_final_exp_202303/p21/4/2023-03-23_15-23-55',
r'/home/azare/groups/PrimNeu/Aryo/copy_data/Elfie_final_exp_202303/p21/7/2023-03-23_15-33-42',
r'/home/azare/groups/PrimNeu/Aryo/copy_data/Elfie_final_exp_202303/p21/6/2023-03-23_15-36-56',
r'/home/azare/groups/PrimNeu/Aryo/copy_data/Elfie_final_exp_202303/p21/5/2023-03-23_15-44-31',
r'/home/azare/groups/PrimNeu/Aryo/copy_data/Elfie_final_exp_202303/p21/8/2023-03-23_15-47-23'
      ]

# here the directory is a penetration, not a subfolder (measurement), since it's from all measurements.
dest_dir = r'/home/azare/groups/PrimNeu/Aryo/analysis/Elfie/p21'

#   number of vectors (channels) to analyze.
nva = 384
# => mua_fit for the explanation.
leap = 1 
# initially this variable was 5 : every 100 microns.
# 1 means : no leap : all channels will be considered.
# 0 can not be an argument here.


# %%

trace_combined = rd_pps_d.get_traces()    # from the combined recording.
# trace_combined.shape
# # (193321932, 384)
# type(trace_combined)
# # Out[965]: numpy.memmap

####

# for extracting the triggers : these are later used to extract the silence periods.
# this uses o_e_p (open_ephys python tools) for reading the trigger channel.
# this uses original (non-drift-corrected) recordings to extract triggers.
# ap.s : action potentials for each subrecording.
aps = [] # unused.
trg_re_total = np.array([])
for i in dr :
    session = Session(i)
    rec = session.recordnodes[0].recordings[0]
    ap = rec.continuous[0].samples
    trg = ap[: , 384]
    t , di_t = ss.find_peaks( trg , plateau_size=20)
    trg_re = di_t['left_edges']
    trg_re_total = np.append( trg_re_total , trg_re )   # np.append does not happen in-place.

# casting to integer is needed below, to be used as an index for extracting traces.
trg_re_total = trg_re_total.astype(int)

trg_re_total_diff = np.diff(trg_re_total)

# index of the triggers right before the silence.
# there should be a few dozen, depending on the experiments.
# silence here is set to be > 10s.
# the array is inside parenthesis (unlike the documentation ! ).
start_silence = np.where( trg_re_total_diff > 300000 )[0]   


# %%
############

# tuple of silence segments.
# tuple, because later in the concatenation of them, the input of the function should be a tuple
tup_sil = ()
for i in start_silence :
    silence_segment = trace_combined[  trg_re_total[i] : trg_re_total[i+1]   ,  : ]
    tup_sil += ( silence_segment , )  # appending an object to a tuple : important : ',' !!!!

# a combined array of all silence segments, attached together.
# this of course contains all channels.
sil_total = np.concatenate( tup_sil , axis=0 )

# %%

# explore

tup_sil
    # Out[61]: 
    # (memmap([[  0,  18,   0, ..., -54,  -5, -50],
    #          [  0,  -5,   0, ..., -44, -10, -38],
    #          [  0,   7,   0, ..., -24,  13, -22],
    #          ...,
    #          [  0, -60,   0, ..., -29, -10, -47],
    #          [  0, -40,   0, ..., -22,  -5, -29],
    #          [  0,   2,   0, ..., -10,  12,  -2]], dtype=int16),
    #  memmap([[  0, -48,   0, ...,   0,   5,   0],
    #          [  0, -72,   0, ...,   0,  -6,   0],
    #          [  0, -67,   0, ...,   0, -15,   0],
    #          ...,
    #          [  0,   3,   0, ...,  -1, -58,  -3],
    #          [  0,  10,   0, ...,   0, -29,  -2],
    #          [  0,   6,   0, ...,  -4,  19,  -5]], dtype=int16),
    #  memmap([[  0,  12,   0, ...,  44,  19,  -3],
    #          ...
         
sil_total.shape
    # Out[62]: (26003910, 384)   #  14 minutes of silence in total.

# %%

# calculating the mean & sd of all columns
mean_total = np.mean( sil_total , axis=0)
sd_total = np.std( sil_total , axis=0)

# Create a pandas dataframe with the mean and standard deviation of each column
# index : if you analyze all of the 384 channels, the automatic index put here is valid to the channels numbers
    # if you analyze a subset of channels, the index should be provided to the dataframe accordingly.
mean_sd_total = pd.DataFrame( { 'mean': mean_total , 'sd': sd_total } )

mean_sd_total.to_pickle( dest_dir + '/mean_sd_total.pkl')

# mean_total.shape
    # Out[66]: (384,)

# mean_sd_total
    # Out[69]: 
    #          mean        sd
    # 0    0.007199 11.374399
    # 1    0.026136 42.611286
    # 2    0.014531 17.417401
    # 3    0.038513 36.094194
    # 4    0.005317 16.892530
    # ..        ...       ...
    # 379  0.015343 22.275422
    # 380  0.035035 31.659256
    # 381  0.021040 21.368213
    # 382 -0.004133 32.094681
    # 383  0.022967 21.216196
    
    # [384 rows x 2 columns]

# %%

# ap.shape
# # (58238496, 385)
# type(ap)
# # Out[964]: numpy.memmap

