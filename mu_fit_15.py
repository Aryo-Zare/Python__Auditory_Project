
# server // analysis / fit / mu_fit / mu_fit_n.py
# env_2

# %%

# run the 'functions=' cell in the pre-req file.
# create a new destination folder ('mu') in windows explorer.
# fill the values of the 2 consequative cells below
    # 1st cell ; fill the directories & the suptitle string
    # 2nd cell ; sample_index :  beginning_end ;  G:\Aryo\analysis\sample \ sample .xlsx

# %%

# note : if you analyze not all channels, but with leaps : in this program : the initialization of the datframe is by 384 channels
    # but those leaped channels are 0s.

# nmz in file's name = normalized.
# re_sig in file's name = response significance.
# also change the baseline for short sois.

# %%

# These columns should be later added to the dataframes :
    # 'frequency', 'nsr', 'animal', 'hemisphere',
    # 'penetration', 'stimulus', 'ck_pt_n', 'latency_average',
    # 'nsr_6soi', 'AP', 'bf', 'rsbf', 'rsbf_abs']

# %%

# multi-block recordings 
    # for the extraction of the raw traces from the combined drift-corrected recordings : do not make any difference here, 
        # since it's already taken into account when merging several single recordings together.
    # triggers  =>   refer below : rec = session.recordnodes[0]... .

# for trigger
#   change this directory.
directory = r'/home/azare/groups/PrimNeu/Aryo/copy_data/Elfie_final_exp_202303/p20/7/2023-03-23_13-25-57'

# you should 1st create this in windows explorer.
# for pdfs & the database.
dest_dir = r'/home/azare/groups/PrimNeu/Aryo/analysis/Elfie/p20/7/mu'

# this will be printed at sup-title.
description_session = 'Elfie_terminal , left hemisphere _ p20_7_R (primate probe) , tone 16000 Hz'

#######
# these 2 are penetration-dependent , not measurement-dependent.

# for extracting the spikes.
rd_pps_d = si.load_extractor(r'/home/azare/groups/PrimNeu/Aryo/analysis/sort/Elfie/p20/drift' )

# directory of the silence periods' mean & sd of a particular penetration
# pnt : penetration
dir_sil_pnt = r'/home/azare/groups/PrimNeu/Aryo/analysis/Elfie/p20'

# %%

# =>   G:\Aryo\analysis\sample  \  sample .xlsx

sample_index_beginning = 137878504

sample_index_end = 196826000


# %%
# %%

# related to the trigger.

# nte : umber of trials (out of 100) to exclude.
nte = 3
# nrt : number of remaining trials.
nrt = 100 - nte

# %%

# related to the channels

#   number of vectors (channels) to analyze.
nva = 384
# the step for analysis of channels. Every 'leap' channels will be analysed. 5 channels = 100 microns. 
# the total number of channels analyzed would be 384 / 5.
# 1 means : no leap : all channels will be considered.
leap = 1

# %%
# %%

# mean & sd pf the silence periods' of this penetration 
# includes the silence periods of the whole 2 hours recording.
mean_sd_total = pd.read_pickle( dir_sil_pnt + '/mean_sd_total.pkl')

# %%
#################
#################

# pre-block rate.
ticks_pbr = np.arange(7)  #  tick positions : generally automatically set, but specifically needed if you want to set the labels (next line).
labels_pbr = np.arange(1,8)

########

#  for significance testing : this version is modified to contain each bin containing 100ms :   of course for pre-block part.
# tdm : tandem.
#  res_tdm_8_soi : for the response : 10-110ms after each stimulus.
#  base_tdm_8_soi : for baseline : during the 1min silence interval.

#############

# bin sizes : 

# for response magnitude calculation (normalized ... ) : 25 ms : of course for both response & baseline periods.
#     corresponding variables :  base_neb_mean_8_100_rint  , base_event_8 ,  m.
    
# for significance testing (distributions) : 100 ms : of course for both response & baseline periods.
#      corresponding variables :  they have 'tdm' (tandem) in their name.    base_tdm_8_soi  ,  res_tdm_8_soi.
    

###############

# changes in this version : bottom subfigure : adding another trace : base events as measured before each event (base_event_8).
#  changes in this version ; instead  subtracting 1 minute silence interval baseline from the max response : here the baseline before the event is subtracted 

# in modules . py  :
# from open_ephys.analysis import Session
# from scipy.optimize import curve_fit


# def fit_func(SOI, A, tau_0):
#     return A*(1-np.exp(-(SOI-0.05)/tau_0))

# sois = np.array([ 0.11 , 0.195 , 0.345 , 0.611 , 1.081 , 1.914 , 3.388 , 6])

################

# for extracting the triggers.
# in case of non-separate recorindgs (multi-block recordings) : example : Elfie p2_2 : due to Michael's mistake : run the below line
    # rec = session.recordnodes[0].recordings[1]
    # the main 'directory' [ in session = Session(directory) ] is the directory of the parent recording (p2_1 : contains p2_2 ).
session = Session(directory)
rec = session.recordnodes[0].recordings[0]
ap = rec.continuous[0].samples


####################

#	triggers.
trg = ap[: , 384]

t , di_t = ss.find_peaks( trg , plateau_size=20)

trg_re = di_t['left_edges']

#   you can then check the shape of the array in Ipython console:
#   trg_re.shape


#############
#############

# trg_re_r : trigger reshaped.
trg_re_r = trg_re.reshape(8,100)


# the new sorted trigger.
t_8_100 = np.zeros((8,100))

# this is the order of presentation of the sois.
# the order of presentation of the sois were randomized.
# here, one can fetch the actual order.
soi_order = [1,1,1,1,1,1,1,1]   #  string (below)
soi_order_numeric = [1,1,1,1,1,1,1,1]

# below : diff functions requires an array, not a vector. hence I reshaped it.

for i in range(8) :
    if ( ( np.diff(trg_re_r[i , :].reshape(1,100))[0,0] > 3000) & ( np.diff(trg_re_r[i , :].reshape(1,100))[0,0] < 3500) ) :
        t_8_100[0 , :] = trg_re_r[i,:]
        soi_order[i] = 'soi_1'
        soi_order_numeric[i] = 1
    if ( ( np.diff(trg_re_r[i , :].reshape(1,100))[0,0] > 5000) & ( np.diff(trg_re_r[i , :].reshape(1,100))[0,0] < 7000) ) :
        t_8_100[1 , :] = trg_re_r[i,:]
        soi_order[i] = 'soi_2'
        soi_order_numeric[i] = 2
    if ( ( np.diff(trg_re_r[i , :].reshape(1,100))[0,0] > 10000) & ( np.diff(trg_re_r[i , :].reshape(1,100))[0,0] < 11000) ) :
        t_8_100[2 , :] = trg_re_r[i,:]
        soi_order[i] = 'soi_3'
        soi_order_numeric[i] = 3
    if ( ( np.diff(trg_re_r[i , :].reshape(1,100))[0,0] > 15000) & ( np.diff(trg_re_r[i , :].reshape(1,100))[0,0] < 20000) ) :
        t_8_100[3 , :] = trg_re_r[i,:]
        soi_order[i] = 'soi_4'
        soi_order_numeric[i] = 4
    if ( ( np.diff(trg_re_r[i , :].reshape(1,100))[0,0] > 30000) & ( np.diff(trg_re_r[i , :].reshape(1,100))[0,0] < 35000) ) :
        t_8_100[4 , :] = trg_re_r[i,:]
        soi_order[i] = 'soi_5'
        soi_order_numeric[i] = 5
    if ( ( np.diff(trg_re_r[i , :].reshape(1,100))[0,0] > 50000) & ( np.diff(trg_re_r[i , :].reshape(1,100))[0,0] < 60000) ) :
        t_8_100[5 , :] = trg_re_r[i,:]
        soi_order[i] = 'soi_6'
        soi_order_numeric[i] = 6
    if ( ( np.diff(trg_re_r[i , :].reshape(1,100))[0,0] > 100000) & ( np.diff(trg_re_r[i , :].reshape(1,100))[0,0] < 110000) ) :
        t_8_100[6 , :] = trg_re_r[i,:]
        soi_order[i] = 'soi_7'
        soi_order_numeric[i] = 7
    if ( ( np.diff(trg_re_r[i , :].reshape(1,100))[0,0] > 170000) & ( np.diff(trg_re_r[i , :].reshape(1,100))[0,0] < 200000) ) :
        t_8_100[7 , :] = trg_re_r[i,:]
        soi_order[i] = 'soi_8'
        soi_order_numeric[i] = 8



########################
#######################
#######################
#######################


# %%    after trigger.

# dataframe
# clm ; columns
# r2s : r2-score (goodness of fit).
# trs : test for response significance (p-value). 
# tmp : template waveform.
# kde : the kde curves for all 8 sois.
# res_mag_8_soi' , 'res_abs_8_soi' , 'base_evt_8_soi' :  respectively : a-b , a , b.
clm = [  
       'Tau_6' , 'A_6' , 'y_fit_6' , 'r2s_6' ,   # last 6 sois.  2 free parameters in the fit function.
       'Tau_6_3p' , 'A_6_3p' , 't0_6_3p' , 'y_fit_6_3p' , 'r2s_6_3p' ,   # last 6 sois. 3 free parameters in the fit function.
       
       'Tau_all' , 'A_all' , 'y_fit_all' , 'r2s_all' ,  # all sois. 2 parameter.
       'Tau_all_3p' , 'A_all_3p' , 't0_all_3p' , 'y_fit_all_3p' , 'r2s_all_3p' ,  # all sois. 3 parameter.
       
       'idx_sig' , 'Tau_sig' , 'A_sig' , 'y_fit_sig' , 'r2s_sig' , # only sois with statistically significant responses. 2 parameter.
       'Tau_sig_3p' , 'A_sig_3p' , 't0_sig_3p' , 'y_fit_sig_3p' , 'r2s_sig_3p' , # only sois with statistically significant responses. 3 parameter.
       
       'pbr_8_soi' , 'cv_ibr' , 'mean_ibr' , 'std_ibr' ,  # pre-inter-block rate : a criterion for stationarity.  
       
       'kde' ,
       'res_mag_8_soi' , 'res_abs_8_soi' , 'base_evt_8_soi' , 'latency_8_soi' , 'window_8_soi' , 'x_lr_ms' , 'hh' ,
       'l_f_8_soi' , 'l_f_8_soi_ms' , 'l_8_100' ,
       'trs' , 'base_rate_8_soi' , 'res_tdm_8_soi'  ,  # statistical comparison.
       
       # errors : fit-6 : 6_sois , sig : incorporating only the significant ones.   2p , 3p : 2 parameter or 3 parameter fit.
       # 0 : pre-set (no error).  1 : error.
       'err_kde' , 'err_fit_all_2p' , 'err_fit_all_3p' , 'err_fit_6_2p' , 'err_fit_6_3p' , 'err_fit_sig_2p' , 'err_fit_sig_3p' , 
]

# initialize data
init_data = np.zeros( ( nva , len(clm) ) )

# initializing the dataframe.
# note : the index will be channel numbers
df = pd.DataFrame( data=init_data , index=np.arange(nva)  , columns=clm )

###########################

# %%

# Means of all vectors (384 elements).
# vec_means = np.array([])
# vec_sds = np.array([])


# vector peaks.
#vec_p_all = []


length_v = ap[:,0].size

# you get them from the combined recording because you want to use the drift-corrected recording, which is combined one.
trace_combined = rd_pps_d.get_traces()    # from the combined recording.
trace_section = trace_combined[ sample_index_beginning : sample_index_end   ,  :  ]    # extracting 1 measurement's trace.

# v = vector = column index = channel number - 1
for v in range(0 , nva , leap ) :  
    
    # mean of sd extracted from the silence periods ( => sd_mean.py in this folder )
    v_mean = mean_sd_total.loc[ v , 'mean']
    v_sd = mean_sd_total.loc[ v , 'sd']
    
    # vec_means = np.append(vec_means , v_mean )
    # vec_sds = np.append(vec_sds , v_sd )
    
    
    #	vector peaks : positive & negative.
    vec_p_pos , di_pos = ss.find_peaks( trace_section[ : , v ] ,  height = (v_mean + (3* v_sd) ) 	)
    vec_p_neg , di_neg = ss.find_peaks( -trace_section[ : , v ] ,  height = (-v_mean + (3* v_sd) ) 	)
    
    vec_p = np.concatenate(( vec_p_pos , vec_p_neg ))
    
    #	vector peaks.
    #vec_p_all.append( vec_p )
    
    
    #   vector peacks _ continuous.
    vec_p_c = np.zeros(length_v)
    vec_p_c[vec_p] = 1
    
    
    ################
    ##################
    
    # t_8_100 : the new trigger (for randomize sois): shape : (8,100). 
    # previous trigger (non-randomied sois) : trig_8_soi seemed to be a list of numpy arrays. since the number of triggers were not equal initially.

    #################
    #################

    #### for the fit plot.
    #### defining the baseline for all 8 sois.

    ####   base_8 : list of n arrays.  each array is the peaks (spikes in the raw trace) of 1 soi during 10 second brfore the start of the train.
    # :  timestamp (or sample number) of each spike.
    # here, 0 need not to be changed to nte. Since the baseline, not the intra-train period is needed.
    base_8=[]
    for i in range(8) :
        base = vec_p[ (vec_p > ( t_8_100[i , 0] - 300000 ) ) & ( vec_p < t_8_100[i , 0] ) ]  # here, 0 need not to be changed to nte. 
        base_8.append(base)
    
    
    # these are possibly not needed.
    ####  base_neb_mean_8 :   base of each of 8 sois / bins / mean / summed into 1 array. 
    ####  here, there’s no need for concatenation of different segments, hence the continuous vector (vec_p_c) is not needed.
    # here, 0 (in [i,0]) need not to be changed to nte. Since the baseline, not the intra-train period is needed.
    
    base_neb_mean_8 = np.array([])  #   this is for plotting the trend of baselines at the bottom of the multi_plot. This is the base in the silent inter-train interval.
    base_tdm_8_soi = [1,1,1,1,1,1,1,1]  #   a list of 8 np arrays.  each np array corresponds to 1 soi baseline. This is used to make a distribution to compare with response to test the significance.
    for i in range(8) :
        base_neb , edges =  np.histogram(base_8[i] , bins=400 , range=( t_8_100[i , 0] - 300000 , t_8_100[i , 0] ))  #   creating a histogram from a non-continuous data (base_8).
        base_neb_tdm , edges =  np.histogram(base_8[i] , bins=100 , range=( t_8_100[i , 0] - 300000 , t_8_100[i , 0] )) # this is for testing the significance.
        base_neb_mean_8 = np.append(base_neb_mean_8 , np.mean(base_neb)  )
        base_tdm_8_soi[i] = base_neb_tdm
    
    
    #  this snippet is modified to contain each bin containing 100ms :   of course for pre-block part.
    # the bin width (for example 25ms) should be constant between the baseline & response distributions.
    # the total number of bins need not to be equal : for example 400 bins and 800 bins. It's like comparing 2 samples with different sample sizes.
    
    
    # for the new stats based on half-heights, it's not needed.
    # this is for testing the response significance :  response part.
    # res_tdm_8_soi  : a list of 8 np arrays.
    # each np array corresponds to 1 soi (response, not baseline).
    # one_stimulus_segment : time stamps of peaks.
    # 100ms = big bin.
    res_tdm_8_soi = [1,1,1,1,1,1,1,1]

    for i in range (8):
        res_tdm_1_soi	= np.array([])		#	response tandem , tandem = not overlapped.
        for j in range (nte , 100) :
            one_stimulus_segment = vec_p[ (vec_p > ( t_8_100[i , j] + 300 ) ) & ( vec_p < (t_8_100[i , j] + 3300 )) ]
            res_tdm_1_soi	 = np.append ( res_tdm_1_soi  , one_stimulus_segment.shape )     #    here, all (not sum of) spike counts in each big bin (100ms) are added together.   
        res_tdm_8_soi[i] = res_tdm_1_soi

    
 

    # not_needed : a junk variable. not needed.
    # pv = p value
    not_needed , pv = mannwhitneyu(base_tdm_8_soi[7] , res_tdm_8_soi[7])


    # if pv < 0.05 :
    #     significance_soi_8 = pv
    # else :
    #     significance_soi_8 = 'not significant'

    
    ####   Must be multiplied to 100 to be compatible with 100 repeatitions.
    base_neb_mean_8_100 = base_neb_mean_8 * 100


    #### rint : rounded to the nearest integer.
    base_neb_mean_8_100_rint = np.rint(base_neb_mean_8_100)


# %% 

    #################
    ###################
    
    #   l_f_8_soi is the common & nuclear step for both psth & fit plots.
    l_f_8_soi = [1,1,1,1,1,1,1,1]
    
    # this is used for creating raster plots.
    # this is for 8 sois * 100 repeatitions / soi  : it's dimension is actually 8 * 101 
    # this could have also been defined as an empty list : l_8_100 = [] instead, but then adding each soi to it should have been done differntly (see below).
    l_8_100 = [1,1,1,1,1,1,1,1]  
 
    
    for h in range(8):
    
        smp = np.zeros(18000)
        for i in t_8_100[h , nte:]   :
            smp = np.vstack( (smp , vec_p_c[ int(i-3000) : int(i+15000) ] ) )
        
    
        l = []
        for j in range(nrt+1):  # +1 : probably because smp has one row of 0s.
            l.append( np.asarray(np.where(smp[j,:]==1)).flatten().tolist() )    #   hence converting a continuous to a discrete	array. This discrete array will be used to make a histogram.
    
        l_f = [j for i in l for j in i]
    
        l_f_8_soi[h] = l_f
        l_8_100[h] = l
    
    
    ###################
    
    c = 600/18000
    d = 100
    
    
    l_f_8_soi_ms = []
    for k in range(8) :
        l_f_8_soi_ms.append(	(np.array(l_f_8_soi[k]))*c - d	)
    
    
    for i in range(8):
        for j in range(nrt+1):
            l_8_100[i][j] = ((np.array(l_8_100[i][j]))*c)-d
    
    
    #################

    
    # %%    parameters + plots.
    
    #   figure : for both psth & fit.
    
    fig = plt.figure(figsize=(17,14) , constrained_layout=True)
    subfigs = fig.subfigures(3,1 , wspace=0.1 , height_ratios=[2,1,1])

    ax_top = subfigs[0].subplots(2,4 , sharex=True , sharey=True)
    ax_r =ax_top.ravel()
    
    ax_bottom = subfigs[1].subplots(1,3)
    ax_3 = subfigs[2].subplots(1,3)

    #############
    ##############
    
    ## response magnitude 
    res_mag_8_soi = np.zeros(8)
    
    ## absolute response.
    res_abs_8_soi = np.zeros(8)
    
    ## baseline activity , pre-event .   evt : event.
    base_evt_8_soi = np.zeros(8)
    
    #  pbr : pre-block rate  [1 minute silence period].
    pbr_8_soi = np.zeros(8)
    
    ####
    
    # xl_ms & xr_ms : left & right x values of the half-height window : 
    # each row is 1 soi : 2 columns corresponding to xl_ms & sr_ms respectively.
    x_lr_ms = np.zeros((8,2)) 
    hh = np.zeros(8)  # half-height : not the abolute value : the value of y on the curve at which the half-height intercepts it.
    
    ####
    
    # not used : intra-period rate normalization.
    # ultimate goal was : hh_rate_8_soi / period_rate_8_soi  _  as a normalization tool.
    
    hh_spk_8_soi = [0,0,0,0,0,0,0,0] # spk : spikes : the number of spikes for each trial inside the hh-window is saved here.
    
    # tns : total number of spikes : inside the window of the half-hight : equals the sum of the above array for each soi.
    hh_tns_8_soi = np.zeros(8)  
    period_tns_8_soi = np.zeros(8)   # period ; the 110 ms period starting from the stimulus onset : to be equal in all sois. 
    
    hh_rate_8_soi = np.zeros(8)  # hh_tns_8_soi / window period for each particular soi.
    period_rate_8_soi = np.zeros(8)  # period_tns_8_soi / 110 ms 
    
    ####
    
    # the width of the response kde.
    window_8_soi = np.zeros(8)
    # latency for all 8 sois.
    latency_8_soi = np.zeros(8)
    
    # the kde curves.
    # since the values for each soi is an array with a different shape, I gather them as a list here.
    kde_8_soi = [ 0 , 0 , 0 , 0 , 0 , 0 , 0 , 0 ]
    
    # these 3 are for the statistical testing.
    # the reason the first 2 are 'lists' is that each entry contains an array of shape '100'.
    res_tdm_8_soi = [1,1,1,1,1,1,1,1]
    base_rate_8_soi = [1,1,1,1,1,1,1,1]
    trs = np.zeros(8)
    


    for i in range(8):
        
        # necf : normalized_to_estimated rate , conversion factor. this is used below for the kernel density estimation.
        # * 1000 : to convert from (/ms) to Hz.
        # x_vk , y_vk  :  vk : variable kernel.
        necf = ( ( l_f_8_soi_ms[i].size ) / nrt ) * 1000 
        
        # variable kernel.
        try :
            y_vk , x_vk , o3 , o4 , o5 , o6 , o7 = opt.ssvkernel( x = l_f_8_soi_ms[i] )
        except ValueError :     #  culprit array : an array of size 0 [an empty array].
            x_vk , y_vk = np.array([]) , np.array([])
        except UnboundLocalError :    # while loop not satisfied (not 'True').  culprit array was : an array of size 3.
            x_vk , y_vk = np.array([]) , np.array([])
        except IndexError :    #  culprit : an array of size 1 (only 1 spike)
            x_vk , y_vk = np.array([]) , np.array([])
        
        # c : converted.
        y_vk_c = y_vk * necf  # deriving the estimated firing rate.
        
        # xy : the coupled (for vectorized operations below) x & y.
        xy_vk = np.vstack(( x_vk , y_vk_c ))
        
        kde_8_soi[i] = xy_vk
        
        # res : response
        res_vk = xy_vk[: ,    (xy_vk[0 , :] > 0 ) & (xy_vk[0 , :] < 100 )   ]  
        # be : baseline relative to event (pre-event).
        be_vk = xy_vk[:  ,    (xy_vk[0 , :] < 0 )   ]
     
        # checking if it is an empty array : this may result 'nan' values or a value error by further operations.
        if res_vk.size==0 :  # you can write : ... res_gk[1,:].size==0 : this makes no difference.
            max_vk = 0
            idx = 0
            latency = 0
        else :
            # maximum of the response period.
            max_vk = np.max(res_vk[1 , :])
            # idx : index of the maximum.
            idx = np.argmax( res_vk[1, :] )
            # latency (ms) of the max_gk defined above, relative to 0 (stimulus onset). 
            latency = res_vk[0,idx]
        
        latency_8_soi[i] = latency
        
        # m : mean of baseline.
        if be_vk.size == 0 :
            be_vk_m = 0
        else :
            be_vk_m = np.mean(be_vk[1,:])
        
        res_mag = max_vk - be_vk_m
        res_mag_8_soi[i] = res_mag
        
        res_abs_8_soi[i] = max_vk
        base_evt_8_soi[i] = be_vk_m
        
        ########
        
        # half-heights : x values 
        
        # input of the next step should be an array (not an integer).
        idx_a = np.array([idx])
        
        # xl , xr : interpolated indices , in ms.
        # rel_height = 1 : calculates the width at the base of the peak.
        if res_vk[1, :].size == 0 :
            width , height , xl , xr = np.array([0]) , np.array([0]) , np.array([0]) , np.array([0])
        else :
            width , height , xl , xr = ss.peak_widths( res_vk[1, :] , idx_a , rel_height=0.5 )
        
        # i : integer : it's still the index of the kde curve.
        xl_i = int(xl[0])
        xr_i = int(xr[0])
        
        # converting the index to ms.
        if res_vk[1, :].size == 0 :
            xl_ms , xr_ms = 0 , 0
        else :
            xl_ms = res_vk[0 , xl_i]
            xr_ms = res_vk[0 , xr_i]
        
        
        x_lr_ms[i , 0] = xl_ms
        x_lr_ms[i , 1] = xr_ms
        
        hh[i] = height  # half-height.
        
        # window : the period in x axis between the half-heights.
        # this can also be derived from the 'width' variable above.
        window = xr_ms - xl_ms
        window_8_soi[i] = window
        
        
        # converting ms to samples.
        # note : these x distances start from time 0 (stimulus onset).
        xl_sa = int(xl_ms * 30)
        xr_sa = int(xr_ms * 30)

        ###########
        
        #onb , o2,o3,o4,o5 = opt.sshist(x=l_f_8_soi_ms[i])     # this is for optimal bin size.
        ax_r[i].plot( x_vk , y_vk_c , linewidth=3 )
        ax_r[i].hlines( height , xl_ms , xr_ms , color='m' )
        
        #############
    
        ax_ep = ax_r[i].twinx()  #  ep : event plot  (raster).
        ax_ep.eventplot(l_8_100[i] , linewidths=0.75 , linelengths=0.75 , colors='k')
    
        ax_r[i].axvline(x=0 , color='k' )
    
    #############
    
    # %% 
    
    ###############
    ###############

    # this is for testing the response significance :  response part.
    # res_tdm_8_soi  : a list of 8 np arrays.
    # each np array corresponds to 1 soi (response, not baseline).
    # one_stimulus_segment : time stamps of peaks.
    # 100ms = big bin.

        res_tdm_1_soi	= np.array([])		#	response tandem , tandem = not overlapped. rate based (per trial) : for the statistical comparison. 
        hh_spk_1_soi = np.zeros(nrt)  # used for measuring adaptation trend per soi. 
        hh_tns_1_soi = np.array([]) # not used : for intra-period rate normalization.
        for j in range (nte,100) :
            one_trial_response_spikes = vec_p[ (vec_p >= ( t_8_100[i , j] + xl_sa  ) ) & ( vec_p <= (t_8_100[i , j] + xr_sa )) ]
            hh_spk_1_soi[j-nte] = one_trial_response_spikes.size
            # .size : number of spikes in that period.
            if window == 0 :
                one_trial_response_rate = 0
            else :
                one_trial_response_rate = one_trial_response_spikes.size / window
            # here, all (not sum of) spike counts in each big bin (100ms) are added together. 
            res_tdm_1_soi	 = np.append ( res_tdm_1_soi  , one_trial_response_rate )     
        res_tdm_8_soi[i] = res_tdm_1_soi
        hh_spk_8_soi[i] = hh_spk_1_soi
    
    ######
    
        base_rate_1_soi	= np.array([])		#	response tandem , tandem = not overlapped.
        for j in range (nte,100) :
            # here, all (not sum of) spike counts in each big bin (100ms) are added together. 
            one_trial_base_spikes = vec_p[ (vec_p > ( t_8_100[i , j] - 3000 ) ) & ( vec_p < t_8_100[i , j] ) ]
            # .size : number of spikes in that period.
            one_trial_base_rate = one_trial_base_spikes.size / 100   # 100 : 100ms : the unit (response rate ? (window)) is ms since in reponse it's also ms.
            base_rate_1_soi	 = np.append ( base_rate_1_soi  , one_trial_base_rate )     
        base_rate_8_soi[i] = base_rate_1_soi


# %% 

    ######
    
        # her the pre-block period is the preferred terminolgy ; because of the 1st soi. Later the other periods will be named inter-block.
        pre_block_spikes = vec_p[ (vec_p > ( t_8_100[i , 0] -  1800000 ) ) & ( vec_p < t_8_100[i , 0] ) ]
        pre_block_rate = pre_block_spikes.size / 60   # 60 s : the 1 minute silence interval.
        pbr_8_soi[i] = pre_block_rate
    
    ######
    
        # 'not_needed' : a junk variable. not needed.
        # pv = p value
        # trs : test for response significance
        # the 'if' statement is needed if there is no single spike in all 100 repeatitions of 1 soi, in the corresponding time periods.
    
        if res_tdm_8_soi[i].size == 0 :
            pv = 1
        elif base_rate_8_soi[i].size == 0 :	# due to stepping from the top, res.size here is not 0.
            	pv = 0
        else :
            try :
                not_needed , pv = wilcoxon( base_rate_8_soi[i] , res_tdm_8_soi[i] , alternative='less')
            except ValueError :  #  => DELL / analysis / stat / stat.docx for the details.
                pv = 1
        
        trs[i] = pv
    
    ##########


    # %% 

    # ibr : inter-block rate.
    std_ibr = np.std(pbr_8_soi[1:])
    mean_ibr = np.mean(pbr_8_soi[1:])
    cv_ibr = std_ibr  /  mean_ibr   # cv : coefficient of variation.

# %%

    # fit for normalized response.
    # 6 : 6 sois : from soi_3 onwards :  soi_1 & soi_2 are omitted due to the overlapping of response on baseline.
    # respecting pre-event baseline.
    # nan values were converted above.
    try :
        popt_6, pcov_6 = curve_fit(fit_func, sois[2:] , res_mag_8_soi[2:] )
    except RuntimeError :
        popt_6 = np.array([ 0.5 , 0 ])
        pcov_6 = 0
        df.loc[ v , 'err_fit_6_2p' ] = 1
    
    
    # t0 as a free parameter (instead of being sd (stimulus duration) as before).
    try :
        popt_6_3p, pcov_6_3p = curve_fit(fit_func_3p, sois[2:] , res_mag_8_soi[2:] )
    except RuntimeError :
        popt_6_3p = np.array([ 0.5 , 0 , 0.05 ])
        pcov_6_3p = 0
        df.loc[ v , 'err_fit_6_3p' ] = 1
    
    
    
    # y of the fitted curve based on the formerly derived parmeters (popt)
    y_fit_6 = fit_func(sois[2:] , *popt_6)
    y_fit_6_3p = fit_func_3p(sois[2:] , *popt_6_3p)
    
    # r2_score : goodness of fit.
    r2s_6 = r2_score( res_mag_8_soi[2:] , y_fit_6 )
    r2s_6_3p = r2_score( res_mag_8_soi[2:] , y_fit_6_3p )
    
    ##################
    
    # all : for all sois.
    # fit for absolute responses.
    try :
        popt_all , pcov_all = curve_fit(fit_func, sois , res_abs_8_soi )
    except RuntimeError :
        popt_all= np.array([ 0.5 , 0 ])
        pcov_all = 0
        df.loc[ v , 'err_fit_all_2p' ] = 1
    
    
    try :
        popt_all_3p , pcov_all_3p = curve_fit(fit_func_3p, sois , res_abs_8_soi , method='lm' )
    except RuntimeError :
        popt_all_3p = np.array([ 0.5 , 0 , 0.05 ])
        pcov_all_3p = 0
        r2s_all_3p = 0
        df.loc[ v , 'err_fit_all_3p' ] = 1
    
    # the other fitting method
    try :
        popt_all_3p_dogbox , pcov_all_3p_dogbox = curve_fit(fit_func_3p, sois , res_abs_8_soi , method='dogbox' )
    except RuntimeError :
        popt_all_3p_dogbox = np.array([ 0.5 , 0 , 0.05 ])
        pcov_all_3p_dogbox = 0
        r2s_all_3p_dogbox = 0
        # df.loc[ v , 'err_fit_all_3p' ] = 1
    

    # y of the fitted curve based on the formerly derived parmeters (popt)
    y_fit_all = fit_func(sois , *popt_all)
    
    y_fit_all_3p = fit_func_3p(sois , *popt_all_3p)
    y_fit_all_3p_dogbox = fit_func_3p(sois , *popt_all_3p_dogbox)
    
    
    # r2_score : goodness of fit.
    r2s_all = r2_score( res_abs_8_soi , y_fit_all )
    
    r2s_all_3p = r2_score( res_abs_8_soi , y_fit_all_3p )
    r2s_all_3p_dogbox = r2_score( res_abs_8_soi , y_fit_all_3p_dogbox )
    
    
    # here, it compares the results of the 2 fitting methods (lm & dogbox) & selects the one with a better fit. 
    if r2s_all_3p_dogbox > r2s_all_3p :
        popt_all_3p = popt_all_3p_dogbox
        pcov_all_3p = pcov_all_3p_dogbox
        y_fit_all_3p = y_fit_all_3p_dogbox
        r2s_all_3p = r2s_all_3p_dogbox
        df.loc[ v , 'err_fit_all_3p' ] = 0
    else :
        pass # this means : don't do anything !
    
    #########
    
    # here, only sois with statistically significant responses are fitted to the function.
    
    # index of sois with significant responses.
    # [0] @ the end of it : because the output is a tuple. The 1st element is a numpy array.
    idx_sig = np.where(trs<0.05)[0]
    
    if idx_sig.size < 2 :
        popt_sig = np.array([ 0.5 , 0 ])
        popt_sig_3p = np.array([ 0.5 , 0 , 0.05 ])  # t0 is put 0.05 so that it would be similar to the original function : no additional error.
        
        pcov_sig = 0
        pcov_sig_3p = 0
        
        df.loc[ v , 'err_fit_sig_2p' ] = 1
        df.loc[ v , 'err_fit_sig_3p' ] = 1
    
    elif idx_sig.size < 3 :
        try :
            popt_sig , pcov_sig = curve_fit( fit_func, sois[idx_sig] , res_mag_8_soi[idx_sig] ) # a-b according to Michale's suggestion.
        except RuntimeError :
            popt_sig = np.array([ 0.5 , 0 ])
            pcov_sig = 0
            df.loc[ v , 'err_fit_sig_2p' ] = 1
        
        popt_sig_3p = np.array([ 0.5 , 0 , 0.05 ])
        pcov_sig_3p = 0
        df.loc[ v , 'err_fit_sig_3p' ] = 1
        
    else :
        try :
            popt_sig , pcov_sig = curve_fit( fit_func, sois[idx_sig] , res_mag_8_soi[idx_sig] )
        except RuntimeError :
            popt_sig = np.array([ 0.5 , 0 ])
            pcov_sig = 0
            df.loc[ v , 'err_fit_sig_2p' ] = 1
        
        try :
            popt_sig_3p , pcov_sig_3p = curve_fit( fit_func_3p , sois[idx_sig] , res_mag_8_soi[idx_sig] )
        except RuntimeError :
            popt_sig_3p = np.array([ 0.5 , 0 , 0.05 ])
            pcov_sig_3p = 0
            df.loc[ v , 'err_fit_sig_3p' ] = 1
    
    
    # y of the fitted curve based on the formerly derived parmeters (popt)
    y_fit_sig = fit_func(sois , *popt_sig)
    y_fit_sig_3p = fit_func_3p(sois , *popt_sig_3p)
    
    
    # r2_score : goodness of fit.
    if idx_sig.size < 2 :  # r2_score needs a minimum amount of input data to function.
        r2s_sig = 0
        r2s_sig_3p = 0
    else :
        r2s_sig = r2_score( res_mag_8_soi[idx_sig] , y_fit_sig[idx_sig] )
        r2s_sig_3p = r2_score( res_mag_8_soi[idx_sig] , y_fit_sig_3p[idx_sig] )
    
# %%
    
    df.loc[ v , 'A_6' ] = popt_6[0]
    df.loc[ v , 'Tau_6' ] = popt_6[1]    
    df.loc[ v , 'r2s_6' ] = r2s_6
    df.loc[ v:v , 'y_fit_6' ] = pd.Series(data=[y_fit_6] , index=[v])
    
    #######
    
    df.loc[ v , 'A_6_3p' ] = popt_6_3p[0]
    df.loc[ v , 'Tau_6_3p' ] = popt_6_3p[1]    
    df.loc[ v , 't0_6_3p' ] = popt_6_3p[2]
    df.loc[ v , 'r2s_6_3p' ] = r2s_6_3p
    df.loc[ v:v , 'y_fit_6_3p' ] = pd.Series(data=[y_fit_6_3p] , index=[v])
    
    #######
    #######
    
    df.loc[ v , 'A_all' ] = popt_all[0]
    df.loc[ v , 'Tau_all' ] = popt_all[1]    
    df.loc[ v , 'r2s_all' ] = r2s_all
    df.loc[ v:v , 'y_fit_all' ] = pd.Series(data=[y_fit_all] , index=[v])
    
    #######
    
    df.loc[ v , 'A_all_3p' ] = popt_all_3p[0]
    df.loc[ v , 'Tau_all_3p' ] = popt_all_3p[1]  
    df.loc[ v , 't0_all_3p' ] = popt_all_3p[2]
    df.loc[ v , 'r2s_all_3p' ] = r2s_all_3p
    df.loc[ v:v , 'y_fit_all_3p' ] = pd.Series(data=[y_fit_all_3p] , index=[v])
    
    #######
    #######
    
    df.loc[ v:v , 'idx_sig' ] = pd.Series(data=[idx_sig] , index=[v])
    df.loc[ v , 'A_sig' ] = popt_sig[0]
    df.loc[ v , 'Tau_sig' ] = popt_sig[1]    
    df.loc[ v , 'r2s_sig' ] = r2s_sig
    df.loc[ v:v , 'y_fit_sig' ] = pd.Series(data=[y_fit_sig] , index=[v])
    
    #######
    
    df.loc[ v , 'A_sig_3p' ] = popt_sig_3p[0]
    df.loc[ v , 'Tau_sig_3p' ] = popt_sig_3p[1]  
    df.loc[ v , 't0_sig_3p' ] = popt_sig_3p[2]
    df.loc[ v , 'r2s_sig_3p' ] = r2s_sig_3p
    df.loc[ v:v , 'y_fit_sig_3p' ] = pd.Series(data=[y_fit_sig_3p] , index=[v])
    
    ###########
    ###########
    
    df.loc[ v:v , 'pbr_8_soi' ] = pd.Series(data=[pbr_8_soi] , index=[v])
    df.loc[ v , 'cv_ibr' ] = cv_ibr
    df.loc[ v , 'mean_ibr' ] = mean_ibr
    df.loc[ v , 'std_ibr' ] = std_ibr
    
    ########
    
    df.loc[ v:v , 'trs' ] = pd.Series(data=[trs] , index=[v])
    df.loc[ v:v , 'base_rate_8_soi' ] = pd.Series(data=[base_rate_8_soi] , index=[v])
    df.loc[ v:v , 'res_tdm_8_soi' ] = pd.Series(data=[res_tdm_8_soi] , index=[v])
    
    df.loc[ v:v , 'res_mag_8_soi' ] = pd.Series(data=[res_mag_8_soi] , index=[v])
    df.loc[ v:v , 'res_abs_8_soi' ] = pd.Series(data=[res_abs_8_soi] , index=[v])  
    df.loc[ v:v , 'base_evt_8_soi' ] = pd.Series(data=[base_evt_8_soi] , index=[v])
    df.loc[ v:v , 'latency_8_soi' ] = pd.Series(data=[latency_8_soi] , index=[v])
    df.loc[ v:v , 'x_lr_ms' ] = pd.Series(data=[x_lr_ms] , index=[v])
    df.loc[ v:v , 'hh' ] = pd.Series(data=[hh] , index=[v])
    df.loc[ v:v , 'window_8_soi' ] = pd.Series(data=[window_8_soi] , index=[v])
    
    df.loc[ v:v , 'kde' ] = pd.Series(data=[kde_8_soi] , index=[v])
    
    df.loc[ v:v , 'l_f_8_soi' ] = pd.Series(data=[l_f_8_soi] , index=[v])
    df.loc[ v:v , 'l_f_8_soi_ms' ] = pd.Series(data=[l_f_8_soi_ms] , index=[v])
    df.loc[ v:v , 'l_8_100' ] = pd.Series(data=[l_8_100] , index=[v])

# %%

    ax_r[0].axvline(x=-110 , color='k')
    ax_r[0].axvline(x=110 , color='k')
    ax_r[0].axvline(x=220 , color='k')
    ax_r[0].axvline(x=330 , color='k')
    ax_r[0].axvline(x=440 , color='k')
    ax_r[0].axvline(x=550 , color='k')
    
    
    ax_r[1].axvline(x=195 , color='k')
    ax_r[1].axvline(x= 390 , color='k')
    
    ax_r[2].axvline(x=345 , color='k')
    
    
    for i in range(8):
        ax_r[i].set_xlim( -100 , 500 )
    
    # this highlights the frame of the plots with significant responses.
    for i in idx_sig :
        ax_r[i].spines[:].set_color('blue')
        ax_r[i].spines[:].set_linewidth(4)
    
    ##################
    
    
    ax_r[7].set_xlabel('time in ms')
    ax_r[0].set_ylabel('KDE' , fontsize=9) 
    
    
    ax_r[0].set_title('soi_1 = 110 ms')
    ax_r[1].set_title('soi_2 = 195 ms')
    ax_r[2].set_title('soi_3 = 345 ms')
    ax_r[3].set_title('soi_4 = 611 ms')
    ax_r[4].set_title('soi_5 = 1.081 s')
    ax_r[5].set_title('soi_6 = 1.914 s')
    ax_r[6].set_title('soi_7 = 3.388 s')
    ax_r[7].set_title('soi_8 = 6 s')

    
    ########################
    ################

    #   fit plot.
    #  here, the main solid plot incoroporates the baseline before events, not the baseline during the 1min silence interval.

    #   fit plot.
    #  here, the main solid plot incoroporates the baseline before events, not the baseline during the 1min silence interval.

    
    ########################
    ################

    #   fit plot for all sois.
    #  here, the main solid plot incoroporates the baseline before events, not the baseline during the 1min silence interval.
    # fit_2p & _3p : 2 parameters (A & Tau) or 3 parameters (A , Tau , t0).

    ax_bottom[0].plot(sois , res_abs_8_soi , linestyle='solid' , color='k' , label='actual response')
    ax_bottom[0].plot(sois , y_fit_all , linestyle='dotted' , color='k' , 
                      label='fit_2p: A_all=' + str(np.around( popt_all[0], decimals=3)) + 
                      ' , τ_all=' + str(np.around( popt_all[1], decimals=3)) )
    
    ax_bottom[0].plot(sois , y_fit_all_3p , linestyle='dashdot' , color='k' , 
                      label='fit_3p: A_all_3p=' + str(np.around( popt_all_3p[0], decimals=3)) + 
                      ' , τ_all_3p=' + str(np.around( popt_all_3p[1], decimals=3))  + 
                      ' , t0_all_3p=' + str(np.around( popt_all_3p[2], decimals=3)) )

    ax_bottom[0].set_xticks(ticks=sois)
    ax_bottom[0].tick_params(axis='x' , labelrotation=90 , labelsize=6)
    ax_bottom[0].set_xlabel('soi(ms)' , loc='right')
    ax_bottom[0].set_title('all sois : actual response _ fit \n r2_score all_sois_2p : ' + 
                           str(np.around(r2s_all , decimals=2)) +
                           '  __  r2_score all_sois_3p : ' + str(np.around(r2s_all_3p , decimals=2))
                           , fontsize=9)
    
    ax_bottom[0].legend( fontsize=8 )  
    
    ################

    #  fit plot _ 6 sois.
    #  here, the main solid plot incoroporates the baseline before events, not the baseline during the 1min silence interval.

    ax_bottom[1].plot(sois[2:] , res_mag_8_soi[2:] , linestyle='solid' , color='k' , label='normalized response')
    ax_bottom[1].plot(sois[2:] , y_fit_6 , linestyle='dotted' , color='k' , 
                      label='fit_2p : A_6=' + str(np.around( popt_6[0], decimals=3)) + 
                      ' , τ_6=' + str(np.around( popt_6[1], decimals=3)) )
    
    ax_bottom[1].plot(sois[2:] , y_fit_6_3p , linestyle='dashdot' , color='k' , 
                      label='fit_3p : A_6_3p=' + str(np.around( popt_6_3p[0], decimals=3)) + 
                      ' , τ_6_3p=' + str(np.around( popt_6_3p[1], decimals=3))  + 
                      ' , t0_6_3p=' + str(np.around( popt_6_3p[2], decimals=3)) )

    ax_bottom[1].set_xticks(ticks=sois[2:])
    ax_bottom[1].tick_params(axis='x' , labelrotation=90 , labelsize=6)
    ax_bottom[1].set_xlabel('soi(ms)' , loc='right')
    ax_bottom[1].set_title('6 sois : normalized response respecting pre_event _ fit \n r2_score 6_soi_2p: ' + 
                           str(np.around(r2s_6 , decimals=2)) +
                           '  __  r2_score 6_soi_3p: ' + str(np.around(r2s_6_3p , decimals=2)) 
                           , fontsize=9)
    
    ax_bottom[1].legend( fontsize=8 )  
    
    ################

    #   fit plot for significant sois.
    #  here, the main solid plot incoroporates the baseline before events, not the baseline during the 1min silence interval.
    # TypeError : The number of func parameters must not exceed the number of data points.
    
    empty = np.array([])  # this is only to automatically add a text as 'label' in the plot.
    
    ax_bottom[2].plot(sois[idx_sig] , res_abs_8_soi[idx_sig] , linestyle='solid' , color='k' , label='actual significant responses')
    if idx_sig.size < 2 :
        ax_bottom[2].plot( empty , color='w' , label='number of significant responses < 2' ) # I made the line legend color invisible : color='w'.
    elif idx_sig.size < 3 :
        ax_bottom[2].plot(sois , y_fit_sig , linestyle='dotted' , color='k' , 
                      label='fit_2p: A_sig=' + str(np.around( popt_sig[0], decimals=3)) + 
                      ' , τ_sig=' + str(np.around( popt_sig[1], decimals=3)) )
        ax_bottom[2].plot( empty , color='w' , label='number of significant responses = 2 : no 3-parameter calculation.' )
    else :
        ax_bottom[2].plot(sois , y_fit_sig , linestyle='dotted' , color='k' , 
                      label='fit_2p: A_sig=' + str(np.around( popt_sig[0], decimals=3)) + 
                      ' , τ_sig=' + str(np.around( popt_sig[1], decimals=3)) )
        ax_bottom[2].plot(sois , y_fit_sig_3p , linestyle='dashdot' , color='k' , 
                      label='fit_3p: A_sig_3p=' + str(np.around( popt_sig_3p[0], decimals=3)) + 
                      ' , τ_sig_3p=' + str(np.around( popt_sig_3p[1], decimals=3))  + 
                      ' , t0_sig_3p=' + str(np.around( popt_sig_3p[2], decimals=3)) )

    ax_bottom[2].set_xticks(ticks=sois)
    ax_bottom[2].tick_params(axis='x' , labelrotation=90 , labelsize=6)
    ax_bottom[2].set_xlabel('soi(ms)' , loc='right')
    ax_bottom[2].set_title('significant sois : normalized significant responses respecting pre_event _ fit \n r2_score sig_2p : ' + 
                           str(np.around(r2s_sig , decimals=2)) + 
                           '  __  r2_score sig_3p : ' + str(np.around(r2s_sig_3p , decimals=3)) , 
                           fontsize=9)
    
    ax_bottom[2].legend( fontsize=8 )   
    
    ###################
    ###################
    
    # here, since I don't know if a total 1 minute silence period existed, before the first soi, I exclude the 1st pre-block silence period.
    ax_3[0].plot(pbr_8_soi[1:])
    ax_3[0].set_title(
        'inter_block rate (Hz) \n cv : ' + str(np.around( cv_ibr ,  decimals=3)) +  
        '  _  mean : ' + str(np.around( mean_ibr ,  decimals=3)) +
        '  _  std : ' + str(np.around( std_ibr ,  decimals=3)) , 
        fontsize=11 )
    
    ax_3[0].set_xticks(ticks=ticks_pbr , labels=labels_pbr )
    ax_3[0].set_xlabel('inter-block interval' , loc='right')
    
    
    ###################  
    
    ax_3[2].text(0.1 , 0.2 , 'test for response significance, p-value : \n\nsoi-1 : ' + 
                       str(np.around(trs[0], decimals=3)) + '\nsoi-2 : ' + 
                       str(np.around(trs[1], decimals=3)) + '\nsoi-3 : ' + 
                       str(np.around(trs[2], decimals=3)) + '\nsoi-4 : ' + 
                       str(np.around(trs[3], decimals=3)) + '\nsoi-5 : ' + 
                       str(np.around(trs[4], decimals=3)) + '\nsoi-6 : ' + 
                       str(np.around(trs[5], decimals=3)) + '\nsoi-7 : ' + 
                       str(np.around(trs[6], decimals=3)) + '\nsoi-8 : ' + 
                       str(np.around(trs[7], decimals=3)) , 
                       fontsize=12)
    
    ############
    
    fig.suptitle(
        'v_' +  str(v)  + '  -  channel ' +  str(v+1) + '\n' + description_session + 
        '  _  soi order : ' + str(soi_order_numeric) , 
        fontsize=12)
    #plt.gcf().text(0.02, 0.9, 'mean = ' + str(np.around(v_mean , decimals=1)) + '\n sd = ' + str(np.around(v_sd , decimals=1)) , fontsize=14)
    
    ##############
    ##################
    
    plt.savefig( dest_dir + '/v_' +  str(v)  + '.pdf' )
    plt.close()


# %%
####################

# this is for merging files with leaped channels : but it dows not merge them with spatial order.
#   plt.savefig( r'D:\analysis _ rec\2022-2-7\1\all vectors\v_' +  str(v)  + '.svg' )
    

# #   keep this below line in this program (don't move t to the REPL Ipython console).
# #   otherwise, when creating new pdfs, the new ones will be appended to the old ones !
# mergedObject = PdfFileMerger()

# # this has been changed to attach pdfs according to the depth oder of the electrodes.
# for fileNumber in range(0 , nva , leap ) :
#     mergedObject.append(PdfFileReader( dest_dir + '/v_' + str(fileNumber) + '.pdf'))


# mergedObject.write( dest_dir + '/total.pdf')


# %%

# this is for merging files taking the spatial order into consideration.
    # but it dows not handle leaped channels : all channles should have been analyzed.

#   keep this below line in this program (don't move t to the REPL Ipython console).
#   otherwise, when creating new pdfs, the new ones will be appended to the old ones !
mergedObject = PdfFileMerger()

#   change the range correspnding to the channels you think are catching signals.
for fileNumber in ( list(range(0,384, 2)) + list(range(1,384, 2)) ):
    mergedObject.append(PdfFileReader( dest_dir + '/v_' + str(fileNumber) + '.pdf'))

mergedObject.write( dest_dir + '/total.pdf')

# %%

# mu ; multi-units
df.to_pickle( dest_dir + '/df_mu.pkl')

# %%
#####
# => pre_req_n.py    ( analysis / general / pre_req_n.py  )

