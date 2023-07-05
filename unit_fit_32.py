
# this program is in  /home/azare/groups/PrimNeu/Aryo/analysis/fit
# env_2     ( environment in the server )
# this plots & analyzes the units.
# the immediate program before this is pipeline_n .docx  or pipe_sort_n.py : spike sorting program.


# 1st :
# create a new destination folder for saving the outputs : corresponds to 'dest_dir' here.
# pre-req_n.py :
#   run the function & the soi array.
#   calculate : sample_correction.
# adjust, in here :
#    the 4 variables defined below.
#    stream_id : for the trigger : corresponding to the hemiphere.
# copy the changeable lines below to  Dell \ D:\address \ file_fit  .docx



################
################
################
################

# the source directory for the sorting objects ( = dest_dir in the pipe_sort_n.py file).
#  this is a combined recording directory.
source_dir = r'/home/azare/groups/PrimNeu/Aryo/analysis/sort/Lucy/p2'

# the directory for extracting the triggers.
# this is a single (not combined) recording directory.
# you can copy it from pre_req_n.py
directory = r'/home/azare/groups/PrimNeu/Aryo/copy_data/Lucy_20221219/2022-12-19 _ Lucy _ terminal/P2/6/2022-12-20_02-18-35'

# you should 1st create this in windows explorer.
# for pdfs & the database.
# this is also logically a single (not combined) recording directory.
dest_dir = r'/home/azare/groups/PrimNeu/Aryo/analysis/Lucy/p2/6'

##########

# this will be printed at sup-title.
description_session = 'Lucy_terminal , right hemisphere _ p2_6_R (primate probe) _ unit _ tone 440 Hz '

##############
##############
##############
##############

# nte : umber of trials (out of 100) to exclude.
nte = 3

# nrt : number of remaining trials.
nrt = 100 - nte

##############

# for the trigger.
# this is the number of samples from the sum of previous recordings that must be added to the current recording.
# you may 1st calculate this in pre_req.py
# sample_correction = 

##############
##############
##############
##############

# loading the unit information :
# this is only used here to plot isi distributions.
# in this version of spike-interface (0.97.1), installed in env_17 (sorting environment), 
#   the output of sorting has a different path structure :
#   '/sorter_output' is added @ the end of the directory.
srt = se.read_spykingcircus( source_dir + '/srt/sorter_output' )

#      load the unit spike trains that have been extracted previously by sorting :
#      this contains all units (not filtered by quality metrics).
#      skt_dm : spike-train dimensionalized.

## ::
    
skt_dm = np.load( source_dir + '/skt_dm.npy')  #  this is the spike train (skt) of all units.
#      load the unit ids of the filtered (after quality metrics) waveform object :
unit_id = np.load( source_dir + '/unit_id.npy' )  # this is virtually replacing the channels (mua).

## these are needed for plotting the waveforms :

wfe_c = si.WaveformExtractor.load_from_folder( source_dir + '/wfe_c')
## this is a bit time consuming :
eci = post.get_template_extremum_channel(wfe_c , outputs='index')  #  this is needed to plot the template waveform.


# cul : compute_unit_locations
cul = post.compute_unit_locations(wfe_c , outputs='by_unit')

###############

# # quality metrics _ c : curated units  _  a : all units.   
qm_c = pd.read_pickle(source_dir + '/qm_c.pkl')
qm_a = pd.read_pickle(source_dir + '/qm_a.pkl')

# qm = pd.read_hdf( source_dir + '/qm.h5' )
# qm = pd.read_csv( source_dir + '/qm.csv' )

# # template metrics
tm = pd.read_pickle(source_dir + '/tm.pkl')
# tm = pd.read_hdf( source_dir + '/tm.h5' )
# tm = pd.read_csv( source_dir + '/tm.csv' )

###############

# x axis of the template waveform plot.
x_template = np.linspace(-1 , 1 , 60 )

# pre-block rate.
ticks_pbr = np.arange(7)  #  tick positions : generally automatically set, but specifically needed if you want to set the labels (next line).
labels_pbr = np.arange(1,8)

################
################
################
################



#   change in this page : 
    # input directory
    # output directories ( 4 : individual pdfs , db , pdf_merge (2)) _ 
    # suptitle :  date , click_tone .
#   create a new destination folder ('mix').
#   nmz in file's name = normalized.
#   re_sig in file's name = response significance.
# also change the baseline for short sois.


#################
# tdm : tandem : this naming is a vestigial remanant from 

###############

# changes in this version : bottom subfigure : adding another trace : base events as measured before each event (base_event_8).
#  changes in this version ; instead  subtracting 1 minute silence interval baseline from the max response : here the baseline before the event is subtracted 

# in modules . py  :
# from open_ephys.analysis import Session
# from scipy.optimize import curve_fit


# def fit_func(SOI, A, tau_0):
#     return A*(1-np.exp(-(SOI-0.05)/tau_0))

# sois = np.array([ 0.11 , 0.195 , 0.345 , 0.611 , 1.081 , 1.914 , 3.388 , 6])


# %%
################

# this is taken from a single (not combined) recording
# the trigger smaple will then be corrected to match with the combined recording.
session = Session(directory)
rec = session.recordnodes[0].recordings[0]
ap = rec.continuous[0].samples  # numpy.memmap _ memory mapped array _ change the index according to the stream you want to choose.


####################

#	triggers.
trg = ap[: , 384]

t , di_t = ss.find_peaks( trg , plateau_size=20)

trg_re = di_t['left_edges']

# broadcasting : all array elements will be added with 'sample_correction'.
trg_re = trg_re + sample_correction

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



# Means of all vectors (384 elements).
# vec_means = np.array([])
# vec_sds = np.array([])


# vector peaks.
#vec_p_all = []


# nsacr : number of samples in the whole (all of the) combined recording.   =>  pre_req_n.py
length_v = nsacr

###########################

# dataframe
# clm ; columns
# r2s : r2-score (goodness of fit).
# trs : test for response significance (p-value). 
# tmp : template waveform.
# kde : the kde curves for all 8 sois.
# res_mag_8_soi' , 'res_abs_8_soi' , 'base_evt_8_soi' :  respectively : a-b , a , b.
clm = [ 
       'location' , 'ext_ch' , 
       'tmp' ,  
       
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
init_data = np.zeros( (unit_id.size , len(clm) ) )

# initializing the dataframe.
# note : the index will be unit_ids
df = pd.DataFrame( data=init_data , index=unit_id  , columns=clm )

###########################


# v = vector = column index = channel number - 1
for v in unit_id :

    
    # v_mean = np.mean(ap[:100000 , v ])
    # v_sd = np.std(ap[:100000 , v ])
    
    # db[v , 3] = v_mean
    # db[v , 4] = v_sd
    
    # # vec_means = np.append(vec_means , v_mean )
    # # vec_sds = np.append(vec_sds , v_sd )
    
    
    # #	vector peaks : positive & negative.
    # vec_p_pos , di_pos = ss.find_peaks( ap[ : , v ] ,  height = (v_mean + (3* v_sd) ) 	)
    # vec_p_neg , di_neg = ss.find_peaks( -ap[ : , v ] ,  height = (-v_mean + (3* v_sd) ) 	)
    
    vec_p = skt_dm[0 , np.where( skt_dm[1,:]==v )].ravel().astype(int)  # astype(int) : to convert (cast) the float output to integer.
    
    #	vector peaks.
    #vec_p_all.append( vec_p )
    
    
    #   vector peacks _ continuous.
    vec_p_c = np.zeros(length_v)
    vec_p_c[vec_p] = 1


    # %%
    #################
    ###################
    
    # after the operations smp will be of dimensions (100 * 18000) & is similar to f in the tuning curve analysis.
    #   l_f_8_soi is the common & nuclear step for both psth & fit plots.
    l_f_8_soi = [1,1,1,1,1,1,1,1]
    
    
    # this is used for creating raster plots.
    # this is for 8 sois * 100 repeatitions / soi  : it's dimension is actually 8 * 101 
    # this could have also been defined as an empty list : l_8_100 = [] instead, but then adding each soi to it should have been done differntly (see below).
    l_8_100 = [1,1,1,1,1,1,1,1]  
    
    for h in range(8):
    
        smp = np.zeros(18000)
        for i in t_8_100[h , nte:]   :  # 5 : eliminating the 1st 5 stimuli to eliminate the initial sensitivity after the 1 minute interval.
            smp = np.vstack( (smp , vec_p_c[ int(i-3000) : int(i+15000) ] ) )
        
    
        l = []  #  this is for 1 soi.   len(l) : 101 .  each element is a list of all spikes (rasters) corresponding to 1 repeatition.
        
        for j in range(nrt+1):  # 96 = 95 stimuli + 1 zeros array.
            l.append( np.asarray(np.where(smp[j,:]==1)).flatten().tolist() )    #   hence converting a continuous to a discrete	array. This discrete array will be used to make a histogram.
    
        
    
        # l_f : f : flattened.
        # this part was not done in the tuning curve analysis. 
        # This flattens all 100 repeatitions of 1 soi into 1 array or list.
        # the reason is that unlike the tuning curve or other raster type of analysis which needs a line of rasters for each bip (sound stimulus) ; 
        # here, a pool of all rasters in the time window of histogram (-100-500 ms) is needed.
        # hence, all rasters are vertically collapsed here.
        l_f = [j for i in l for j in i]     
    
        l_f_8_soi[h] = l_f
        l_8_100[h] = l
        
        # instead : l_8_100.append(l)  could have been  done if l_8_100 would have been initially defined as an empty list []. See above.
    
    
    ###################
    #################
    
    # converting samples to ms.
    
    c = 600/18000
    d = 100
    
    
    l_f_8_soi_ms = []
    for k in range(8) :
        l_f_8_soi_ms.append(	(np.array(l_f_8_soi[k]))*c - d	)
    
    
    
    for i in range(8):
        for j in range(nrt+1):
            l_8_100[i][j] = ((np.array(l_8_100[i][j]))*c)-d
    
    
    
    
    
    #################
    #############
    
    # %%    parameters + plots.
    
    #   figure : for both psth & fit.
    
    fig = plt.figure(figsize=(17,17.5) , constrained_layout=True)
    subfigs = fig.subfigures(4,1 , wspace=0.1 , height_ratios=[2,1,1,1])

    # sharey=True : no need to calculate to calculate the max to set the y_limit.
    ax_top = subfigs[0].subplots(2,4 , sharex=True , sharey=True)  
    ax_r =ax_top.ravel()
    
    ax_bottom = subfigs[1].subplots(1,3)
    ax_tpm_isi = subfigs[2].subplots(1,3)   #  tpm : template !  ,  isi : inter-spike-interval distribution.
    ax_next = subfigs[3].subplots(1,3)
    
    #############
    ##############
    
    
    #	neb = number of elements in each bin of a graph : along all 8 sois.
    #   This is needed to adjust the y_limit to be in accordance with it (Michael).
    #	here automatically the plot is also generated.
    # neb_8soi =[]
    # max_8 = np.ones(8)  #  maximum bin size for 8 sois (8 elementns).
    # base_event_8 = np.ones(8) #  base_events of all 8 sois. to be plotted in the bottom subplot.
    # r_nmz_8 = np.ones(8) #  normalized response. respecting pre-event base.  All 8 sois.
    # r_nmz_pre_block_8 = np.ones(8) #  normalized response. respecting pre-train base.  All 8 sois.
    
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
        # x_vk , y_vk  :  vk : variable kernel.
        necf = ( l_f_8_soi_ms[i].size ) / nrt  # 95 : number of trials (5 out of 100 were eliminated).
        
        # variable kernel.
        # o : output : other outputs.
        try :
            y_vk , x_vk , o3 , o4 , o5 , o6 , o7 = opt.ssvkernel( x = l_f_8_soi_ms[i] )
        except ValueError :     #  culprit array : an array of size 0 [an empty array].
            x_vk , y_vk = np.array([]) , np.array([])
            df.loc[ v , 'err_kde' ] = 1
        except UnboundLocalError :    # while loop not satisfied (not 'True').  culprit array was : an array of size 3.
            x_vk , y_vk = np.array([]) , np.array([])
            df.loc[ v , 'err_kde' ] = 1
        except IndexError :    #  culprit : an array of size 1 (only 1 spike)
            x_vk , y_vk = np.array([]) , np.array([])
            df.loc[ v , 'err_kde' ] = 1

        
        # c : converted.
        y_vk_c = y_vk * necf  # deriving the estimated firing rate.
        
        # xy : the coupled (for vectorized operations below) x & y.
        xy_vk = np.vstack(( x_vk , y_vk_c ))
        
        kde_8_soi[i] = xy_vk
        
        # res : response
        res_vk = xy_vk[: ,    (xy_vk[0 , :] > 0 ) & (xy_vk[0 , :] < 100 )   ]  
        # be : baseline relative to event (pre-event).
        be_vk = xy_vk[:  ,    (xy_vk[0 , :] < 0 )   ]
        
        ####
        
        # max & latency of the response period.
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
            latency = res_vk[  0	 , idx	]
        
        latency_8_soi[i] = latency
        
        ####
        
        # m : mean of baseline.
        if be_vk.size == 0 :
            be_vk_m = 0
        else :
            be_vk_m = np.mean(be_vk[1,:])
            
        ####
        
        # response magnitude.
        res_mag = max_vk - be_vk_m
        
        res_mag_8_soi[i] = res_mag
        res_abs_8_soi[i] = max_vk
        base_evt_8_soi[i] = be_vk_m
        
        
        ####
        
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
        
        ax_r[i].plot( x_vk , y_vk_c , linewidth=3 )
        ax_r[i].hlines( height , xl_ms , xr_ms , color='m' )
        
        ###################
        ###################
        ###################
        
        ax_ep = ax_r[i].twinx()  #  ep : event plot  (raster).
        ax_ep.eventplot(l_8_100[i] , linewidths=1.5 , linelengths=1.5 , colors='k')
        
        # m = np.max(neb[4:8]) #   max of 1 single soi. during 5 bins (125ms) after the event.  for the fit plot.
        # max_8[i] = m
        
        # base_event = np.mean(neb[2:4])  #   base firing rate respeting the mean value along 50ms before the event.
        # base_event_8[i] = base_event    #   
        # r_nmz = (m - base_event)/(m + base_event) # response _ normalized.   pre_event base.
        # r_nmz_pre_block = (m - base_neb_mean_8_100_rint[i])/(m + base_neb_mean_8_100_rint[i]) # response _ normalized.  based on pre_train (pre-block) base.
        # r_nmz_8[i] = r_nmz
        # r_nmz_pre_block_8[i] = r_nmz_pre_block
        # db[v, (5+i)] = np.argmax(neb)   #   latency : order of the bin with maximum value. +5 : for putting it at a particular column in the database.
        # db[v , 24:32] = base_event_8  #  
        # db[v , 32:40] = max_8
        
        ax_r[i].axvline(x=0 , color='k' )   #   for psth.
        
        # neb_8soi.append(neb.tolist())       #   for psth.
    

    #############
    
    #  this snippet is modified to contain each bin containing 100ms :   of course for pre-block part.
    # the bin width (for example 25ms) should be constant between the baseline & response distributions (?) .
    # the total number of bins need not to be equal : for example 400 bins and 800 bins. It's like comparing 2 samples with different sample sizes.

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
    
    # ibr : inter-block rate.
    std_ibr = np.std(pbr_8_soi[1:])
    mean_ibr = np.mean(pbr_8_soi[1:])
    cv_ibr = std_ibr  /  mean_ibr   # cv : coefficient of variation.
    
    #####################

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
        popt_all_3p , pcov_all_3p = curve_fit(fit_func_3p, sois , res_abs_8_soi )
    except RuntimeError :
        popt_all_3p = np.array([ 0.5 , 0 , 0.05 ])
        pcov_all_3p = 0
        df.loc[ v , 'err_fit_all_3p' ] = 1
    
    
    # y of the fitted curve based on the formerly derived parmeters (popt)
    y_fit_all = fit_func(sois , *popt_all)
    y_fit_all_3p = fit_func_3p(sois , *popt_all_3p)
    
    # r2_score : goodness of fit.
    r2s_all = r2_score( res_abs_8_soi , y_fit_all )
    r2s_all_3p = r2_score( res_abs_8_soi , y_fit_all_3p )
    
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
    
    ###############
    ###############
    
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
    
    df.loc[ v , 'location' ] = int(cul[v][1])
    df.loc[ v , 'ext_ch' ] = eci[v]
    
    ########
    
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
    
    df.loc[ v:v , 'tmp' ] = pd.Series(data=[wfe_c.get_template(unit_id=v)[ : ,  eci[v] ]] , index=[v])
    df.loc[ v:v , 'kde' ] = pd.Series(data=[kde_8_soi] , index=[v])
    
    df.loc[ v:v , 'l_f_8_soi' ] = pd.Series(data=[l_f_8_soi] , index=[v])
    df.loc[ v:v , 'l_f_8_soi_ms' ] = pd.Series(data=[l_f_8_soi_ms] , index=[v])
    df.loc[ v:v , 'l_8_100' ] = pd.Series(data=[l_8_100] , index=[v])
    
    ############

    #   root mean square error _ weighted _ goodness of fit.
    # rmse_nmz = np.sqrt((np.sum((r_nmz_8[2:] - y_fit_exp_nmz)**2))/8)
    # mean_response_nmz = np.mean(r_nmz_8[2:])
    # rmse_w_nmz = rmse_nmz / mean_response_nmz

    # db[v , 43] = rmse_w_nmz


    ################
    ###########
    
    # fit for normalized response.
    # respecting pre-train baseline.
    # try :
    #     popt_nmz_pre_block , pcov_nmz = curve_fit(fit_func, sois , r_nmz_pre_block_8  )
    # except RuntimeError :
    #     popt_nmz_pre_block = np.array([ 0.5 , 0 ])
    #     pcov_nmz = 0
        
    # y_fit_exp_nmz_pre_block = fit_func(sois , *popt_nmz_pre_block )


    #   root mean square error _ weighted _ goodness of fit.
    # rmse_nmz_pre_block = np.sqrt((np.sum((r_nmz_pre_block_8 - y_fit_exp_nmz_pre_block)**2))/8)
    # mean_response_nmz_pre_block = np.mean(r_nmz_pre_block_8[2:])
    # rmse_w_nmz_pre_block = rmse_nmz_pre_block / mean_response_nmz_pre_block


    ###########
    ###########
    
    #   for the psth plot.
    
    #   flattened list.
    # neb_8soi_f = [j for i in neb_8soi for j in i]
    
    #	y limit. 
    #   attention : it's 'l' not '1'.
    # yl = 1.1 * max(neb_8soi_f)
    
    
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
        # ax_r[i].set_ylim( 0 , yl )
    
    
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
    
    #   fit plot.
    #  here, the main solid plot incoroporates the baseline during the 1min silence interval (10s of that 1min).

    # ax_bottom[1].plot(sois , r_nmz_pre_block_8 , linestyle='solid' , color='k' , label='normalized response')
    # ax_bottom[1].plot(sois , y_fit_exp_nmz_pre_block , linestyle='dotted' , color='k' , label='fit: A=' + str(np.around( popt_nmz_pre_block[0], decimals=3)) + ' , τ=' + str(np.around( popt_nmz_pre_block[1], decimals=3)) )

    # ax_bottom[1].set_xticks(ticks=sois)
    # ax_bottom[1].tick_params(axis='x' , labelrotation=90 , labelsize=6)
    # ax_bottom[1].set_xlabel('soi(ms)' , loc='right')
    # ax_bottom[1].set_title('normalized response respecting pre_block _ fit \n weighted rmse: ' + str(np.around(rmse_w_nmz_pre_block , decimals=2)) + ' _ sum of 8 normalized responses: ' + str(np.around(r_nmz_pre_block_8_sum , decimals=2)) , fontsize=9)
    
    # ax_bottom[1].legend( fontsize=8 )  
    

    ##############
    #################
    
    #	3rd trace (base firing rate) @ the fit plot.
    # ax_bottom[2].plot(sois , base_event_8 , linestyle='solid' , color='k' , label='base(events) ')
    # ax_bottom[2].plot(sois , base_neb_mean_8_100 , linestyle='dotted' , color='k' ,label='base(interval)')
    
    # ax_bottom[2].set_xticks(ticks=sois)
    # ax_bottom[2].tick_params(axis='x' , labelrotation=90 , labelsize=6)
    # ax_bottom[2].set_xlabel('soi(ms)' , loc='right')
    # ax_bottom[2].set_title('baselines' , fontsize=9 )
    # ax_bottom[2].legend( fontsize=8 )  #   to display the above defined legends !


    ################
    #############
    
    y_template = wfe_c.get_template(unit_id=v)[ : ,  eci[v] ]
    ax_tpm_isi[0].plot(	x_template , y_template 	)
    ax_tpm_isi[0].set_title('template' , fontsize=11 )
    
    sw.plot_isi_distribution(srt , unit_ids=[v] , axes=ax_tpm_isi[1] )
    ax_tpm_isi[1].set_title('isi' , fontsize=11 )


    ax_tpm_isi[2].text(0.1 , 0.2 , 'test for response significance, p-value : \n\nsoi-1 : ' + 
                       str(np.around(trs[0], decimals=3)) + '\nsoi-2 : ' + 
                       str(np.around(trs[1], decimals=3)) + '\nsoi-3 : ' + 
                       str(np.around(trs[2], decimals=3)) + '\nsoi-4 : ' + 
                       str(np.around(trs[3], decimals=3)) + '\nsoi-5 : ' + 
                       str(np.around(trs[4], decimals=3)) + '\nsoi-6 : ' + 
                       str(np.around(trs[5], decimals=3)) + '\nsoi-7 : ' + 
                       str(np.around(trs[6], decimals=3)) + '\nsoi-8 : ' + 
                       str(np.around(trs[7], decimals=3)) , 
                       fontsize=12)
    
    
    ax_tpm_isi[2].text(0.5 , 0.2 ,
                      'snr : ' + str(np.around(qm_a.loc[ v , 'snr' ] , decimals=3))  + '\n' + 
                      'isi_violations_ratio : ' + str(np.around(qm_a.loc[ v , 'isi_violations_ratio' ] , decimals=3))  + '\n' +
                      'isi_violations_count : ' + str(np.around(qm_a.loc[ v , 'isi_violations_count' ] , decimals=3))  + '\n' + 
                      'amplitude_cutoff : ' + str(np.around(qm_a.loc[ v , 'amplitude_cutoff' ] , decimals=3))  + '\n' +
                      'presence_ratio : ' + str(np.around(qm_a.loc[ v , 'presence_ratio' ] , decimals=3)) ,
                      fontsize=12
                      )
    
    
    ############
    
    # here, since I don't know if a total 1 minute silence period existed, before the first soi, I exclude the 1st pre-block silence period.
    ax_next[0].plot(pbr_8_soi[1:])
    ax_next[0].set_title(
        'inter_block rate (Hz) \n cv : ' + str(np.around( cv_ibr ,  decimals=3)) +  
        '  _  mean : ' + str(np.around( mean_ibr ,  decimals=3)) +
        '  _  std : ' + str(np.around( std_ibr ,  decimals=3)) , 
        fontsize=11 )
    
    ax_next[0].set_xticks(ticks=ticks_pbr , labels=labels_pbr )
    ax_next[0].set_xlabel('inter-block interval' , loc='right')
    
################
    
    fig.suptitle('unit_id : ' +  str(v) + ' _ unit location : ' + str(int(cul[v][1])) + ' μ | extremum channel : ' + str(eci[v]) + '  _  soi order : ' + str(soi_order_numeric) + '\n' + description_session , fontsize=12)
    #plt.gcf().text(0.02, 0.9, 'mean = ' + str(np.around(v_mean , decimals=1)) + '\n sd = ' + str(np.around(v_sd , decimals=1)) , fontsize=14)
    
    ##############
    ##################
    
    plt.savefig( dest_dir + '/v_' +  str(v)  + '.pdf' )
    plt.close()

####################

# adding the quality metrics & template metrics to the main dataframe.
df = pd.concat( [ df , qm_c , tm ] , axis=1 )

# there are 2 reasons for saving as binary ( HDF-5 or pickle) instead of saving in csv format :
# 1 : csv can not save all values in a cell that is an array object.
# 2 : when reloading the csv, it adds another column at the beginning of the frame as index, \
#     renaming the already index column to some unknown header.
df.to_pickle( dest_dir + '/df.pkl')
df.to_hdf( dest_dir + '/df.h5', key='df', mode='w')



#     plt.savefig( r'D:\analysis _ rec\2022-2-7\1\all vectors\v_' +  str(v)  + '.svg' )
    


# %% 
###########################


#   keep this below line in this program (don't move t to the REPL Ipython console).
#   otherwise, when creating new pdfs, the new ones will be appended to the old ones !
mergedObject = PdfFileMerger()
for fileNumber in unit_id :
    mergedObject.append(PdfFileReader( dest_dir + '/v_' + str(fileNumber) + '.pdf'))
mergedObject.write( dest_dir + '/total.pdf')



    #   converting possible nan.s to 0.
    #   these nan.s are a results of (0-0)/(0-0).
    #   nan would only make error in the fitting function.
    # r_nmz_8 = np.nan_to_num(r_nmz_8)
    # r_nmz_pre_block_8 =  np.nan_to_num(r_nmz_pre_block_8)
    
    
    # r_nmz_6_sum = r_nmz_8[2:].sum()     #   this is for 6 responses since the 1st 2 were overall ignored.
    # r_nmz_pre_block_8_sum = r_nmz_pre_block_8.sum()
    
    # r_nmz_8_sum = r_nmz_8.sum() 
    # db[v , 40] = r_nmz_8_sum
    
    
    ###############
    

# %%
#######
#######

# this is in pre_req_n.py

# dr_1 = r'/home/azare/groups/PrimNeu/Aryo/copy_data/Lucy_20221219/2022-12-19 _ Lucy _ terminal/P15/1/2022-12-21_17-26-23'
# dr_2 = r'/home/azare/groups/PrimNeu/Aryo/copy_data/Lucy_20221219/2022-12-19 _ Lucy _ terminal/P15/2/2022-12-21_17-29-49'
# dr_3 = r'/home/azare/groups/PrimNeu/Aryo/copy_data/Lucy_20221219/2022-12-19 _ Lucy _ terminal/P15/3/2022-12-21_18-08-10'
# dr_4 = r'/home/azare/groups/PrimNeu/Aryo/copy_data/Lucy_20221219/2022-12-19 _ Lucy _ terminal/P15/4/2022-12-21_18-39-25'
# dr_5 = r'/home/azare/groups/PrimNeu/Aryo/copy_data/Lucy_20221219/2022-12-19 _ Lucy _ terminal/P15/5/2022-12-21_18-42-59'


# rd_1 = se.read_openephys(dr_1 , stream_id='0')
# rd_2 = se.read_openephys(dr_2 , stream_id='0')
# rd_3 = se.read_openephys(dr_3 , stream_id='0')
# rd_4 = se.read_openephys(dr_4 , stream_id='0')
# rd_5 = se.read_openephys(dr_5 , stream_id='0')


# # for recording # 3 :
# # used for the trigger.
# sample_correction = rd_1.get_num_samples() + rd_2.get_num_samples() 

# # used for vec_p_c.
# rd = si.concatenate_recordings( [ rd_1 , rd_2 , rd_3 , rd_4 , rd_5 ] )

# # number of samples in the whole (all of the) combined recording.
# nsacr = rd.get_num_samples()


# %%


# absolute value : if the basleine would be 0.   =>  medion / python / ... / scipy signal / peak .doc
    # res_mag_8_soi - hh :  aboslute value of the half-height.
    # (res_mag_8_soi - hh) * 2 :  aboslute value of the height.

