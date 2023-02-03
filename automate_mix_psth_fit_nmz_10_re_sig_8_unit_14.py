
# this program is in  /home/azare/groups/PrimNeu/Aryo/analysis/fit
# env_2    environment in the server.
# this plots & analyzes the units.
# the immediate program before this is pipeline_n .docx  or pipe_sort_n.py : spike sorting program.


# 1st :
# create a new destination folder for savinf the outputs.
# run the function & the soi array.
# only adjust the 4 variables defined below.



################
################
################
################

# the source directory for the sorting objects.
source_dir = r'/home/azare/groups/PrimNeu/Aryo/analysis/sort/Benny_p3'


# the directory for extracting the triggers.
# the same directory named 'dr' used in spike-sorting.
directory = r'/home/azare/groups/PrimNeu/Aryo/copy_data/Benny_terminalexp/p3/2/2022-05-18_00-59-05'

# you should 1st create this in windows explorer.
# for pdfs & the database.
dest_dir = r'/home/azare/groups/PrimNeu/Aryo/analysis/B_t/p3_2'

######################

# this will be printed at sup-title.
description_session = 'Benny_terminal , right hemisphere , unit ,  p3_2 _ 180 Hz '

##############
##############
##############
##############

# loading the unit information :

srt = se.read_spykingcircus( source_dir + '/srt' )

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

# x axis of the template waveform plot.
x_template = np.linspace(-1 , 1 , 60 )

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


def fit_func(SOI, A, tau_0):
    return A*(1-np.exp(-(SOI-0.05)/tau_0))

sois = np.array([ 0.11 , 0.195 , 0.345 , 0.611 , 1.081 , 1.914 , 3.388 , 6])

################

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



# Means of all vectors (384 elements).
# vec_means = np.array([])
# vec_sds = np.array([])


# vector peaks.
#vec_p_all = []


length_v = ap[:,0].size

###########################

# db = database.
# column index :  if it is m_n : n is an index : if you want to later slice it : (m:n+1).
# 0 = 1 
# 1 = 1
# 2 = 1
# 3 = mean amplitude of the channel !
# 4 = SD of the amplitude of the channel !
# 5 = the bin index with maximum response (latency) for the 1st soi.
# 6 = the bin index with maximum response (latency) for the 2nd soi.
# 7 = the bin index with maximum response (latency) for the 3rd soi.
# 8 = the bin index with maximum response (latency) for the 4th soi.
# 9 = the bin index with maximum response (latency) for the 5th soi.
# 10 = the bin index with maximum response (latency) for the 6th soi.
# 11 = the bin index with maximum response (latency) for the 7th soi.
# 12 = the bin index with maximum response (latency) for the 8th soi.
# 13_20 (including the index 20) = baseline during 3 seconds before the start of each train (during the 1 minute silence period). 8 elements.
# 21 = 1
# 22 = 1
# 23 = 1
# 24_31 = base_event_8 : 8 elements : baselines immediately before the trigger (50ms = 2 bins).
# 32_39 = max_8 :  maxes of each soi : 8 elemetns.


# 40 = r_nmz_8_sum : sum of all normalized values for all 8 sois : for setting a threshold to rull-out noisy channels.
# 41 = A : normalized response.
# 42 = τ : normalized response.
# 43 = goodness of fit : normalized response.

db = np.ones(( (unit_id.max() + 1 )  , 44))    #   23 + 0(index) = 24 !!


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
    base_8=[]
    for i in range(8) :
        base = vec_p[ (vec_p > ( t_8_100[i , 0] - 300000 ) ) & ( vec_p < t_8_100[i , 0] ) ]
        base_8.append(base)
    
    
    ####  base_neb_mean_8 :   base of each of 8 sois / bins / mean / summed into 1 array. 
    ####  here, there’s no need for concatenation of different segments, hence the continuous vector (vec_p_c) is not needed.
    
    base_neb_mean_8 = np.array([])  #   this is for plotting the trend of baselines at the bottom of the multi_plot. This is the base in the silent inter-train interval.
    base_tdm_8_soi = [1,1,1,1,1,1,1,1]  #   a list of 8 np arrays.  each np array corresponds to 1 soi baseline. This is used to make a distribution to compare with response to test the significance.
    for i in range(8) :
        base_neb , edges =  np.histogram(base_8[i] , bins=400 , range=( t_8_100[i , 0] - 300000 , t_8_100[i , 0] ))  #   creating a histogram from a non-continuous data (base_8).
        base_neb_tdm , edges =  np.histogram(base_8[i] , bins=100 , range=( t_8_100[i , 0] - 300000 , t_8_100[i , 0] )) # this is for thesting the significance.
        base_neb_mean_8 = np.append(base_neb_mean_8 , np.mean(base_neb)  )
        base_tdm_8_soi[i] = base_neb_tdm
    
    

    #  this snippet is modified to contain each bin containing 100ms :   of course for pre-block part.
    # the bin width (for example 25ms) should be constant between the baseline & response distributions.
    # the total number of bins need not to be equal : for example 400 bins and 800 bins. It's like comparing 2 samples with different sample sizes.

    # this is for testing the response significance :  response part.
    # res_tdm_8_soi  : a list of 8 np arrays.
    # each np array corresponds to 1 soi (response, not baseline).
    # one_stimulus_segment : time stamps of peaks.
    # 100ms = big bin.
    res_tdm_8_soi = [1,1,1,1,1,1,1,1]

    for i in range (8):
        res_tdm_1_soi	= np.array([])		#	response tandem , tandem = not overlapped.
        for j in range (100) :
            one_stimulus_segment = vec_p[ (vec_p > ( t_8_100[i , j] + 300 ) ) & ( vec_p < (t_8_100[i , j] + 3300 )) ]
            res_tdm_1_soi	 = np.append ( res_tdm_1_soi  , one_stimulus_segment.shape )     #    here, all (not sum of) spike counts in each big bin (100ms) are added together.   
        res_tdm_8_soi[i] = res_tdm_1_soi



    # 'not_needed' : a junk variable. not needed.
    # pv = p value
    # trs : test for response significance
    trs = [1,1,1,1,1,1,1,1]
    for i in range (8) :
        not_needed , pv = mannwhitneyu( base_tdm_8_soi[i] , res_tdm_8_soi[i] , alternative='less')
        trs[i] = pv


    # if pv < 0.05 :
    #     significance_soi_8 = pv
    # else :
    #     significance_soi_8 = 'not significant'

    
    ####   Must be multiplied to 100 to be compatible with 100 repeatitions.
    base_neb_mean_8_100 = base_neb_mean_8 * 100


    #### rint : rounded to the nearest integer.
    base_neb_mean_8_100_rint = np.rint(base_neb_mean_8_100)

    db[v , 13:21] = base_neb_mean_8_100_rint 

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
        for i in t_8_100[h , :]   :
            smp = np.vstack( (smp , vec_p_c[ int(i-3000) : int(i+15000) ] ) )
        
    
        l = []  #  this is for 1 soi.   len(l) : 101 .  each element is a list of all spikes (rasters) corresponding to 1 repeatition.
        
        for j in range(101):
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
        for j in range(101):
            l_8_100[i][j] = ((np.array(l_8_100[i][j]))*c)-d
    
    
    
    
    
    #################
    #############
    
    # %%    parameters + plots.
    
    #   figure : for both psth & fit.
    
    fig = plt.figure(figsize=(17,14) , constrained_layout=True)
    subfigs = fig.subfigures(3,1 , wspace=0.1 , height_ratios=[2,1,1])


    ax_top = subfigs[0].subplots(2,4 , sharex=True , sharey=True)
    ax_r =ax_top.ravel()
    
    ax_bottom = subfigs[1].subplots(1,3)
    ax_tpm_isi = subfigs[2].subplots(1,3)   #  tpm : template !  ,  isi : inter-spike-interval distribution.
    

    #############
    ##############
    
    
    #	neb = number of elements in each bin of a graph : along all 8 sois.
    #   This is needed to adjust the y_limit to be in accordance with it (Michael).
    #	here automatically the plot is also generated.
    neb_8soi =[]
    max_8 = np.ones(8)  #  maximum bin size for 8 sois (8 elementns).
    base_event_8 = np.ones(8) #  base_events of all 8 sois. to be plotted in the bottom subplot.
    r_nmz_8 = np.ones(8) #  normalized response. respecting pre-event base.  All 8 sois.
    r_nmz_pre_block_8 = np.ones(8) #  normalized response. respecting pre-train base.  All 8 sois.
    for i in range(8):
        neb , bins, patches = ax_r[i].hist(l_f_8_soi_ms[i] , bins=24 , range=(-100 , 500))   #   neb : is for the PSTHs.
        ax_ep = ax_r[i].twinx()  #  ep : event plot  (raster).
        ax_ep.eventplot(l_8_100[i] , linewidths=1.5 , linelengths=1.5 , colors='k')
        m = np.max(neb[4:8]) #   max of 1 single soi. during 5 bins (125ms) after the event.  for the fit plot.
        max_8[i] = m
        base_event = np.mean(neb[2:4])  #   base firing rate respeting the mean value along 50ms before the event.
        base_event_8[i] = base_event    #   
        r_nmz = (m - base_event)/(m + base_event) # response _ normalized.   pre_event base.
        r_nmz_pre_block = (m - base_neb_mean_8_100_rint[i])/(m + base_neb_mean_8_100_rint[i]) # response _ normalized.  based on pre_train (pre-block) base.
        r_nmz_8[i] = r_nmz
        r_nmz_pre_block_8[i] = r_nmz_pre_block
        db[v, (5+i)] = np.argmax(neb)   #   latency : order of the bin with maximum value. +5 : for putting it at a particular column in the database.
        db[v , 24:32] = base_event_8  #  
        db[v , 32:40] = max_8
        ax_r[i].axvline(x=0 , color='k' )   #   for psth.
        neb_8soi.append(neb.tolist())       #   for psth.
    
    
    
    #############
    
    #   converting possible nan.s to 0.
    #   these nan.s are a results of (0-0)/(0-0).
    #   nan would only make error in the fitting function.
    r_nmz_8 = np.nan_to_num(r_nmz_8)
    r_nmz_pre_block_8 =  np.nan_to_num(r_nmz_pre_block_8)
    
    
    r_nmz_6_sum = r_nmz_8[2:].sum()     #   this is for 6 responses since the 1st 2 were overall ignored.
    r_nmz_pre_block_8_sum = r_nmz_pre_block_8.sum()
    
    r_nmz_8_sum = r_nmz_8.sum() 
    db[v , 40] = r_nmz_8_sum
    
    #####################

    # fit for normalized response.
    # this version : from soi_3 onwards :  soi_1 & soi_2 are omitted due to the overlapping of response on baseline.
    # respecting pre-event baseline.
    # nan values were converted above.
    try :
        popt_nmz, pcov_nmz = curve_fit(fit_func, sois[2:] , r_nmz_8[2:] )
    except RuntimeError :
        popt_nmz = np.array([ 0.5 , 0 ])
        pcov_nmz = 0
    
    y_fit_exp_nmz = fit_func(sois[2:] , *popt_nmz)

    db[v , 41] = popt_nmz[0]
    db[v , 42] = popt_nmz[1]

    #   root mean square error _ weighted _ goodness of fit.
    rmse_nmz = np.sqrt((np.sum((r_nmz_8[2:] - y_fit_exp_nmz)**2))/8)
    mean_response_nmz = np.mean(r_nmz_8[2:])
    rmse_w_nmz = rmse_nmz / mean_response_nmz

    db[v , 43] = rmse_w_nmz


    ################
    ###########
    
    # fit for normalized response.
    # respecting pre-train baseline.
    try :
        popt_nmz_pre_block , pcov_nmz = curve_fit(fit_func, sois , r_nmz_pre_block_8  )
    except RuntimeError :
        popt_nmz_pre_block = np.array([ 0.5 , 0 ])
        pcov_nmz = 0
        
    y_fit_exp_nmz_pre_block = fit_func(sois , *popt_nmz_pre_block )


    #   root mean square error _ weighted _ goodness of fit.
    rmse_nmz_pre_block = np.sqrt((np.sum((r_nmz_pre_block_8 - y_fit_exp_nmz_pre_block)**2))/8)
    mean_response_nmz_pre_block = np.mean(r_nmz_pre_block_8[2:])
    rmse_w_nmz_pre_block = rmse_nmz_pre_block / mean_response_nmz_pre_block


    ###########
    ###########
    
    #   for the psth plot.
    
    #   flattened list.
    neb_8soi_f = [j for i in neb_8soi for j in i]
    
    #	y limit. 
    #   attention : it's 'l' not '1'.
    yl = 1.1 * max(neb_8soi_f)
    
    
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
        ax_r[i].set_ylim( 0 , yl )
    
    
    ##################
    
    
    ax_r[7].set_xlabel('time in ms')
    ax_r[0].set_ylabel('total number of spikes along 100 repeatitions' , fontsize=9) 
    
    
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

    
    ax_bottom[0].plot(sois[2:] , r_nmz_8[2:] , linestyle='solid' , color='k' , label='normalized response')
    ax_bottom[0].plot(sois[2:] , y_fit_exp_nmz , linestyle='dotted' , color='k' , label='fit: A=' + str(np.around( popt_nmz[0], decimals=3)) + ' , τ=' + str(np.around( popt_nmz[1], decimals=3)) )

    ax_bottom[0].set_xticks(ticks=sois[2:])
    ax_bottom[0].tick_params(axis='x' , labelrotation=90 , labelsize=6)
    ax_bottom[0].set_xlabel('soi(ms)' , loc='right')
    ax_bottom[0].set_title('normalized response respecting pre_event _ fit \n weighted rmse: ' + str(np.around(rmse_w_nmz , decimals=2)) + ' _ sum of 6 normalized responses: ' + str(np.around(r_nmz_6_sum , decimals=2)) , fontsize=9)
    
    ax_bottom[0].legend( fontsize=8 )  
    
    
    ###################
    ###################
    
    #   fit plot.
    #  here, the main solid plot incoroporates the baseline during the 1min silence interval (10s of that 1min).

    ax_bottom[1].plot(sois , r_nmz_pre_block_8 , linestyle='solid' , color='k' , label='normalized response')
    ax_bottom[1].plot(sois , y_fit_exp_nmz_pre_block , linestyle='dotted' , color='k' , label='fit: A=' + str(np.around( popt_nmz_pre_block[0], decimals=3)) + ' , τ=' + str(np.around( popt_nmz_pre_block[1], decimals=3)) )

    ax_bottom[1].set_xticks(ticks=sois)
    ax_bottom[1].tick_params(axis='x' , labelrotation=90 , labelsize=6)
    ax_bottom[1].set_xlabel('soi(ms)' , loc='right')
    ax_bottom[1].set_title('normalized response respecting pre_block _ fit \n weighted rmse: ' + str(np.around(rmse_w_nmz_pre_block , decimals=2)) + ' _ sum of 8 normalized responses: ' + str(np.around(r_nmz_pre_block_8_sum , decimals=2)) , fontsize=9)
    
    ax_bottom[1].legend( fontsize=8 )  
    

    ##############
    #################
    
    #	3rd trace (base firing rate) @ the fit plot.
    ax_bottom[2].plot(sois , base_event_8 , linestyle='solid' , color='k' , label='base(events) ')
    ax_bottom[2].plot(sois , base_neb_mean_8_100 , linestyle='dotted' , color='k' ,label='base(interval)')
    
    ax_bottom[2].set_xticks(ticks=sois)
    ax_bottom[2].tick_params(axis='x' , labelrotation=90 , labelsize=6)
    ax_bottom[2].set_xlabel('soi(ms)' , loc='right')
    ax_bottom[2].set_title('baselines' , fontsize=9 )
    ax_bottom[2].legend( fontsize=8 )  #   to display the above defined legends !


    ################
    #############
    
    y_template = wfe_c.get_template(unit_id=v)[ : ,  eci[v] ]
    ax_tpm_isi[0].plot(	x_template , y_template 	)
    ax_tpm_isi[0].set_title('template' , fontsize=11 )
    
    sw.plot_isi_distribution(srt , unit_ids=[87] , axes=ax_tpm_isi[1] )
    ax_tpm_isi[1].set_title('isi' , fontsize=11 )


    ax_tpm_isi[2].text(0.2 , 0.2 , 'test for response significance, p-value : \n\nsoi-1 : ' + str(np.around(trs[0], decimals=3)) + '\nsoi-2 : ' + str(np.around(trs[1], decimals=3)) + '\nsoi-3 : ' + str(np.around(trs[2], decimals=3)) + '\nsoi-4 : ' + str(np.around(trs[3], decimals=3)) + '\nsoi-5 : ' + str(np.around(trs[4], decimals=3)) + '\nsoi-6 : ' + str(np.around(trs[5], decimals=3)) + '\nsoi-7 : ' + str(np.around(trs[6], decimals=3)) + '\nsoi-8 : ' + str(np.around(trs[7], decimals=3)) , fontsize=14)


################
    
    fig.suptitle('unit_id : ' +  str(v) + ' _ unit location : ' + str(int(cul[v][1])) + ' μ | extremum channel : ' + str(eci[v]) + '  _  soi order : ' + str(soi_order_numeric) + '\n' + description_session , fontsize=12)
    #plt.gcf().text(0.02, 0.9, 'mean = ' + str(np.around(v_mean , decimals=1)) + '\n sd = ' + str(np.around(v_sd , decimals=1)) , fontsize=14)
    
    ##############
    ##################
    
    plt.savefig( dest_dir + '/v_' +  str(v)  + '.pdf' )
    plt.close()

####################

# np.save( r'/home/azare/groups/PrimNeu/Aryo/analysis/2022-5-17/long/db.npy' , db)

#     plt.savefig( r'D:\analysis _ rec\2022-2-7\1\all vectors\v_' +  str(v)  + '.svg' )
    
    


# ###########################


# np.save( r'D:\analysis _ rec\2022-2-7\1\all vectors\vec_means' , vec_means)
# np.save( r'D:\analysis _ rec\2022-2-7\1\all vectors\vec_sds' , vec_sds)

# np.save( r'D:\analysis _ rec\2022-2-7\1\all vectors\vec_p_all' , vec_p_all )



#   keep this below line in this program (don't move t to the REPL Ipython console).
#   otherwise, when creating new pdfs, the new ones will be appended to the old ones !
mergedObject = PdfFileMerger()

for fileNumber in unit_id :
    mergedObject.append(PdfFileReader( dest_dir + '/v_' + str(fileNumber) + '.pdf'))

mergedObject.write( dest_dir + '/total.pdf')



