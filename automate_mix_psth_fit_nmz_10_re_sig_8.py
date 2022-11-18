

#   change in this page : 
    # input directory
    # output directories (2 : pdfs , db) _ 
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


# def fit_func(SOI, A, tau_0):
#     return A*(1-np.exp(-(SOI-0.05)/tau_0))

# sois = np.array([ 0.11 , 0.195 , 0.345 , 0.611 , 1.081 , 1.914 , 3.388 , 6])

################


#   number of vectors (channels) to analyze.
nva = 300

#   change this directory.
directory = r'/home/azare/groups/PrimNeu/Aryo/rec_2/2022-3-23/3/2022-03-23_11-36-05'
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

# below : diff functions requires an array, not a vector. hence I reshaped it.

for i in range(8) :
    if ( ( np.diff(trg_re_r[i , :].reshape(1,100))[0,0] > 3000) & ( np.diff(trg_re_r[i , :].reshape(1,100))[0,0] < 3500) ) :
        t_8_100[0 , :] = trg_re_r[i,:]
    if ( ( np.diff(trg_re_r[i , :].reshape(1,100))[0,0] > 5000) & ( np.diff(trg_re_r[i , :].reshape(1,100))[0,0] < 7000) ) :
        t_8_100[1 , :] = trg_re_r[i,:]
    if ( ( np.diff(trg_re_r[i , :].reshape(1,100))[0,0] > 10000) & ( np.diff(trg_re_r[i , :].reshape(1,100))[0,0] < 11000) ) :
        t_8_100[2 , :] = trg_re_r[i,:]
    if ( ( np.diff(trg_re_r[i , :].reshape(1,100))[0,0] > 15000) & ( np.diff(trg_re_r[i , :].reshape(1,100))[0,0] < 20000) ) :
        t_8_100[3 , :] = trg_re_r[i,:]
    if ( ( np.diff(trg_re_r[i , :].reshape(1,100))[0,0] > 30000) & ( np.diff(trg_re_r[i , :].reshape(1,100))[0,0] < 35000) ) :
        t_8_100[4 , :] = trg_re_r[i,:]
    if ( ( np.diff(trg_re_r[i , :].reshape(1,100))[0,0] > 50000) & ( np.diff(trg_re_r[i , :].reshape(1,100))[0,0] < 60000) ) :
        t_8_100[5 , :] = trg_re_r[i,:]
    if ( ( np.diff(trg_re_r[i , :].reshape(1,100))[0,0] > 100000) & ( np.diff(trg_re_r[i , :].reshape(1,100))[0,0] < 110000) ) :
        t_8_100[6 , :] = trg_re_r[i,:]
    if ( ( np.diff(trg_re_r[i , :].reshape(1,100))[0,0] > 170000) & ( np.diff(trg_re_r[i , :].reshape(1,100))[0,0] < 200000) ) :
        t_8_100[7 , :] = trg_re_r[i,:]


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

db = np.ones((nva , 44))    #   23 + 0(index) = 24 !!


###########################


# v = vector = column index = channel number - 1
for v in range(31,300) :

    
    v_mean = np.mean(ap[:100000 , v ])
    v_sd = np.std(ap[:100000 , v ])
    
    db[v , 3] = v_mean
    db[v , 4] = v_sd
    
    # vec_means = np.append(vec_means , v_mean )
    # vec_sds = np.append(vec_sds , v_sd )
    
    
    #	vector peaks : positive & negative.
    vec_p_pos , di_pos = ss.find_peaks( ap[ : , v ] ,  height = (v_mean + (3* v_sd) ) 	)
    vec_p_neg , di_neg = ss.find_peaks( -ap[ : , v ] ,  height = (-v_mean + (3* v_sd) ) 	)
    
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

    db[v , 13:21] = base_neb_mean_8_100_rint 

    #################
    ###################
    
    #   l_f_8_soi is the common & nuclear step for both psth & fit plots.
    l_f_8_soi = [1,1,1,1,1,1,1,1]
    
    for h in range(8):
    
        smp = np.zeros(18000)
        for i in t_8_100[h , :]   :
            smp = np.vstack( (smp , vec_p_c[ int(i-3000) : int(i+15000) ] ) )
        
    
        l = []
        for j in range(101):
            l.append( np.asarray(np.where(smp[j,:]==1)).flatten().tolist() )    #   hence converting a continuous to a discrete	array. This discrete array will be used to make a histogram.
    
        l_f = [j for i in l for j in i]
    
        l_f_8_soi[h] = l_f
    
    
    ###################
    
    c = 600/18000
    d = 100
    
    
    l_f_8_soi_ms = []
    for k in range(8) :
        l_f_8_soi_ms.append(	(np.array(l_f_8_soi[k]))*c - d	)
    
    
    #################
    #############
    
    # %%    parameters + plots.
    
    #   figure : for both psth & fit.
    
    fig = plt.figure(figsize=(17,9) , constrained_layout=True)
    subfigs = fig.subfigures(2,1 , wspace=0.1 , height_ratios=[1,1])

    ax_top = subfigs[0].subplots(2,4 , sharex=True , sharey=True)
    ax_bottom = subfigs[1].subplots(1,3)
    ax_r =ax_top.ravel()

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
    
    db[v , 40] = r_nmz_8_sum
    
    #####################

    # fit for normalized response.
    # this version : from soi_3 onwards :  soi_1 & soi_2 are omitted due to the overlapping of response on baseline.
    # respecting pre-event baseline.
    popt_nmz, pcov_nmz = curve_fit(fit_func, sois[2:] , r_nmz_8[2:] )
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
    popt_nmz_pre_block , pcov_nmz = curve_fit(fit_func, sois , r_nmz_pre_block_8  )
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
    
    fig.suptitle('v_' +  str(v)  + '  -  channel ' +  str(v+1) + '\n   Benny, left hemisphere , mua ,  2022-3-23_3 , tone 12 kHz' + ' _ significance @ soi_8: ' + str(format( pv , '4f')) , fontsize=12)
    #plt.gcf().text(0.02, 0.9, 'mean = ' + str(np.around(v_mean , decimals=1)) + '\n sd = ' + str(np.around(v_sd , decimals=1)) , fontsize=14)
    
    ##############
    ##################
    
    plt.savefig( r'/home/azare/groups/PrimNeu/Aryo/analysis/2022-3-23/3/mix_2/v_' +  str(v)  + '.pdf' )
    plt.close()

####################

np.save( r'/home/azare/groups/PrimNeu/Aryo/analysis/2022-3-23/3/mix_2/db.npy' , db)

#     plt.savefig( r'D:\analysis _ rec\2022-2-7\1\all vectors\v_' +  str(v)  + '.svg' )
    
    


# ###########################


# np.save( r'D:\analysis _ rec\2022-2-7\1\all vectors\vec_means' , vec_means)
# np.save( r'D:\analysis _ rec\2022-2-7\1\all vectors\vec_sds' , vec_sds)

# np.save( r'D:\analysis _ rec\2022-2-7\1\all vectors\vec_p_all' , vec_p_all )


