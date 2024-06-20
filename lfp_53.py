
# env_18
    # the set of packages needed for this environment is separately written in a file named 'module_lfp.py' in the same folder as this file :
    # these are a subset of the module.py, to be able to load all of them faster, I collected them in a separate file.
        # G:\Aryo\analysis\fit\lfp.

# why one should not use CMR for lfp    =>    G:\Aryo\analysis\fit\lfp\lfp.docx
# 'raw' or 'absolute' were used with the same meaning here : not baseline corrected 

# %%

# import the following packages ;
# scipy ...
# spike-interface ; without _ sortingcomponents.motion_correction
# pypdf_2
# open_ephys

# %%

# in case of multi-block recording, you may need to change 2 lines :
        # rd = se.read_openephys(dr , stream_id='1' )   # , block_index=0
        # rec = session.recordnodes[0].recordings[0]
    # see below for further details.

# %%

# create a folder named 'lfp' in the dest_dir path below.

dr = r'/home/azare/groups/PrimNeu/Aryo/copy_data/Elfie_final_exp_202303/p4/4/2023-03-21_08-45-35'
# dr = r'/home/azare/groups/PrimNeu/Aryo/copy_data/Lucy_20221219/2022-12-19 _ Lucy _ terminal/P14/9/2022-12-21_16-27-13'
# dr = r'/home/azare/groups/PrimNeu/Aryo/copy_data/Lucy_20221219/2022-12-19 _ Lucy _ terminal/P12/6/2022-12-21_08-02-35'
# dr = r'/home/azare/groups/PrimNeu/Aryo/copy_data/Elfie_final_exp_202303/p21/3/2023-03-23_14-51-44'
# dr = r'/home/azare/groups/PrimNeu/Aryo/copy_data/Lucy_20221219/2022-12-19 _ Lucy _ terminal/P10/3/2022-12-21_03-04-08'

description_session = 'LFP ( 6-12 Hz ) , Elfie , P4_4_R , tone : 100 HZ'

# 1st : create a folder named 'lfp' in this path.
# lfp file : precprocessed in this page   =>  will be saved here :
dest_dir = r'/home/azare/groups/PrimNeu/Aryo/analysis/Elfie/p4/4/lfp_6_12'

# %%
# %%

# related to the trigger.

# nte : number of trials (out of 100) to exclude.
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

# conversion factor to μv ( =>  lfp.docx in this folder ).
bit_volt = 0.195

# %%

prb = read_probeinterface(r'/home/azare/groups/PrimNeu/Aryo/copy_data/Lucy_20221219/Lucy.json')

# sample : if there are multiple experiments in 1 recording (multi-block), use 'block_index' to extract each one separately.
    # rd_6 = se.read_openephys(dr_6 , stream_id='0' , block_index=0 )    
    # this happens during data acquisition when in open-ephys for a new recording a new destination folder is not set (by mistake).
    # partially because of open_ephys gui's bug that you need to delete & re-bring the record node plug-in.
rd = se.read_openephys(dr , stream_id='1' )   # , block_index=0

# trigger channel
# trg_ch = rd.get_traces()[: , 384]      # this is non-functional

rd_prb = rd.set_probegroup(prb)

# change the filtering parameters back.
rd_prb_f = bandpass_filter( rd_prb , freq_min=5 , freq_max=50 )

rd_pps_lfp = rd_prb_f.save(folder= dest_dir + '/pps_lfp' ,  format='binary' , n_jobs=-1, chunk_size=2500  )


# to load a pre-processed file.
# rd_pps_lfp = si.load_extractor( r'/home/azare/groups/PrimNeu/Aryo/analysis/Lucy/p7/4/lfp_9/pps_lfp' )
# I encountered this error :
    # ValueError: This folder is not a cached folder /home/azare/groups/PrimNeu/Aryo/analysis/Lucy/p7/4/lfp/pps_lfp
# seemingly the file was not properly saved. you may rerun the preprocessing steps & resave it !


lfp = rd_pps_lfp.get_traces() * bit_volt  

# type(lfp)
    # Out[1038]: numpy.ndarray

# lfp.shape
    # Out[1037]: (4837640, 384)

# %%

#	triggers.
# each trigger is 6 smaples in lfp stream.


# for extracting the triggers.
# in case of non-separate recorindgs (multi-block recordings) : example : Elfie p2_2 : due to Michael's mistake : run the below line
    # rec = session.recordnodes[0].recordings[1]
    # the main 'directory' [ in session = Session(directory) ] is the directory of the parent recording (p2_1 : contains p2_2 ).
session = Session(dr)
rec = session.recordnodes[0].recordings[0]
lfp_oep = rec.continuous[1].samples

trg = lfp_oep[: , 384]

t , di_t = ss.find_peaks( trg , plateau_size=2)

trg_lfp_re = di_t['left_edges']

##########

# trg_lfp_re_r : trigger reshaped.
trg_lfp_re_r = trg_lfp_re.reshape(8,100)

# the new sorted trigger.
t_lfp_8_100 = np.zeros((8,100))

###############

soi_order = [1,1,1,1,1,1,1,1]   #  string (below)
soi_order_numeric = [1,1,1,1,1,1,1,1]

###############


for i in range(8) :
    if ( ( np.diff(trg_lfp_re_r[i , :].reshape(1,100))[0,0] > 200) & ( np.diff(trg_lfp_re_r[i , :].reshape(1,100))[0,0] < 350) ) :
        t_lfp_8_100[0 , :] = trg_lfp_re_r[i,:]
        soi_order[i] = 'soi_1'
        soi_order_numeric[i] = 1
    if ( ( np.diff(trg_lfp_re_r[i , :].reshape(1,100))[0,0] > 400) & ( np.diff(trg_lfp_re_r[i , :].reshape(1,100))[0,0] < 600) ) :
        t_lfp_8_100[1 , :] = trg_lfp_re_r[i,:]
        soi_order[i] = 'soi_2'
        soi_order_numeric[i] = 2
    if ( ( np.diff(trg_lfp_re_r[i , :].reshape(1,100))[0,0] > 800) & ( np.diff(trg_lfp_re_r[i , :].reshape(1,100))[0,0] < 1000) ) :
        t_lfp_8_100[2 , :] = trg_lfp_re_r[i,:]
        soi_order[i] = 'soi_3'
        soi_order_numeric[i] = 3
    if ( ( np.diff(trg_lfp_re_r[i , :].reshape(1,100))[0,0] > 1400) & ( np.diff(trg_lfp_re_r[i , :].reshape(1,100))[0,0] < 1800) ) :
        t_lfp_8_100[3 , :] = trg_lfp_re_r[i,:]
        soi_order[i] = 'soi_4'
        soi_order_numeric[i] = 4
    if ( ( np.diff(trg_lfp_re_r[i , :].reshape(1,100))[0,0] > 2500) & ( np.diff(trg_lfp_re_r[i , :].reshape(1,100))[0,0] < 3000) ) :
        t_lfp_8_100[4 , :] = trg_lfp_re_r[i,:]
        soi_order[i] = 'soi_5'
        soi_order_numeric[i] = 5
    if ( ( np.diff(trg_lfp_re_r[i , :].reshape(1,100))[0,0] > 4000) & ( np.diff(trg_lfp_re_r[i , :].reshape(1,100))[0,0] < 5000) ) :
        t_lfp_8_100[5 , :] = trg_lfp_re_r[i,:]
        soi_order[i] = 'soi_6'
        soi_order_numeric[i] = 6
    if ( ( np.diff(trg_lfp_re_r[i , :].reshape(1,100))[0,0] > 8000) & ( np.diff(trg_lfp_re_r[i , :].reshape(1,100))[0,0] < 9000) ) :
        t_lfp_8_100[6 , :] = trg_lfp_re_r[i,:]
        soi_order[i] = 'soi_7'
        soi_order_numeric[i] = 7
    if ( ( np.diff(trg_lfp_re_r[i , :].reshape(1,100))[0,0] > 14000) & ( np.diff(trg_lfp_re_r[i , :].reshape(1,100))[0,0] < 16000) ) :
        t_lfp_8_100[7 , :] = trg_lfp_re_r[i,:]
        soi_order[i] = 'soi_8'
        soi_order_numeric[i] = 8


# %% 
# %%


# 'base_left_y' , 'base_right_y' : & consequntly :  'base_left_y_8_soi' , 'base_right_y_8_soi' :
    # these are absolute values : hence if the polarity is negative, you should negate it (* -1 ).


# columns of the dataframe.
clm = [  
       'smp_8_soi' ,
       'bound_x' , 'bound_x_l_t' , 'bound_x_r_t' , 
       'p_t_x_ms_8_soi' , 'p_t_y_8_soi' , 
       'peak_x_ms_8_soi' , 'base_left_x_ms_8_soi' , 'base_right_x_ms_8_soi' ,
       'peak_y_8_soi' , 'base_left_y_8_soi' , 'base_right_y_8_soi' ,
       'err_fit_bc' , 'polarity_8_soi' , 'polarity_baseline' ,
       'Tau' , 'A' , 't0' , 'r2s' ,
       'p_t_y_bl' , 'p_t_x_bl'
]

# initialize data
init_data = np.zeros( ( nva , len(clm) ) )

# initializing the dataframe.
# note : the index will be channel numbers
df = pd.DataFrame( data=init_data , index=np.arange(nva)  , columns=clm )

# %%

# 3D : 8 sois , 100 repeatitions , time-span of interest.
# 8 * 100 * samples (time)

# wrong ?    smp = np.empty(lfp[:,1].size)
# for debugging, do not run this cell separately, otherwise 380 matplotlib windows will eb opened.
    # if mistakenly happened  =>  plt.close('all')

# for the 8 sois
x = np.linspace(-100 , 500 , 1500 )     # the x axis of the plot.
x_ticks = np.array([ 0 , 50 , 100 , 250 , 500 ])
x_cont = np.linspace( 0 , 6 , 1000 )   #  cont : continuous : for plotting the fitted line : from 0 to 6 s.


# obl : offset baseline (ms) _ from the stimulus time.
obl = 5000   # 
x_baseline = np.linspace( 5000 , 5600 , 1500 )     # the x axis of the plot.
x_ticks_baseline = np.array([ 5000 , 5250 , 5500 ])


# %%
# %%

for v in range(0 , nva , leap ) :  

# this is for testing, so that it would not take so long to analyze, & no need to break the process in the middle. 
    # annotate it for the real analysis.
    # other reason is that without a loop (starting from the line below), indentation will be jammed ! :
        # see the error report at the bottom of the page.
        # you should still put the curser at the begining of the 'for loop' line when hitting CTRL+F9 (run from the current line).
# for v in range(4) :    
    
    # %%
    
    # for channel id : v :
        # below : axis=0 in stacking smp_8_soi : the 1st axis of the 'result' array.
    smp_8_soi = np.empty((8 , 1500))
    
    # pre-allocating memory !!
    # note : you should put this cell after the for loop.
        # if putting it before it, unlike expected, it will not replace te values with the new 'v' (channel) values !
    # last 4 sois : the 1st 4 rows are useless : the loop is not written in a good way :
            # at best the loop should have been writen in reverse : for i in range (8,4) ...
            # then the 1st 4 rows would have not been needed.
    bound_x = np.zeros((8,2))   # x values of peak & trough.  see below

    # bound_x_l_t = 0  # average of the left bounds
    # bound_x_r_t = 0  # average of the right bounds

    # peak to trough ( x & y distances )
    p_t_x_ms_8_soi = np.zeros(8)    
    p_t_y_8_soi = np.zeros(8)

    ###########

    # ( left base , peak , right base ) * ( x , y )
    
    peak_x_ms_8_soi = np.zeros(8)  
    base_left_x_ms_8_soi = np.zeros(8)  
    base_right_x_ms_8_soi = np.zeros(8)  

    peak_y_8_soi = np.zeros(8)  
    base_left_y_8_soi = np.zeros(8) 
    base_right_y_8_soi = np.zeros(8) 

    polarity_8_soi = np.zeros(8)

    # %%
    
    fig = plt.figure(figsize=(17,14) , constrained_layout=True)
    subfigs = fig.subfigures(2,1 , wspace=0.1 , height_ratios=[2,1] )

    ax_top = subfigs[0].subplots(2,4 , sharex=True , sharey=True)
    ax_r =ax_top.ravel()
    
    ax_bottom = subfigs[1].subplots(1,3)
    
    # %%
    # %%
    # %%
    # %%

    for h in range(4,8) :
        smp = np.zeros(1500)
        for i in t_lfp_8_100[h , nte:]   :
            smp = np.vstack( (smp , lfp[: , v][ int(i-250) : int(i+1250) ] ) )  # this is set for channel 298.
        smp_m = np.mean(smp , axis=0)  # m : mean : mean of 97 trials.
        smp_8_soi[h , :] = smp_m  # you can not use np.stackhere, but you may use np.append.
        
        # %%
        
        # extracting the relavant parameters from the wave.
        
        ts = smp_m[250:500] # ts : trace section : 0-100 ms after the stimulus.
        
        # pos , neg : correspond to the positive & negative waves.

        pos_peak_y = np.max( ts )
        pos_peak_x = np.argmax( ts )

        # the output here is the 'absolute' values of the negative peaks.  trough
        neg_peak_y = np.max( -ts )
        neg_peak_x = np.argmax( -ts )

# %%

# bound : boundary.  x values of peak & trough.  latency bounday
# the y value is not important here.
        if pos_peak_x < neg_peak_x :
            bound_x_l = pos_peak_x
            bound_x_r = neg_peak_x
        else :
            bound_x_l = neg_peak_x
            bound_x_r = pos_peak_x

        bound_x[h,0] = bound_x_l
        bound_x[h,1] = bound_x_r
     
    # %%

    # t : total : the average of the last 4 sois + a confidence period.
    # 6 ms = 15 samples
    bound_x_l_t = int (  np.min( bound_x[4:,0] ) - 15  )   # int : since it will be used as an index for slicing.
    bound_x_r_t = int (  np.max( bound_x[4:,1] ) + 15  )
    
    # this is because the possility of bound_x_l_t to be negative, due to the subraction by 15 in its calculation ( see above ).
        # conequently, a negative index is not valid.
        # a negative index is not valid.
    if bound_x_l_t < 0 :
        bound_x_l_t = 0
    else :
        pass
    
    
    # this is because the possility of bound_x_r_t to bigger than 249 (the length of 'ts' is 250), due to the addition by 15 in its calculation ( see above ).
        # conequently, in the ss.peak_prominences function below, the index would be biggger than the length of the input array.
    if bound_x_r_t > 249 :
        bound_x_r_t = 249
    else :
        pass
    
    # is used in 'peak_prominences' function : parameter 'wlen'.
    # this is specially needed for the small sois.
    # *2 : reason : for both right & left bases.
    window_length = ( bound_x_r_t - bound_x_l_t ) * 2
    
    # %%
    # %%
    # %%
    # %%

    # all sois
    for h in range(8) :
        smp = np.zeros(1500)
        for i in t_lfp_8_100[h , nte:]   :
            smp = np.vstack( (smp , lfp[: , v][ int(i-250) : int(i+1250) ] ) )  
        smp_m = np.mean(smp , axis=0)  # m : mean : mean of 97 trials.
        smp_8_soi[h , :] = smp_m  # you can not use np.stackhere, but you may use np.append.
        
        # %%
        # %%
        
        # extracting the relavant parameters from the wave.
        
        ts = smp_m[250:500] # ts : trace section : 0-100 ms after the stimulus.
        # trace section restricted to the latency period : mainly used in the 1st 4 sois.
        ts_2 = smp_m[ 250 + bound_x_l_t   :   250 + bound_x_r_t ]
        
        # pos , neg : correspond to the positive & negative waves.

        pos_peak_y = np.max( ts_2 )
        pos_peak_x = bound_x_l_t + np.argmax( ts_2 )    #  + bound_x_l_t  :   so that it would again be referenced to time 0 .

        # the output here is the 'absolute' values of the negative peaks.  trough
        neg_peak_y = np.max( -ts_2 )
        neg_peak_x = bound_x_l_t + np.argmax( -ts_2 ) 

        # %%

        # for the comparison : since the find_peak function only gets the positive peaks, you do not need to mention the absolute value here.
        if pos_peak_y > neg_peak_y :
            peak_y = pos_peak_y
            peak_x = pos_peak_x   # latency
            ts_adj = ts  # adj : adjusted
            polarity = 'pos'
            polarity_8_soi[h] = 1
        else :
            peak_y = neg_peak_y   # the reversed values. 
            peak_x = neg_peak_x   # latency
            ts_adj = -ts    # adj : adjusted
            polarity = 'neg'
            polarity_8_soi[h] = -1

        peak_y_8_soi[h] = peak_y

        # in case ts_adj would be a negative trace, the outputs here would be the 'absolute' values of the negative peaks.

        # the _array suffix is because the output of the function is a tuple, & I should extracrt the 'value' in the next step.
        prominence_array , base_left_array , base_right_array = ss.peak_prominences( ts_adj , np.array([peak_x]) , wlen=window_length )
        width_array , width_height_array , left_ips_array, right_ips_array = ss.peak_widths( ts_adj , np.array([peak_x]) , rel_height=1 )

        # _x : x values : indices of the wave ( ts )
            # you may use them for slicing the wave to get the y values.

        # [0] : to extract the actual 'value' : see the top explanation.
        prominence = prominence_array[0]
        base_left_x = base_left_array[0]
        base_right_x = base_right_array[0]

        width = width_array[0]
        width_height = width_height_array[0]
        left_ips_x = left_ips_array[0]
        right_ips_x = right_ips_array[0]


        # %%

        base_left_y = ts_adj[base_left_x]
        base_right_y = ts_adj[base_right_x]

        base_left_y_8_soi[h] = base_left_y
        base_right_y_8_soi[h] = base_right_y

        # %%

        left_arm = peak_y - base_left_y
        right_arm = peak_y - base_right_y 

        # p_t : peak-to-trough : x & y values.
        if left_arm > right_arm :
            p_t_y = left_arm
            p_t_x = peak_x - base_left_x
        else :
            p_t_y = right_arm 
            p_t_x = base_right_x - peak_x


        # ms : millisecond
        # convert the x values from 'sample' to 'ms'.
        p_t_x_ms = p_t_x * 0.4
        p_t_x_ms_8_soi[h] = p_t_x_ms

        p_t_y_8_soi[h] = p_t_y

        # %%

        # ms : millisecond
        # convert the x values from 'sample' to 'ms'.
        # this is needed for plotting, since the x axis of plots are in ms unit.
        peak_x_ms = peak_x * 0.4
        base_left_x_ms = base_left_x * 0.4
        base_right_x_ms = base_right_x * 0.4
        

        peak_x_ms_8_soi[h] = peak_x_ms  
        base_left_x_ms_8_soi[h] = base_left_x_ms 
        base_right_x_ms_8_soi[h] = base_right_x_ms


        # %%
        # %%

        ax_r[h].plot(x , smp_m)
        ax_r[h].set_ylim(-100, 100)   # this number is to equalize the y-limits for all channels.  
            # It may need to be modified according to actual values of the channels after exploring the pdf.s.
        ax_r[h].axvline(x=0 , color='k' )
        
        ax_r[h].set_xticks(ticks=x_ticks)
        ax_r[h].tick_params(axis='x' , labelrotation=45 , labelsize=12)
        
        # %%

        # peak-to-trough : y distance & x distance will be plotted as a line.
        # the trace : 4 conditions exist :  * 
            # positive & negative waves
            # left or right arm being longer.

        if polarity == 'pos' :
            ax_r[h].vlines( x=peak_x_ms , ymin=( peak_y - p_t_y ) , ymax=peak_y , color='red' ) 
            if left_arm > right_arm :
                ax_r[h].hlines(  y=( peak_y - p_t_y ) , xmin=base_left_x_ms , xmax=peak_x_ms  , color='red'  )
            else :
                ax_r[h].hlines(  y=( peak_y - p_t_y ) , xmin=peak_x_ms , xmax=base_right_x_ms , color='red'  )
        else :
            ax_r[h].vlines( x=peak_x_ms , ymin=-peak_y , ymax=( -peak_y + p_t_y ) , color='red' )
            if left_arm > right_arm :
                ax_r[h].hlines(  y=( -peak_y + p_t_y ) , xmin=base_left_x_ms , xmax=peak_x_ms  , color='red'  )
            else :
                ax_r[h].hlines(  y=( -peak_y + p_t_y ) , xmin=peak_x_ms , xmax=base_right_x_ms , color='red'  )

# %%

        # plotting the left & right base points
        if polarity == 'pos' :
            ax_r[h].scatter( 
                            [ base_left_x_ms , base_right_x_ms ] , [ base_left_y , base_right_y ] ,
                            s=50 , color='red' )
        else :
            ax_r[h].scatter( 
                            [ base_left_x_ms , base_right_x_ms ] , [ -base_left_y , -base_right_y ] ,
                            s=50 , color='red' )

# %%

        ax_r[0].axvline(x=110 , color='k')
        ax_r[0].axvline(x=220 , color='k')
        ax_r[0].axvline(x=330 , color='k')
        ax_r[0].axvline(x=440 , color='k')
        ax_r[0].axvline(x=550 , color='k')
        
        ax_r[1].axvline(x=195 , color='k')
        ax_r[1].axvline(x= 390 , color='k')
        
        ax_r[2].axvline(x=345 , color='k')
        
        # %%
        
        ax_r[7].set_xlabel('time (ms)')
        ax_r[0].set_ylabel('LFP (μv)') 
    
# %%
# %%
# %%
# %%

    # bl : baseline : for the baseline correction.
    # after 5 seconds from the trigger + 600ms
    
    smp_bl = np.zeros( 1500 )
    for i in t_lfp_8_100[ 7 , nte:]   :  # 7 : 8th soi
        smp_bl = np.vstack( ( smp_bl , lfp[: , v][ int(i+12500) : int( i + 14000 ) ] ) )  
    smp_bl_m = np.mean(smp_bl , axis=0)  # m : mean : mean of 97 trials.
        
        # extracting the relavant parameters from the wave.
        
    ts_bl = smp_bl_m[ : window_length ] # ts : trace section
    
    # pos , neg : correspond to the positive & negative waves.

    pos_peak_y_bl = np.max( ts_bl )
    pos_peak_x_bl = np.argmax( ts_bl )    # referenced (from) to time 0 .

    # the output here is the 'absolute' values of the negative peaks.  trough
    neg_peak_y_bl = np.max( -ts_bl )
    neg_peak_x_bl = np.argmax( -ts_bl ) 

    # %%

    # for the comparison : since the find_peak function only gets the positive peaks, you do not need to mention the abdolute value here.
    if pos_peak_y_bl > neg_peak_y_bl :
        peak_y_bl = pos_peak_y_bl
        peak_x_bl = pos_peak_x_bl   # latency
        ts_adj_bl = ts_bl  # adj : adjusted
        polarity_bl = 'pos'
        df.loc[ v , 'polarity_baseline' ] = 1
    else :
        peak_y_bl = neg_peak_y_bl   # the reversed values. 
        peak_x_bl = neg_peak_x_bl   # latency
        ts_adj_bl = -ts_bl    # adj : adjusted
        polarity_bl = 'neg'
        df.loc[ v , 'polarity_baseline' ] = -1

    # in case ts_adj would be a negative trace, the outputs here would be the 'absolute' values of the negative peaks.

    # the _array suffix is because the output of the function is a tuple, & I should extracrt the 'value' in the next step.
    prominence_array_bl , base_left_array_bl , base_right_array_bl = ss.peak_prominences( ts_adj_bl , np.array([peak_x_bl]) )
    width_array_bl , width_height_array_bl , left_ips_array_bl , right_ips_array_bl = ss.peak_widths( ts_adj_bl , np.array([peak_x_bl]) , rel_height=1 )

    # _x : x values : indices of the wave ( ts )
        # you may use them for slicing the wave to get the y values.

    # [0] : to extract the actual 'value' : see the top explanation.
    prominence_bl = prominence_array_bl[0]
    base_left_x_bl = base_left_array_bl[0]
    base_right_x_bl = base_right_array_bl[0]

    width_bl = width_array_bl[0]
    width_height_bl = width_height_array_bl[0]
    left_ips_x_bl = left_ips_array_bl[0]
    right_ips_x_bl = right_ips_array_bl[0]


    # %%

    base_left_y_bl = ts_adj_bl[base_left_x_bl]
    base_right_y_bl = ts_adj_bl[base_right_x_bl]

    # %%

    left_arm_bl = peak_y_bl - base_left_y_bl
    right_arm_bl = peak_y_bl - base_right_y_bl 

    # p_t : peak-to-trough : x & y values.
    if left_arm_bl > right_arm_bl :
        p_t_y_bl = left_arm_bl
        p_t_x_bl = peak_x_bl - base_left_x_bl
    else :
        p_t_y_bl = right_arm_bl 
        p_t_x_bl = base_right_x_bl - peak_x_bl

    # %%

    # ms : millisecond
    # convert the x values from 'sample' to 'ms'.
    # this is needed for plotting, since the x axis of plots are in ms unit.
    peak_x_ms_bl = peak_x_bl * 0.4
    base_left_x_ms_bl = base_left_x_bl * 0.4
    base_right_x_ms_bl = base_right_x_bl * 0.4

    # %%
    # %%
        # peak-to-trough : y distance & x distance will be plotted as a line.
        # the trace : 4 conditions exist :  * 
            # positive & negative waves
            # left or right arm being longer.

    if polarity_bl == 'pos' :
        ax_bottom[0].vlines( x=peak_x_ms_bl + obl , ymin=( peak_y_bl - p_t_y_bl ) , ymax=peak_y_bl , color='red' ) 
        if left_arm_bl > right_arm_bl :
            ax_bottom[0].hlines(  y=( peak_y_bl - p_t_y_bl ) , 
                                xmin=base_left_x_ms_bl + obl , 
                                xmax=peak_x_ms_bl + obl , 
                                color='red'  )
        else :
            ax_bottom[0].hlines(  y=( peak_y_bl - p_t_y_bl ) , 
                                xmin=peak_x_ms_bl + obl , 
                                xmax=base_right_x_ms_bl + obl , 
                                color='red'  )
    else :
        ax_bottom[0].vlines( x=peak_x_ms_bl + obl , ymin=-peak_y_bl , ymax=( -peak_y_bl + p_t_y_bl ) , color='red' )
        if left_arm_bl > right_arm_bl :
            ax_bottom[0].hlines(  y=( -peak_y_bl + p_t_y_bl ) , 
                                xmin=base_left_x_ms_bl + obl , 
                                xmax=peak_x_ms_bl + obl , 
                                color='red'  )
        else :
            ax_bottom[0].hlines(  y=( -peak_y_bl + p_t_y_bl ) , 
                                xmin=peak_x_ms_bl + obl , 
                                xmax=base_right_x_ms_bl + obl , 
                                color='red'  )
    
# %%

    # plotting the baseline trace.
    ax_bottom[0].plot( x_baseline , smp_bl_m )
    ax_bottom[0].set_ylim(-100 , 100)   # this number is to equalize the y-limits for all channels.  
        # It may need to be modified according to actual values of the channels after exploring the pdf.s.
    
    ax_bottom[0].set_xticks( ticks=x_ticks_baseline )
    ax_bottom[0].tick_params( axis='x' , labelsize=14 )
    
    ax_bottom[0].set_xlabel('time after the stimulus in soi-8 (ms) ' , loc='right' , fontsize=14 )
    
    ax_bottom[0].set_title( 'baseline trace' , fontsize=14 )
   
# %%
# %%
# %%
# %%
# %%

# without baseline correction.
    # since there is only 1 baseline line for all sois ( on contrast to multiunits in which every soi has its own baseline )
        # the only difference with the baseline correction is a y shift.
        # hence, the tau value doesn't difere, but A & t0 differ.


    # try :
    #     popt_raw, pcov_raw = curve_fit( fit_func_3p , sois , p_t_y_8_soi )
    # except RuntimeError :
    #     popt_raw = np.array([ 0.5 , 0 ])
    #     pcov_raw = 0
    #     df.loc[ v , 'err_fit_raw' ] = 1

    # # y of the fitted curve based on the formerly derived parmeters (popt)
    # y_fit_raw = fit_func_3p(sois , *popt_raw)
    
    # # r2_score : goodness of fit.
    # r2s_raw = r2_score( p_t_y_8_soi , y_fit_raw )
    
    # y_fit_raw_cont = fit_func_3p( x_cont , *popt_raw )
    
# %%

# coontinued (without baseline correction). see above

#     y_lim_fit = 1.1 * np.max(p_t_y_8_soi)

#     ax_bottom[1].scatter( sois , p_t_y_8_soi , s=100 , color='blue' , label='actual response' )
#     ax_bottom[1].plot( x_cont , y_fit_raw_cont , color='k' , 
#                label= 'fit_raw: A =' + str(np.around( popt_raw[0], decimals=3)) + 
#                ' , τ_raw=' + str(np.around( popt_raw[1], decimals=3))
# )


#     ax_bottom[1].set_ylim( 0 , y_lim_fit )   # this number is to equalize the y-limits for all channels.  
#         # It may need to be modified according to actual values of the channels after exploring the pdf.s.   
#     ax_bottom[1].set_xticks(ticks=sois)
#     ax_bottom[1].tick_params(axis='x' , labelrotation=45 , labelsize=10)
   
#     ax_bottom[1].set_xlabel('soi(ms)' , loc='right')
#     ax_bottom[1].set_title('peak_to_trough , absolute : actual response _ fit \n '
#                     'r2_score : ' + str(np.around(r2s_raw , decimals=2)) 
#                            , fontsize=11)
    
#     ax_bottom[1].legend( fontsize=8 )  
   
# %%
# %%

    # bc : with baseline correction !
    p_t_y_8_soi_bc = p_t_y_8_soi - p_t_y_bl

# %%

    # the default lm ( Levenberg-Marquardt ) method.
    try :
        popt_bc, pcov_bc = curve_fit( fit_func_3p , sois , p_t_y_8_soi_bc )
        # y of the fitted curve based on the formerly derived parmeters (popt)
        y_fit_bc = fit_func_3p(sois , *popt_bc)
        # r2_score : goodness of fit.
        r2s_bc = r2_score( p_t_y_8_soi_bc , y_fit_bc )
        
    except RuntimeError :
        popt_bc = np.array([ 0.5 , 0 , 0.05 ])
        pcov_bc = 0
        y_fit_bc = np.zeros(8)
        r2s_bc = 0
        df.loc[ v , 'err_fit_bc' ] = 1

    # for plotting a continuous fitted line.
    y_fit_bc_cont = fit_func_3p( x_cont , *popt_bc)
    

    # %%
    
    # db : dogbox
    
    try :
        popt_bc_db , pcov_bc_db = curve_fit( fit_func_3p , sois , p_t_y_8_soi_bc , method='dogbox' )
        # y of the fitted curve based on the formerly derived parmeters (popt)
        y_fit_bc_db = fit_func_3p(sois , *popt_bc_db)
        # r2_score : goodness of fit.
        r2s_bc_db = r2_score( p_t_y_8_soi_bc , y_fit_bc_db )
        
    except RuntimeError :
        popt_bc_db = np.array([ 0.5 , 0 , 0.05 ])
        pcov_bc_db = 0
        y_fit_bc_db = np.zeros(8)
        r2s_bc_db = 0
        df.loc[ v , 'err_fit_bc' ] = 1

    # for plotting a continuous fitted line.
    y_fit_bc_cont_db = fit_func_3p( x_cont , *popt_bc_db)
    

# %%

    # here, it compares the results of the 2 fitting methods (lm & dogbox) & selects the one with a better fit. 
    if r2s_bc_db > r2s_bc :
        popt_bc = popt_bc_db
        pcov_bc = pcov_bc_db
        y_fit_bc = y_fit_bc_db
        y_fit_bc_cont = y_fit_bc_cont_db
        r2s_bc = r2s_bc_db
        df.loc[ v , 'err_fit_bc' ] = 0
    else :
        pass # this means : don't do anything !
    

# %%

# bc : baseline corrected

    ax_bottom[1].scatter( sois , p_t_y_8_soi_bc , s=100 , color='blue' , label='actual response' )
    ax_bottom[1].plot( x_cont , y_fit_bc_cont , color='k' , 
               label= 
               'fit' 
               '\n A =' + str(np.around( popt_bc[0], decimals=3)) + 
               '\n τ=' + str(np.around( popt_bc[1], decimals=3)) +
               '\n t0=' + str(np.around( popt_bc[2], decimals=3))
)

    ax_bottom[1].set_ylim( 0 , 300 )   # this number is to equalize the y-limits for all channels.  
        # It may need to be modified according to actual values of the channels after exploring the pdf.s.   
    ax_bottom[1].set_xticks(ticks=sois)
    ax_bottom[1].tick_params(axis='x' , labelrotation=45 , labelsize=10)
   
    ax_bottom[1].set_xlabel('soi(ms)' , loc='right')
    ax_bottom[1].set_title('peak_to_trough , baseline subtracted _ fit \n '
                    'r2_score : ' + str(np.around(r2s_bc , decimals=2)) 
                           , fontsize=14)
    
    ax_bottom[1].legend( fontsize=8 )  
   
# %%
# %%

    df.loc[ v:v , 'smp_8_soi' ] = pd.Series(data=[smp_8_soi] , index=[v])

    df.loc[ v:v , 'bound_x' ] = pd.Series( data=[bound_x] , index=[v] )
    df.loc[ v , 'bound_x_l_t' ] = bound_x_l_t
    df.loc[ v , 'bound_x_r_t' ] = bound_x_r_t
    
    df.loc[ v:v , 'p_t_x_ms_8_soi' ] = pd.Series( data=[p_t_x_ms_8_soi] , index=[v] )
    df.loc[ v:v , 'p_t_y_8_soi' ] = pd.Series( data=[p_t_y_8_soi] , index=[v] )

    df.loc[ v:v , 'peak_x_ms_8_soi' ] = pd.Series( data=[peak_x_ms_8_soi] , index=[v] )
    df.loc[ v:v , 'base_left_x_ms_8_soi' ] = pd.Series( data=[base_left_x_ms_8_soi] , index=[v] )
    df.loc[ v:v , 'base_right_x_ms_8_soi' ] = pd.Series( data=[base_right_x_ms_8_soi] , index=[v] )
    
    df.loc[ v:v , 'peak_y_8_soi' ] = pd.Series( data=[peak_y_8_soi] , index=[v] )
    df.loc[ v:v , 'base_left_y_8_soi' ] = pd.Series( data=[base_left_y_8_soi] , index=[v] )
    df.loc[ v:v , 'base_right_y_8_soi' ] = pd.Series( data=[base_right_y_8_soi] , index=[v] )

    df.loc[ v:v , 'polarity_8_soi' ] = pd.Series( data=[polarity_8_soi] , index=[v] )

    # from baseline-corrected data.
    df.loc[ v , 'A' ] = popt_bc[0]
    df.loc[ v , 'Tau' ] = popt_bc[1]  
    df.loc[ v , 't0' ] = popt_bc[2]
    df.loc[ v , 'r2s' ] = r2s_bc
    
    df.loc[ v , 'p_t_y_bl' ] = p_t_y_bl
    df.loc[ v , 'p_t_x_bl' ] = p_t_x_bl
    
    # %%
    # %%
    
    plt.suptitle( 'v_' +  str(v)  + '  -  channel ' +  str(v+1) + 
                     '  _  soi order : ' + str(soi_order_numeric) + '\n' + 
                     description_session , 
                     fontsize=16 
)
    
    plt.savefig( dest_dir + '/v_' +  str(v)  + '.pdf' )
    plt.close()
    
# %%

mergedObject = PdfFileMerger()

#   change the range correspnding to the channels you think are catching signals.
for fileNumber in ( list(range(0,384, 2)) + list(range(1,384, 2)) ):
    mergedObject.append(PdfFileReader( dest_dir + '/v_' + str(fileNumber) + '.pdf'))

mergedObject.write( dest_dir + '/total.pdf')

# %%
# %%

df.to_pickle( dest_dir + '/df_lfp.pkl')

# %%


# strange error :

        # Traceback (most recent call last):
        
        #   File ~/anaconda3/envs/env_18/lib/python3.11/site-packages/spyder_kernels/py3compat.py:356 in compat_exec
        #     exec(code, globals, locals)
        
        #   File ~/groups/PrimNeu/Aryo/analysis/fit/lfp/lfp_23.py:65
        #     rd_pps_lfp = rd_prb_f.save(folder= dest_dir + '/pps_lfp' ,  format='binary' , n_jobs=-1, chunk_size=2500  )
        
        #   File ~/anaconda3/envs/env_18/lib/python3.11/site-packages/spikeinterface/core/base.py:845 in save
        #     loaded_extractor = self.save_to_folder(**kwargs)
        
        #   File ~/anaconda3/envs/env_18/lib/python3.11/site-packages/spikeinterface/core/base.py:931 in save_to_folder
        #     cached = self._save(folder=folder, verbose=verbose, **save_kwargs)
        
        #   File ~/anaconda3/envs/env_18/lib/python3.11/site-packages/spikeinterface/core/baserecording.py:462 in _save
        #     write_binary_recording(self, file_paths=file_paths, dtype=dtype, **job_kwargs)
        
        #   File ~/anaconda3/envs/env_18/lib/python3.11/site-packages/spikeinterface/core/recording_tools.py:128 in write_binary_recording
        #     assert Path(file_path).is_file()
        
        # AssertionError

# this error is because I had deleted a forder pps_lfp preprocessing folder inside the destination folder, 
# but seemingly it still had some memory trace in it.
# I ultimately created a new super-set folder for this error to be resolved !
# note : deleting the old folder & creating another folder with the same name also fails ! : create a folder with a different name !!

# %%

    # mergedObject.append(PdfFileReader( dest_dir + '/v_' + str(fileNumber) + '.pdf'))
    # ^
# IndentationError: expected an indented block after 'for' statement on line 400

# %%

# save the triggers.
# np.save( dest_dir + '/t_lfp_8_100.npy' , t_lfp_8_100 )
