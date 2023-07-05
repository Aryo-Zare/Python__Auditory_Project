

#  these need only to be run once.
# /home/azare/groups/PrimNeu/Aryo/analysis/General


# %%

# this is for extracting a single recording out of a combined recording.
# this can equally be used for single or multi-units.
# the recordings (rd [total] or some of them combined ) are not used in either single or multi-units.
    # the sample numbers (total or some of them combined) are only used : 
    # these sample numbers can be saved for later use !


# all recordings should be combined here irrespoective of which one you want to analyze for fitting.
# you can copy these from dell/D/files_sort.doc
dr_1 = r'/home/azare/groups/PrimNeu/Aryo/copy_data/Lucy_20221219/2022-12-19 _ Lucy _ terminal/P2/1/2022-12-20_00-56-59'
dr_2 = r'/home/azare/groups/PrimNeu/Aryo/copy_data/Lucy_20221219/2022-12-19 _ Lucy _ terminal/P2/3/2022-12-20_01-03-23'
dr_3 = r'/home/azare/groups/PrimNeu/Aryo/copy_data/Lucy_20221219/2022-12-19 _ Lucy _ terminal/P2/5/2022-12-20_01-42-43'
dr_4 = r'/home/azare/groups/PrimNeu/Aryo/copy_data/Lucy_20221219/2022-12-19 _ Lucy _ terminal/P2/6/2022-12-20_02-18-35'
dr_5 = r'/home/azare/groups/PrimNeu/Aryo/copy_data/Lucy_20221219/2022-12-19 _ Lucy _ terminal/P2/7/2022-12-20_02-56-58'
dr_6 = r'/home/azare/groups/PrimNeu/Aryo/copy_data/Lucy_20221219/2022-12-19 _ Lucy _ terminal/P2/8/2022-12-20_03-00-59'


# dr_6 = r'/home/azare/groups/PrimNeu/Aryo/copy_data/Lucy_20221219/2022-12-19 _ Lucy _ terminal/P15/8/2022-12-21_16-21-30'
# dr_7 = r'/home/azare/groups/PrimNeu/Aryo/copy_data/Lucy_20221219/2022-12-19 _ Lucy _ terminal/P15/9/2022-12-21_16-27-13'


# all recordings should be combined here irrespoective of which one you want to analyze for fitting.
# for multi-block experiments : block_index=0  :  more info  =>  pipe_sort_n.py
rd_1 = se.read_openephys(dr_1 , stream_id='0')
rd_2 = se.read_openephys(dr_2 , stream_id='0')
rd_3 = se.read_openephys(dr_3 , stream_id='0')
rd_4 = se.read_openephys(dr_4 , stream_id='0')
rd_5 = se.read_openephys(dr_5 , stream_id='0')
rd_6 = se.read_openephys(dr_6 , stream_id='0')
# rd_6 = se.read_openephys(dr_6 , stream_id='0')
# rd_7 = se.read_openephys(dr_7 , stream_id='0')


# This is the only variable that should be adjusted for each single recording.
#      should be changed for each recording.  
#      except from the 1st sub-recording in which all this cell shoud be run, for the 2nd & the following subrecordings, only run this line.
# for recording #n, you should combine all n-1 recordings before that.
# used for the trigger.
sample_correction = \
    rd_1.get_num_samples() + \
    rd_2.get_num_samples() + \
    rd_3.get_num_samples() #+ \
    # rd_4.get_num_samples() + \
    # rd_5.get_num_samples() + \
    # rd_6.get_num_samples() + \
    # rd_7.get_num_samples()
    

# used for vec_p_c (unit_fit_n.py).
# you should keep this always as the combination of all recordings irrespective of which one you want to analyze.
rd = si.concatenate_recordings( [ rd_1 , rd_2 , rd_3 , rd_4 , rd_5 , rd_6  ] )

# number of samples in the whole (all of the) combined recording.
# you should keep this always as the combination of all recordings irrespective of which one you want to analyze.
nsacr = rd.get_num_samples()

# %%  

def fit_func(SOI, A, tau_0):
    return A*(1-np.exp(-(SOI-0.05)/tau_0))

def fit_func_3p(SOI, A, tau_0 , t0):
    return A*(1-np.exp(-(SOI-t0)/tau_0))


sois = np.array([ 0.11 , 0.195 , 0.345 , 0.611 , 1.081 , 1.914 , 3.388 , 6])

# %%  

# tuning curve.


# for the first time to run, you don't need to run tc_first.py :
# only run the annotated part below :  index_40_10 , et_int , c,d
#   1-file-all strategy.

index_40_10 = np.load(r'/home/azare/groups/PrimNeu/Aryo/analysis/General/index_40_10.npy')

et_int = np. array([   78,    90,   104,   120,   138,   159,   183,   211,   244,
          281,   324,   373,   430,   496,   572,   659,   760,   876,
        1010,  1164,  1342,  1547,  1784,  2056,  2370,  2732,  3150,
        3631,  4186,  4825,  5563,  6413,  7392,  8522,  9824, 11325,
        13055, 15050, 17349, 20000])

#	conversion factors from samples to ms & considering the before event (0) range.
c = 300/9000
d = 100


# %% 

# tuning curve _ 2 : 
# run this if you changed the tone frequencies.
# then run the tc_on_site ... .py file as usual.

cf = 1250	#	1250 : center frequency
ob = 8	#	8 : octave band
nt = 39	#	number of tones : actually the number of tones generated in this code would be : nt+1 

####	number of repeatitions has been fixed as 10 here, without defining any variable !

ea = np.power(2, np.linspace(-(ob/2),(ob/2),(nt+1)))  		#	exponential array.

et = np.array([]) 	#	exponential tones.  empty array.
for i in ea:
 	et = np.append(et , np.around(cf*i)) 	#	exponential tones.

et_int = et.astype(int)


# %%


# band-pass noise.

# defining each trigger's identity.
index_41_10_bp_br = np.load(r'/home/azare/groups/PrimNeu/Aryo/analysis/bpn/index_41_10_bp_br.npy')

#	bands to be printed as y-labels.
bn_ob = np.load(r'/home/azare/groups/PrimNeu/Aryo/analysis/bpn/bn_ob.npy')

#	conversion factors from samples to ms & considering the before event (0) range.
c = 300/9000
d = 100

########################

