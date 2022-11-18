
#   or directly play a pre-saved array :
# insl_bp_410_trg_T = np.load(r'D:\python\band noise\3\insl_bp_410_trg_T.npy')
# sd.play(insl_bp_410_trg_T , sr)


##################


#   load the below line once in the Ipython console.
#   lh_410 = np.load(r'D:\python\band noise\3\lh_410.npy')


d = 0.1	#	duration of the tone (seconds , when the sampling rate is defined by you, the device will understand what 3 seconds means !)	
ds = 0.25	# duration of silence.

sr = 48000	#	sampling rate

#   number of frequencies within a bandpass range.
#   to be created & added.
nf = 400


###############

#   dots_duratio here is analogous to total number of samples within a definite period of time & is fixed.

dots = np.arange(0,d,1/sr)	

silence = np.zeros(int(ds*sr))


#   banded-noise , ramped , with silence.
#   includes the broad-band range.
bp_410 = np.array([])


################

#   ramp.

#	rpr = ramp__plateu__ramp  
rpr = np.ones(int(d*sr))

#	ascending ramp. 	
#	1st 7ms.
r_ns = int(0.007 * sr)   # number of samples of ramp.
rpr[0: r_ns ] = np.linspace(0,1,r_ns)

#	descending ramp.
#	last 7ms.
rpr[ -r_ns : ] = np.linspace(1,0,r_ns)

#############



#   sf_rf : single freqeuncy (inside the range) , random phase.

for n in range (lh_410[:,0].size) :
    f_range = np.linspace( lh_410[n,0] , lh_410[n,1] , nf)
    bn = np.zeros(int(d*sr))
    for f in f_range : 
        r_phase = np.random.random(1) * 2 * pi
        sf_rf = np.sin( r_phase + (2*pi*f*dots)) 
        bn = bn + sf_rf
    bn_r = bn * rpr 	#	ramped, tone vector.
    bn_r_s = np.concatenate((bn_r , silence) , axis=None)
    bp_410 = np.append(bp_410 , bn_r_s ) 



#   to make the maximum intensity 1.
#   this prevents distortion.
#   60 = np.max(bp_410)
bp_410_60 = bp_410/60

#############

#   trigger for 1 sound & 1 silence.
#   sound can be banded noise or broad band noise.
trg = np.zeros(int((d + ds)*sr))
trg[:4000] = 1      #   This is the right way.
                            #   instead of a loop.


trg_410 = np.tile(trg , 410)

##################

#   bn_br : banded noise _ broad-band noise. + trg.
bp_410_trg = np.concatenate( ( [bp_410_60] , [trg_410] ) , axis=0)

bp_410_trg_T = bp_410_trg.T



#############

#   initial silence.
#   here : 4s.
#   this is to prevent initial glitch at the loudspeaker to disturb the adaptation while starting the paradigm.
insl = np.zeros((4*sr , 2))


insl_bp_410_trg_T = np.concatenate((insl , bp_410_trg_T) , axis=0)

################

sd.play(insl_bp_410_trg_T , sr)
