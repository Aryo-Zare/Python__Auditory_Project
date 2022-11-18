
# import numpy as np
# import sounddevice as sd

####################


d = 0.0001	#	duration of the click (seconds , when the sampling rate is defined by you, the device will understand what 3 seconds means !)	
dt = 0.002    #   duration of the trigger.

sr = 50000	#	sampling rate. sampling rate of 48000 makes error !
rpt = 100	#	If you change the repeatition to 10000, the program crashes.


###################

#	increment factor
a = (600/11)**(1/7)

soi_1 = 0.11	
soi_2 = soi_1 * a
soi_3 = soi_1 * (a**2)
soi_4 = soi_1 * (a**3)
soi_5 = soi_1 * (a**4)
soi_6 = soi_1 * (a**5)
soi_7 = soi_1 * (a**6)
soi_8 = 6


#	itl = interval

itl_1 = soi_1 - d	
itl_2 = soi_2 - d
itl_3 = soi_3 - d
itl_4 = soi_4 - d
itl_5 = soi_5 - d
itl_6 = soi_6 - d
itl_7 = soi_7 - d
itl_8 = soi_8 - d


#####################


#	trg_itl = triggers interval.

trg_itl_1 = soi_1 - dt	 #   This makes trouble.
trg_itl_2 = soi_2 - dt
trg_itl_3 = soi_3 - dt
trg_itl_4 = soi_4 - dt
trg_itl_5 = soi_5 - dt
trg_itl_6 = soi_6 - dt
trg_itl_7 = soi_7 - dt
trg_itl_8 = soi_8 - dt



####################

#	duration of the tone is 'd' here.
#	if 'int' would not be inserted, this error would emerge : 
#	TypeError: 'float' object cannot be interpreted as an integer



######################################
######################################


#	if not mentioning 'int', this error would result : 
#	TypeError: 'float' object cannot be interpreted as an integer
silence_1 = np.zeros(int(itl_1*sr))
silence_2 = np.zeros(int(itl_2*sr))
silence_3 = np.zeros(int(itl_3*sr))
silence_4 = np.zeros(int(itl_4*sr))
silence_5 = np.zeros(int(itl_5*sr))
silence_6 = np.zeros(int(itl_6*sr))
silence_7 = np.zeros(int(itl_7*sr))
silence_8 = np.zeros(int(itl_8*sr))


####################

#  not necessarily all the tone-noise period should be on trigger.
#  only the START is important.
#	list(range(10))
#	Out[40]: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]	

# clk = click  (sound)
clk = np.ones(int(d*sr))


#############

#	ss = click sound + silence. vectoral concatenation.
ss_1 = np.concatenate((clk,silence_1) , axis=None)
ss_2 = np.concatenate((clk,silence_2) , axis=None)
ss_3 = np.concatenate((clk,silence_3) , axis=None)
ss_4 = np.concatenate((clk,silence_4) , axis=None)
ss_5 = np.concatenate((clk,silence_5) , axis=None)
ss_6 = np.concatenate((clk,silence_6) , axis=None)
ss_7 = np.concatenate((clk,silence_7) , axis=None)
ss_8 = np.concatenate((clk,silence_8) , axis=None)

#	sound channel containing the total number of repeatitions.
sound_channel_1 = np.tile(ss_1 , rpt)
sound_channel_2 = np.tile(ss_2 , rpt)
sound_channel_3 = np.tile(ss_3 , rpt)
sound_channel_4 = np.tile(ss_4 , rpt)
sound_channel_5 = np.tile(ss_5 , rpt)
sound_channel_6 = np.tile(ss_6 , rpt)
sound_channel_7 = np.tile(ss_7 , rpt)
sound_channel_8 = np.tile(ss_8 , rpt)


#########################
#########################



#  'dt instead of d' :  because the trigger need not to be that short like the click, & also it's impossible with the current sampling rate.
trig = np.ones(int(dt*sr))




#	flat period of trigger (corresponding to the silence of the tone).
zeros_1 = np.zeros(round(trg_itl_1*sr))     #  putting 'round' instead of 'int' converts the number 499 to 500.
zeros_2 = np.zeros(int(trg_itl_2*sr))
zeros_3 = np.zeros(int(trg_itl_3*sr))
zeros_4 = np.zeros(int(trg_itl_4*sr))
zeros_5 = np.zeros(int(trg_itl_5*sr))
zeros_6 = np.zeros(int(trg_itl_6*sr))
zeros_7 = np.zeros(int(trg_itl_7*sr))
zeros_8 = np.zeros(int(trg_itl_8*sr))


#####################


#	tz = trigger + zeros	. vectoral concatenation.
tz_1 = np.concatenate((trig,zeros_1) , axis=None)
tz_2 = np.concatenate((trig,zeros_2) , axis=None)
tz_3 = np.concatenate((trig,zeros_3) , axis=None)
tz_4 = np.concatenate((trig,zeros_4) , axis=None)
tz_5 = np.concatenate((trig,zeros_5) , axis=None)
tz_6 = np.concatenate((trig,zeros_6) , axis=None)
tz_7 = np.concatenate((trig,zeros_7) , axis=None)
tz_8 = np.concatenate((trig,zeros_8) , axis=None)


#	trigger channel containing the total number of repeatitions.
trigger_channel_1 = np.tile(tz_1 , rpt)
trigger_channel_2 = np.tile(tz_2 , rpt)
trigger_channel_3 = np.tile(tz_3 , rpt)
trigger_channel_4 = np.tile(tz_4 , rpt)
trigger_channel_5 = np.tile(tz_5 , rpt)
trigger_channel_6 = np.tile(tz_6 , rpt)
trigger_channel_7 = np.tile(tz_7 , rpt)
trigger_channel_8 = np.tile(tz_8 , rpt)

#############

#	st = 2 channels : sound & trigger : Axial channel concatenation :
st_1 = np.concatenate(([sound_channel_1],[trigger_channel_1]) , axis=0)
st_2 = np.concatenate(([sound_channel_2],[trigger_channel_2]) , axis=0)
st_3 = np.concatenate(([sound_channel_3],[trigger_channel_3]) , axis=0)
st_4 = np.concatenate(([sound_channel_4],[trigger_channel_4]) , axis=0)
st_5 = np.concatenate(([sound_channel_5],[trigger_channel_5]) , axis=0)
st_6 = np.concatenate(([sound_channel_6],[trigger_channel_6]) , axis=0)
st_7 = np.concatenate(([sound_channel_7],[trigger_channel_7]) , axis=0)
st_8 = np.concatenate(([sound_channel_8],[trigger_channel_8]) , axis=0)


st_all = [ st_1 ,st_2, st_3, st_4, st_5, st_6, st_7, st_8 ]

random.shuffle(st_all)


#	putting a bracket arround I or trig, is for changing them from a vector with 1 dimenstion, to an array with 2 dimensions. Remember: a 1 column or 1 row array, has 2 dimensions, unlike a vector. 
#	Otherwise concatenation can not be done correctly.

##########

#	iti = inter-train interval
#	here : 60 s
iti = np.zeros((2 , int(60*sr)))

#	iti.shape 	: 	(2, 960000)

###########

total = np.concatenate((st_all[0] , iti , st_all[1] , iti , st_all[2] , iti , st_all[3] , iti , st_all[4] , iti , st_all[5] , iti , st_all[6] , iti , st_all[7]) , axis=1)

##################

#	the above 'total' array is of the shape (2,n). i.e. having 2 rows.
#	sounddevice splits channels into columns. Hence this array should be transposed to get in the shape (n,2).
total_T = total.T


###################

#   initial silence.
#   here : 4s.
#   this is to prevent initial glitch at the loudspeaker to disturb the adaptation while starting the paradigm.
insl = np.zeros((4*sr , 2))

#   initial silence added to the whole sequence.
insl_total_T = np.concatenate((insl , total_T) , axis=0)

#################

sd.play(insl_total_T,sr)	

#	to check the plot :
#	plt.plot(tt_T)


'''

stg software : download & start the program.
turn your volume to max.


'''


