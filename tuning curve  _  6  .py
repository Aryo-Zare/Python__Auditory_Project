
import numpy as np
import sounddevice as sd
from math import pi


d = 0.2		#	duration (seconds , when the sampling rate is defined by you, the device will understand what 3 seconds means !)
sr = 16000	#	sampling rate


cf = 2000	#	center frequency
ob = 8		#	octave band
nt = 40	#	number of tones : actually the number of tones generated in this code would be : nt+1 

####	number of repeatitions has been fixed as 10 here, without defining any variable !

ea = np.power(2, np.linspace(-(ob/2),(ob/2),(nt+1)))  		#	exponential array.

et = np.array([]) 	#	exponential tones.
for i in ea:
	et = np.append(et , np.around(cf*i)) 	#	exponential tones.


#	vector of 40 random numbers (frequencies (pitch)).
et_r = np.random.choice(et , nt+1 , replace=False)	#	randomized
	

#	x axis ticks !	
dots = np.arange(0,d,1/sr)	

silence = np.zeros(int(d*sr))

#	empty numpy array
s_channel = np.array([])	#	sound channel


#	y axis values (or 'values' (named I (intensity) in the previous code).
#	Here, the whole vector of 40 values are created simultansously by a loop.
for i in range (0 , (nt+1)):	#	41 is the number of items (indices) in et_r.
	I = np.sin(2*pi*dots*et_r[i])		#	tone vector
	ts = np.concatenate((I,silence) , axis=None)	#	tone + silence vector
	s_channel = np.append(s_channel , ts )

s_channel_total = np.tile(s_channel,10)	#	repeatitions.

#	when you change the 'd', be careful to also decrease the number in front of 'range'. 
#	otherwise you will get this error : 'index 3200 is out of bounds for axis 0 with size 3200'
trig = np.zeros(int(d*sr))
for i in range (2000) :	 
	trig[i] = 1

zeros = np.zeros(int(d*sr))	#	flat part of the trigger.
trig_z = np.concatenate((trig,zeros) , axis=None)	#	trigger + the flat(silence ) part of it.



t_channel_total = np.tile(trig_z , ((nt+1)*10))	#	trigger channel (41 * 10)


#	tt = 2 channels : tone & trigger : Axial channel concatenation :
tt = np.concatenate(([s_channel_total],[t_channel_total]) , axis=0)
#	putting a bracket arround I or trig, is for changing them from a vector with 1 dimenstion, to an array with 2 dimensions.
#	Remember: a 1 column or 1 row array, has 2 dimensions, unlike a vector. 
#	Otherwise concatenation can not be done correctly.


#	the above tt array is of the shape (2,n). i.e. having 2 rows.
#	sounddevice splits channels into columns. Hence this array should be transposed to get in the shape (n,2).
tt_T = tt.T

sd.play(tt_T,sr)	

#	to check the plot :
#	plt.plot(tt_T)


# this is to feed the software later for creating PSTH.
pitch_array = np.tile(et_r,10)
index = np.array(range(1,(((nt+1)*10)+1)))
trig_array = np.concatenate(([pitch_array],[index]) , axis=0)	# produces an array with the shape (2,400)
trig_array_columnar = trig_array.T 	#	=> an array with the shape (400,2)
