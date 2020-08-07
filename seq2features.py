import numpy as np
from Bio import motifs
from Bio.Seq import Seq
import sys,time
MakeSampleMotif = False

# basic input
num_sample = int(input("Sample Number (Notice that a sample of twice the value you put in will be made) : "))
seq_length = int(input("Sample Length : "))
num_motif = int(input("Motif Number : "))
mot_length = int(input("Motif Length : "))
target_seq = Seq(input("Target Sequence : (Make sure the same your motif Length) : "))
st = time.time()
tar_length = len(target_seq)

# check right input
if seq_length - tar_length - mot_length < 0 or tar_length != mot_length or num_sample < 0 or seq_length < 0 or num_motif < 0 or mot_length < 0:
    print("Wrong input Try again")
    sys.quit()
    
# make target motif
m = motifs.create([target_seq])
pwm = m.counts.normalize(pseudocounts=0.005)
pwm_arr = np.array(list(pwm.values()))
pwm = np.hstack((np.ones((4,mot_length)), pwm_arr, np.ones((4, seq_length-tar_length-mot_length))))

# make sample & motif
pos = np.array( [np.random.choice(['A', 'C', 'G', 'T'], num_sample, p=pwm[:,i]/sum(pwm[:,i])) for i in range(seq_length)]  ).transpose()
neg = np.array( [np.random.choice(['A', 'C', 'G', 'T'], num_sample, p=np.array([1,1,1,1])/4)for i in range(seq_length)]    ).transpose() 
mot = np.array( [np.random.choice(['A', 'C', 'G', 'T'], num_motif, p=np.array([1,1,1,1])/4)for i in range(mot_length)]     ).transpose()

# if you want, you can make a motif that include target sequence
# change the value "MakeSampleMotif"
if MakeSampleMotif == True:
    for i in range(tar_length):
        mot[0][i] = str(target_seq[i])

# sample one-hot encoding
base_dict = {'A':0, 'C':1, 'G':2, 'T':3}
onehot_encode_pos = np.zeros((num_sample, seq_length, 4))
onehot_encode_pos_label = np.zeros((num_sample, 2), dtype=int)
onehot_encode_pos_label[:,0] = 1
onehot_encode_neg = np.zeros((num_sample, seq_length, 4))
onehot_encode_neg_label = np.zeros((num_sample, 2), dtype=int)
onehot_encode_neg_label[:,1] = 1
for i in range(num_sample):
    for j in range(seq_length):
        onehot_encode_pos[i,j,base_dict[pos[i,j]]] = 1
        onehot_encode_neg[i,j,base_dict[neg[i,j]]] = 1
x = np.vstack((onehot_encode_pos, onehot_encode_neg))
y = np.vstack((onehot_encode_pos_label, onehot_encode_neg_label))
num_sample *= 2 # cause sample is double!

# motif one-hot encoding
onehot_encode_mot = np.zeros((num_motif, mot_length, 4))
for i in range(num_motif):
    for j in range(mot_length):
        onehot_encode_mot[i,j,base_dict[mot[i,j]]] = 1
        
# etc calculation & definition
z = onehot_encode_mot
y_train = y
x_train = np.zeros((num_sample, num_motif), dtype=float)

# make convolution
for i in range(num_sample):
    for j in range(num_motif):
        p = np.zeros((seq_length-mot_length+1),dtype=float)
        for k in range(seq_length-mot_length+1):
            for l in range(mot_length):
                p[k] += float(np.convolve(x[i][k+l],np.flip(z[j][l]),'valid'))
        x_train[i,j] = np.convolve(p,np.flip(p/sum(p)),'valid') # Stochastic pooling
        # x_train[i,j] = np.max(p) # max pooling
print("execution time :", str(round(time.time() - st , 4)), "s")
