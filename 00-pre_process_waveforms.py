import numpy as np
import struct
import os
import soundfile as sf
from multiprocessing import Process

def pre_emp(x):
    return np.asarray(x[1:] - 0.97 * x[:-1], dtype=np.float32)

def extract_waveforms(lines, dir_trg):

    f_scp = open(dir_trg + '.scp', 'w')
    f_ark = open(dir_trg + '.ark', 'wb')
    f_scp_pe = open(dir_trg + '_pe.scp', 'w')
    f_ark_pe = open(dir_trg + '_pe.ark', 'wb')
    for line in lines:
        key, fn = line.strip().split(' ')
        print(key, fn)

        wav ,_= sf.read(fn, dtype='int16')
        pointer = f_ark.tell()
        arkf_dir = os.path.abspath(dir_trg + '.ark').replace('\\', '/')
        arkf_dir = '/'.join(arkf_dir.split('/')[-4:])
        f_scp.write('%s %s %d\n'%(key, arkf_dir, pointer))
        f_ark.write(struct.pack('<i', wav.shape[0]))
        f_ark.write(struct.pack('<%dh'%wav.shape[0], *wav.tolist()))
        
        wav = pre_emp(wav)
        pointer = f_ark_pe.tell()
        arkf_dir = os.path.abspath(dir_trg + '_pe.ark').replace('\\', '/')
        arkf_dir = '/'.join(arkf_dir.split('/')[-4:])
        f_scp_pe.write('%s %s %d\n'%(key, arkf_dir, pointer))
        f_ark_pe.write(struct.pack('<i', wav.shape[0]))
        f_ark_pe.write(struct.pack('<%df'%wav.shape[0], *wav.tolist()))

    f_scp.close()
    f_ark.close()
    f_scp_pe.close()
    f_ark_pe.close()
    
def join_scp(f_dir, nb_proc):
    f_scp = open(f_dir + '.scp', 'w')
    f_scp_pe = open(f_dir + '_pe.scp', 'w')

    for i in range(nb_proc):
        with open(f_dir + '_%d.scp'%i, 'r') as f_read:
            lines = f_read.readlines()
        for line in lines:
            f_scp.write(line)
        with open(f_dir + '_%d_pe.scp'%i, 'r') as f_read:
            lines = f_read.readlines()
        for line in lines:
            f_scp_pe.write(line)

        os.remove(f_dir + '_%d.scp'%i)
        os.remove(f_dir + '_%d_pe.scp'%i)

    f_scp.close()
    f_scp_pe.close()



DB_dir = 'C:/DB/VoxCeleb1/voxceleb1_wav/'
scp_dir = 'C:/DB/VoxCeleb1/feature/waveform_eval/'
dataset = 'dev'

if __name__ == '__main__':
    nb_proc = 2
    if not os.path.exists(scp_dir):
        os.makedirs(scp_dir)

    list_f_dir = []
    for r, ds, fs in os.walk(DB_dir):
        for f in fs:
            fn = '/'.join([r, f]).replace('\\', '/')
            key = '/'.join(fn.split('/')[-2:])

			if key[0] == 'E':
            	list_f_dir.append('%s %s\n'%(key, fn))
    print('='*5 + 'done' + '='*5)
    print(len(list_f_dir))


    list_proc = []
    nb_utt_per_proc = int(len(list_f_dir) / nb_proc)
    for i in range(nb_proc):
        if i == nb_proc - 1:
            lines = list_f_dir[i * nb_utt_per_proc : ]
        else:
            lines = list_f_dir[i * nb_utt_per_proc : (i+1) * nb_utt_per_proc]

        print(len(lines))
        list_proc.append(Process(target = extract_waveforms, args = (lines, scp_dir+'%s_wav_%d'%(dataset, i))))
        print('%d'%i)

#lines = list_f_dir[nb_proc * nb_utt_per_proc : ]
#list_proc.append(Process(target = extract_waveforms, args = (lines, scp_dir+'%s_wav_%d'%(dataset, nb_proc))))

    for i in range(nb_proc):
        list_proc[i].start()
        print('start %d'%i)
    for i in range(nb_proc):
        list_proc[i].join()

    join_scp(f_dir = scp_dir + '%s_wav'%dataset, nb_proc = nb_proc)
