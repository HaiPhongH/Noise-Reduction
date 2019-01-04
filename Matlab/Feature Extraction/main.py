from __future__ import print_function
import argparse
import os
import essentia
import essentia.standard
import essentia.streaming
from essentia.standard import *
import librosa.feature
import numpy as np
import logging

logging.basicConfig(format='[%(levelname)s|%(asctime)s] %(message)s',
                    datefmt='%Y%m%d %H:%M:%S',
                    level=logging.DEBUG)

global ARGS


def create_weka_mfcc_13():
    """
    Tinh toan 13 he so MFCC dau tien + delta (12 he so) va delta & delta (14 he so) = 39 he so
    """
    global ARGS

    ## ten thu muc can trich chon vector dac trung (RLS, LMS, NLMS, Kalman, Non)
    name = '';
    fout = open('weka/MFCC78_TUNNING_{}_dataset.arff'.format(name), 'w')
    fout.write('@RELATION {}_dataset\n\n'.format(name))

    fout.write('@ATTRIBUTE MEAN_MFCC1	REAL\n')
    fout.write('@ATTRIBUTE MEAN_MFCC2	REAL\n')
    fout.write('@ATTRIBUTE MEAN_MFCC3	REAL\n')
    fout.write('@ATTRIBUTE MEAN_MFCC4	REAL\n')
    fout.write('@ATTRIBUTE MEAN_MFCC5	REAL\n')
    fout.write('@ATTRIBUTE MEAN_MFCC6	REAL\n')
    fout.write('@ATTRIBUTE MEAN_MFCC7	REAL\n')
    fout.write('@ATTRIBUTE MEAN_MFCC8	REAL\n')
    fout.write('@ATTRIBUTE MEAN_MFCC9	REAL\n')
    fout.write('@ATTRIBUTE MEAN_MFCC10	REAL\n')
    fout.write('@ATTRIBUTE MEAN_MFCC11	REAL\n')
    fout.write('@ATTRIBUTE MEAN_MFCC12	REAL\n')
    fout.write('@ATTRIBUTE MEAN_MFCC13	REAL\n')
    fout.write('@ATTRIBUTE MEAN_MFCCD1	REAL\n')
    fout.write('@ATTRIBUTE MEAN_MFCCD2	REAL\n')
    fout.write('@ATTRIBUTE MEAN_MFCCD3	REAL\n')
    fout.write('@ATTRIBUTE MEAN_MFCCD4	REAL\n')
    fout.write('@ATTRIBUTE MEAN_MFCCD5	REAL\n')
    fout.write('@ATTRIBUTE MEAN_MFCCD6	REAL\n')
    fout.write('@ATTRIBUTE MEAN_MFCCD7	REAL\n')
    fout.write('@ATTRIBUTE MEAN_MFCCD8	REAL\n')
    fout.write('@ATTRIBUTE MEAN_MFCCD9	REAL\n')
    fout.write('@ATTRIBUTE MEAN_MFCCD10	REAL\n')
    fout.write('@ATTRIBUTE MEAN_MFCCD11	REAL\n')
    fout.write('@ATTRIBUTE MEAN_MFCCD12	REAL\n')
    fout.write('@ATTRIBUTE MEAN_MFCCD13	REAL\n')
    fout.write('@ATTRIBUTE MEAN_MFCCDD1	REAL\n')
    fout.write('@ATTRIBUTE MEAN_MFCCDD2	REAL\n')
    fout.write('@ATTRIBUTE MEAN_MFCCDD3	REAL\n')
    fout.write('@ATTRIBUTE MEAN_MFCCDD4	REAL\n')
    fout.write('@ATTRIBUTE MEAN_MFCCDD5	REAL\n')
    fout.write('@ATTRIBUTE MEAN_MFCCDD6	REAL\n')
    fout.write('@ATTRIBUTE MEAN_MFCCDD7	REAL\n')
    fout.write('@ATTRIBUTE MEAN_MFCCDD8	REAL\n')
    fout.write('@ATTRIBUTE MEAN_MFCCDD9	REAL\n')
    fout.write('@ATTRIBUTE MEAN_MFCCDD10	REAL\n')
    fout.write('@ATTRIBUTE MEAN_MFCCDD11	REAL\n')
    fout.write('@ATTRIBUTE MEAN_MFCCDD12	REAL\n')
    fout.write('@ATTRIBUTE MEAN_MFCCDD13	REAL\n')
    fout.write('@ATTRIBUTE STD_MFCC1	REAL\n')
    fout.write('@ATTRIBUTE STD_MFCC2	REAL\n')
    fout.write('@ATTRIBUTE STD_MFCC3	REAL\n')
    fout.write('@ATTRIBUTE STD_MFCC4	REAL\n')
    fout.write('@ATTRIBUTE STD_MFCC5	REAL\n')
    fout.write('@ATTRIBUTE STD_MFCC6	REAL\n')
    fout.write('@ATTRIBUTE STD_MFCC7	REAL\n')
    fout.write('@ATTRIBUTE STD_MFCC8	REAL\n')
    fout.write('@ATTRIBUTE STD_MFCC9	REAL\n')
    fout.write('@ATTRIBUTE STD_MFCC10	REAL\n')
    fout.write('@ATTRIBUTE STD_MFCC11	REAL\n')
    fout.write('@ATTRIBUTE STD_MFCC12	REAL\n')
    fout.write('@ATTRIBUTE STD_MFCC13	REAL\n')
    fout.write('@ATTRIBUTE STD_MFCCD1	REAL\n')
    fout.write('@ATTRIBUTE STD_MFCCD2	REAL\n')
    fout.write('@ATTRIBUTE STD_MFCCD3	REAL\n')
    fout.write('@ATTRIBUTE STD_MFCCD4	REAL\n')
    fout.write('@ATTRIBUTE STD_MFCCD5	REAL\n')
    fout.write('@ATTRIBUTE STD_MFCCD6	REAL\n')
    fout.write('@ATTRIBUTE STD_MFCCD7	REAL\n')
    fout.write('@ATTRIBUTE STD_MFCCD8	REAL\n')
    fout.write('@ATTRIBUTE STD_MFCCD9	REAL\n')
    fout.write('@ATTRIBUTE STD_MFCCD10	REAL\n')
    fout.write('@ATTRIBUTE STD_MFCCD11	REAL\n')
    fout.write('@ATTRIBUTE STD_MFCCD12	REAL\n')
    fout.write('@ATTRIBUTE STD_MFCCD13	REAL\n')
    fout.write('@ATTRIBUTE STD_MFCCDD1	REAL\n')
    fout.write('@ATTRIBUTE STD_MFCCDD2	REAL\n')
    fout.write('@ATTRIBUTE STD_MFCCDD3	REAL\n')
    fout.write('@ATTRIBUTE STD_MFCCDD4	REAL\n')
    fout.write('@ATTRIBUTE STD_MFCCDD5	REAL\n')
    fout.write('@ATTRIBUTE STD_MFCCDD6	REAL\n')
    fout.write('@ATTRIBUTE STD_MFCCDD7	REAL\n')
    fout.write('@ATTRIBUTE STD_MFCCDD8	REAL\n')
    fout.write('@ATTRIBUTE STD_MFCCDD9	REAL\n')
    fout.write('@ATTRIBUTE STD_MFCCDD10	REAL\n')
    fout.write('@ATTRIBUTE STD_MFCCDD11	REAL\n')
    fout.write('@ATTRIBUTE STD_MFCCDD12	REAL\n')
    fout.write('@ATTRIBUTE STD_MFCCDD13	REAL\n')
    fout.write('@ATTRIBUTE class 	{'+ARGS.labels+'}\n\n')
    
    fout.write('@DATA\n')

    ## cua so
    windowing = Windowing(type='hamming',
                        size=1104,
                        zeroPhase=False)
    
    ## quang pho
    spectrum = Spectrum(size=1104)

    ##khoi tao MFCC
    mfcc = MFCC(highFrequencyBound=4000, ## gioi han tren cua tan so
                inputSize=201, 			 ## kich thuoc pho dau vao
                lowFrequencyBound=0,	 ## gioi han duoi cua tan so
                numberBands=40,			 ## so luong cac dai Mels trong bo loc
                numberCoefficients=13,   ## so luong dau ra cac he so Mel
                sampleRate=16000)		 ## tan so lay mau

    for label in ARGS.labels.split(','): ## duyet cac thu muc giong voi ten nhan

        ## dia chi thu muc
        dir = os.path.join(ARGS.dir, label)

        logging.info('Access folder <{}>'.format(dir))

        for file in sorted(os.listdir(dir)):

        	## duyet cac file .wav
            if file.endswith('.wav'):
                logging.info('Process <{}>'.format(file))
                path = os.path.join(dir, file)
                
                ## doc file am thanh
                loader = MonoLoader(filename=path, sampleRate=ARGS.sampleRate)
                audio = loader()
                cnt = 0

                for window in FrameGenerator(audio, 
                                            frameSize=ARGS.window_length*ARGS.sampleRate/1000, 
                                            hopSize=ARGS.window_stride*ARGS.sampleRate/1000, 
                                            startFromZero=True):
                    mfccs = []
                    for frame in FrameGenerator(window, 
                                                frameSize=ARGS.frame_length*ARGS.sampleRate/1000, 
                                                hopSize=ARGS.frame_stride*ARGS.sampleRate/1000, 
                                                startFromZero=True):
                        s = spectrum(windowing(frame))

                        _, m = mfcc(s)

                        m_delta = librosa.feature.delta(m, order=1) ## dao ham bac 1
                        m_delta_delta = librosa.feature.delta(m, order=2) ## dao ham bac 2

                        m_all = np.concatenate((m, m_delta, m_delta_delta), axis=0) ## them vao chuoi
                        mfccs.append(m_all)
                    mfccs = np.array(mfccs)
                    mfccs_mean = np.mean(mfccs, axis=0)
                    mfccs_std = np.std(mfccs, axis=0)
                    feat = np.concatenate((mfccs_mean, mfccs_std), axis=0).tolist()
                    str_feat = [str(x) for x in feat]
                    line = ','.join(str_feat)+','+label
                    fout.write(line+'\n')
                    cnt = cnt+1
                logging.info('{} samples'.format(cnt))

def main():
    global ARGS
    parser = argparse.ArgumentParser(description='Parser arguments')
    parser.add_argument(
        '--dir',
        type=str, 
        default='dataset/example', ## ten thu muc can trich chon
        help='Root dir of dataset')
    parser.add_argument(
        '--labels',
        type=str,
        default='normal_breath,strong_breath,deep_breath',
        help='Label name of each class')
    parser.add_argument(
        '--sampleRate',
        type=int,
        default=16000,
        help='Sample rate of audio')
    parser.add_argument(    
        '--window_length',
        type=int,
        default=2000,
        help='Window length of each sample in ms')
    parser.add_argument(
        '--window_stride',
        type=int,
        default=100,
        help='Window stride of each sample in ms')
    parser.add_argument(
        '--frame_length',
        type=int,
        default=25,
        help='Frame length of each sample in ms')
    parser.add_argument(
        '--frame_stride',
        	type=int,
        default=10,
        help='Frame stride of each sample in ms')
    parser.add_argument(
        '--feature',
        type=str,
        choices=['mfcc', 'gfcc'],
        default='mfcc',
        help='Type of features')
    ARGS = parser.parse_args()
    if ARGS.feature == 'mfcc':
        create_weka_mfcc_13()
    elif ARGS.feature == 'gfcc':
        create_weka_gfcc()

if __name__ == "__main__":
    main()