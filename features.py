import scipy.io as sio
import numpy as np
import glob
import os
import scipy.misc as misc
from sklearn import preprocessing
import scipy as sp
import scipy.signal as spsig
import pandas as pd
import scipy.stats as spstat
import json
import time


# def get_data(file):
#	matfile = sio.loadmat(file)
#	data = (matfile['dataStruct']['data'][0,0]).T
#	return data

def get_data(file):
    try:
        matfile = sio.loadmat(file)
        data = (matfile['data']).T
        return data
    except Exception:
        print('bad file:', file)
        return np.zeros([16, 400 * 10 * 60])


def long_features(pat, outfile, datapath, timer):
    # pat = 3
    # outfile='D:/Semester 3/Research/test_download/train data'+'pat_'+str(pat)+'_long_newtest_sub.csv'
    # file path for the new_test data
    # f = '/mnt/am02_scratch/blang/kaggle_data/test_'+str(pat)+'_new/*mat'
    # file path for the training and hold-out testing
    # f = '/mnt/am02_scratch/blang/kaggle_data/CV/pat_'+str(pat)+'/train/*mat'
    # f = 'D:/Semester 3/Research/test_download/Pat'+str(pat)+'Train/*mat'
    f = datapath + '/*mat'

    pat_num = pat
    ff = glob.glob(f)

    label = [str(os.path.basename(n)) for n in ff]
    print(label)
    # print(len(label))
    output = []
    featureList = []
    mydata = []
    mytimer = []
    bands = [0.1, 4, 8, 12, 30, 70]
    for i in range(len(ff)):
        # print(float(i)/float(len(ff)))
        output = []
        outputtimer = []
        featureList = []
        featureListimer = []
        if os.path.basename(ff[i]) == '1_45_1.mat':
            continue
        data = get_data(ff[i])
        # print(data)
        data = preprocessing.scale(data, axis=1, with_std=True)
        featureList.append('File')
        # featureListimer.append('File')
        output.append(label[i])
        # outputtimer.append(label[i])
        featureList.append('pat')
        # featureListimer.append('pat')
        output.append(pat_num)
        # outputtimer.append(pat_num)
        welsh = []

        # get correlation Coef. this will be 16x16
        # h=np.corrcoef(data)
        # h=np.nan_to_num(h)
        # # only want upper triangle
        # ind = np.triu_indices(16, 1)
        #
        # htri = h[ind]
        # for ii in range(np.size(htri)):
        #         #start = time.time()
        #         featureList.append('coef%i'%(ii))
        #         output.append(htri[ii])
        #         #total_time = time.time() - start
        #         # featureListimer.append('coef%i' % (ii))
        #         # outputtimer.append(total_time)
        #
        #
        # c,v = np.linalg.eig(h)
        # c.sort()
        # c = np.real(c)
        # for e in range(len(c)):
        #         #start = time.time()
        #         featureList.append('coef_timeEig%i'%(e))
        #         output.append(c[e])
        #         # total_time = time.time() - start
        #         # featureListimer.append('coef_timeEig%i'%(e))
        #         # outputtimer.append(total_time)

        for j in range(16):
            hold = spsig.decimate(data[j, :], 5, zero_phase=True)

            start = time.time()
            featureList.append('sigma%i' % (j))
            output.append(hold.std())
            total_time = time.time() - start
            featureListimer.append('sigma%i' % (j))
            outputtimer.append(total_time)

            # start = time.time()
            featureList.append('kurt%i' % (j))
            output.append(spstat.kurtosis(hold))
            # total_time = time.time() - start
            # featureListimer.append('kurt%i'%(j))
            # outputtimer.append(total_time)

            # start = time.time()
            featureList.append('skew%i' % (j))
            output.append(spstat.skew(hold))
            # total_time = time.time() - start
            # featureListimer.append('skew%i'%(j))
            # outputtimer.append(total_time)

            # start = time.time()
            # featureList.append('zero%i'%(j))
            # output.append(((hold[:-1] * hold[1:]) < 0).sum())
            # total_time = time.time() - start
            # featureListimer.append('zero%i'%(j))
            # outputtimer.append(total_time)

            diff = np.diff(hold, n=1)
            diff2 = np.diff(hold, n=2)

            # start = time.time()
            # featureList.append('sigmad1%i'%(j))
            # output.append(diff.std())
            # total_time = time.time() - start
            # featureListimer.append('sigmad1%i'%(j))
            # outputtimer.append(total_time)

            # start = time.time()
            # featureList.append('sigmad2%i'%(j))
            # output.append(diff2.std())
            # total_time = time.time() - start
            # featureListimer.append('sigmad2%i'%(j))
            # outputtimer.append(total_time)

            # start = time.time()
            featureList.append('zerod%i' % (j))
            output.append(((diff[:-1] * diff[1:]) < 0).sum())
            # total_time = time.time() - start
            # featureListimer.append('zerod%i'%(j))
            # outputtimer.append(total_time)

            # start = time.time()
            # featureList.append('zerod2%i'%(j))
            # output.append(((diff2[:-1] * diff2[1:]) < 0).sum())
            # total_time = time.time() - start
            # featureListimer.append('zerod2%i'%(j))
            # outputtimer.append(total_time)

            # start = time.time()
            featureList.append('RMS%i' % (j))
            output.append(np.sqrt((hold ** 2).mean()))
            # total_time = time.time() - start
            # featureListimer.append('RMS%i'%(j))
            # outputtimer.append(total_time)

            # start = time.time()
            f, psd = spsig.welch(hold, fs=80)
            print(f)
            print(psd)
            print('yes')
            # total_time = time.time() - start
            # welsh.append(total_time)

            psd[0] = 0

            # start = time.time()
            featureList.append('MaxF%i' % (j))
            output.append(psd.argmax())
            # total_time = time.time() - start
            # featureListimer.append('MaxF%i'%(j))
            # outputtimer.append(total_time)

            # start = time.time()
            featureList.append('SumEnergy%i' % (j))
            output.append(psd.sum())
            # total_time = time.time() - start
            # featureListimer.append('SumEnergy%i'%(j))
            # outputtimer.append(total_time)

            psd /= psd.sum()
            for c in range(1, len(bands)):
                # start = time.time()
                featureList.append('BandEnergy%i%i' % (j, c))
                output.append(psd[(f > bands[c - 1]) & (f < bands[c])].sum())
                # total_time = time.time() - start
                # featureListimer.append('BandEnergy%i%i'%(j,c))
                # outputtimer.append(total_time)

            # start = time.time()
            # featureList.append('entropy%i'%(j))
            # output.append(-1.0*np.sum(psd[f>bands[0]]*np.log10(psd[f>bands[0]])))
            # total_time = time.time() - start
            # featureListimer.append('entropy%i'%(j))
            # outputtimer.append(total_time)

            # pdb.exit()
            # start = time.time()
            featureList.append('Mobility%i' % (j))
            output.append(np.std(diff) / hold.std())
            # total_time = time.time() - start
            # featureListimer.append('Mobility%i'%(j))
            # outputtimer.append(total_time)

            # start = time.time()
            featureList.append('Complexity%i' % (j))
            output.append(np.std(diff2) * np.std(hold) / (np.std(diff) ** 2.))
            # total_time = time.time() - start
            # featureListimer.append('Complexity%i'%(j))
            # outputtimer.append(total_time)

        mydata.append(pd.DataFrame({'Features': output}, index=featureList).T)
        # mytimer.append(pd.DataFrame({'Features':outputtimer},index=featureListimer).T)

    welsh_df = pd.DataFrame(welsh, columns=["value"])
    # welsh_df.to_csv('welsh.csv')
    trainSample = pd.concat(mydata, ignore_index=True)
    # trainTimer = pd.concat(mytimer, ignore_index=True)
    # trainTimer.to_csv(timer)
    # trainSample.to_csv(outfile)
    return 1


def main():
    feat = json.load(open('SETTINGS.json'))

    keys = list(feat.keys())

    # pat = feat['pat']
    pat = '1'
    print(pat)
    if feat['make_test'] == 3:
        print('test')
        outfile = 'D:/Semester 3/Research/test_download/test data' + 'pat_' + str(pat) + '_long_newtest_sub.csv'
        l = long_features(pat, outfile, feat['test'])

    if feat['make_train'] == 1:
        starting = time.time()
        print('train')
        outfile = 'D:/Semester 3/Research/test_download/train data' + 'pat_' + str(pat) + '_long_newtrain_sub.csv'
        timer = 'D:/Semester 3/Research/test_download/' + 'pat_' + str(pat) + '_long_train_timer.csv'
        l = long_features(pat, outfile, feat['train'], timer)
        outfile = 'D:/Semester 3/Research/test_download/train data' + 'pat_' + str(pat) + '_short_newtrain_sub.csv'

        total_t = time.time() - starting
        print(total_t)
    if feat['make_hold'] == 99:
        print('hold')
        outfile = feat['feat'] + '/pat_' + str(pat) + '_long_test.csv'
        l = long_features(pat, outfile, feat['hold-out'], timer)


if __name__ == "__main__":
    main()
