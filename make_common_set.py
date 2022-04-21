from __future__ import division
from __future__ import print_function
#import math
import numpy as np
import csv
#import scipy.misc
import pickle
#import pandas as pd
#from shapely.geometry import Polygon
#import gc


RT_min=10.0
path='/data/fzohora/dilution_series_syn_pep/feature_list/'  #'/media/anne/Study/bsi/dilution_series_syn_peptide/feature_list/' #'/data/fzohora/water_raw_ms1/'
dataname=['130124_dilA_1_01','130124_dilA_1_02','130124_dilA_1_03','130124_dilA_1_04', 
'130124_dilA_2_01','130124_dilA_2_02','130124_dilA_2_03','130124_dilA_2_04','130124_dilA_2_05','130124_dilA_2_06','130124_dilA_2_07',
'130124_dilA_8_01','130124_dilA_8_02','130124_dilA_8_03','130124_dilA_8_04',
'130124_dilA_9_01','130124_dilA_9_02','130124_dilA_9_03','130124_dilA_9_04','130124_dilA_10_01','130124_dilA_10_02', '130124_dilA_10_03', '130124_dilA_10_04', '130124_dilA_11_01', '130124_dilA_11_02', '130124_dilA_11_03', '130124_dilA_11_04', '130124_dilA_12_01', '130124_dilA_12_02', '130124_dilA_12_03', '130124_dilA_12_04'] 
feature_file=['130124_dilA_1_01.raw.fea.isotopes.csv','130124_dilA_1_02.raw.fea.isotopes.csv','130124_dilA_1_03.raw.fea.isotopes.csv','130124_dilA_1_04.raw.fea.isotopes.csv','130124_dilA_2_01.raw.fea.isotopes.csv','130124_dilA_2_02.raw.fea.isotopes.csv','130124_dilA_2_03.raw.fea.isotopes.csv','130124_dilA_2_04.raw.fea.isotopes.csv','130124_dilA_2_05.raw.fea.isotopes.csv','130124_dilA_2_06.raw.fea.isotopes.csv','130124_dilA_2_07.raw.fea.isotopes.csv','130124_dilA_8_01.raw.fea.isotopes.csv','130124_dilA_8_02.raw.fea.isotopes.csv','130124_dilA_8_03.raw.fea.isotopes.csv','130124_dilA_8_04.raw.fea.isotopes.csv','130124_dilA_9_01.raw.fea.isotopes.csv','130124_dilA_9_02.raw.fea.isotopes.csv','130124_dilA_9_03.raw.fea.isotopes.csv','130124_dilA_9_04.raw.fea.isotopes.csv','130124_dilA_10_01.raw.fea.isotopes.csv','130124_dilA_10_02.raw.fea.isotopes.csv', '130124_dilA_10_03.raw.fea.isotopes.csv', '130124_dilA_10_04.raw.fea.isotopes.csv', '130124_dilA_11_01.raw.fea.isotopes.csv', '130124_dilA_11_02.raw.fea.isotopes.csv', '130124_dilA_11_03.raw.fea.isotopes.csv', '130124_dilA_11_04.raw.fea.isotopes.csv', '130124_dilA_12_01.raw.fea.isotopes.csv', '130124_dilA_12_02.raw.fea.isotopes.csv', '130124_dilA_12_03.raw.fea.isotopes.csv',  '130124_dilA_12_04.raw.fea.isotopes.csv']#, 'Demo_LC_Chymotrypsin.raw.fea.isotopes.csv',  'Demo_LC_Trypsin.raw.fea.isotopes.csv',   'Demo_HC_AspN.raw.fea.isotopes.csv',   'Demo_HC_Chymotrypsin.raw.fea.isotopes.csv',  'Demo_HC_Trypsin.raw.fea.isotopes.csv'] 
feature_count=[26515, 29696, 30785, 31985, 27345,26585,27750,28193,28474,28335,27475,25294, 22608, 22927, 23756, 22960, 23204, 23859, 23766, 26680, 19483, 24859, 25220, 28409, 25967, 30328, 29802, 32097, 30707, 32444, 33155] 
delim=','
for data_index in range (15,  len(dataname)): # 19, 20, 21
    print(dataname[data_index])
#    

#
#    #-----------------------------------read peptide features summury-------------------------------------#
#    
#
#    #-----------------------------------read peptide features -------------------------------------#
    filename_feature= feature_file[data_index]
    total_feature=feature_count[data_index]
    peptide_feature=np.zeros((total_feature, 16)) #0=mz, 1=rtstsr, 2=rtend, 3=z, 4=auc, 5=kept or removed or non_overlapping_feature_id, 6=end_mz, 7=min_rt, 8=max_rt, 9=endof2ndisotope, 10=start of second isotope, 11=overlapped/not, 12=maxI, 13=meanRT, 14=num of iso, 15=id of maxquant
    isotope_gap=np.zeros((10))
    isotope_gap[0]=0.01
    isotope_gap[1]=1.00
    isotope_gap[2]=0.500
    isotope_gap[3]=0.333
    isotope_gap[4]=0.250
    isotope_gap[5]=0.200
    isotope_gap[6]=0.167
    isotope_gap[7]=0.143
    isotope_gap[8]=0.125
    isotope_gap[9]=0.111
    mz_resolution=3

    avoid=[]
    f = open(path+'PEAKs/'+filename_feature, 'r')
    line=f.readline()
    line=f.readline()
    i=0;
    while line!='':
        temp=line.split(',')
        id=temp[0] # mz, rtstsr, rtend, z, auc
        peptide_feature[i, 0]=round(float(temp[2]), mz_resolution) #mz
        peptide_feature[i, 1]=round(float(temp[8]), 2) #st
        peptide_feature[i, 2]=round(float(temp[9]), 2) #en
        peptide_feature[i, 3]=temp[5] #charge
        peptide_feature[i, 4]=temp[6] #area
        peptide_feature[i, 6]=peptide_feature[i, 0] #??


        peptide_feature[i, 12]=round(float(temp[10]), 2) #PeakI
        peptide_feature[i, 13]=round(float(temp[3]), 2) #meanRT
        line=f.readline() 
        min_rt=peptide_feature[i, 1]
        max_rt=peptide_feature[i, 2]
        isotope_no=0
        while line!='':
            temp=line.split(',')
            if temp[0]!=id:
                break
            #else this isotope belongs to this same peptide        
            isotope_no=isotope_no+1
            if isotope_no==1:
                peptide_feature[i, 9]=round(float(temp[9]), 2) #end_of_second_isotope
                peptide_feature[i, 10]=round(float(temp[8]), 2) #start_of_second_isotope


            peptide_feature[i, 6]=round(peptide_feature[i, 6]+isotope_gap[int(peptide_feature[i, 3])], mz_resolution) #end_mz
            if round(float(temp[8]), 2)<min_rt:
                min_rt=round(float(temp[8]), 2) #st
            if round(float(temp[9]), 2)>max_rt:
                max_rt=round(float(temp[9]), 2) #en

            line=f.readline()  
        peptide_feature[i, 7]=min_rt
        peptide_feature[i, 8]=max_rt
        peptide_feature[i, 14]=isotope_no+1
        if peptide_feature[i, 7]<RT_min: #min_rt
            avoid.append(i)
            peptide_feature[i, 5]=-1
        i=i+1
    f.close() 

    

##################################################################
    # peptide_feature: feature list generated by PEAKs
    # feature_list: feature_list generated by maxQuant

#    feature_list=np.loadtxt(path+'maxQ/'+dataname[data_index]+'_2.csv', delimiter=delim)
    filename ='/data/fzohora/dilution_series_syn_pep/feature_list/maxQ/'+dataname[data_index]+'_2.csv'
    # initializing the titles and peptide_mascot list
    MQ_peptide= [] 
    # reading csv file
    csvfile=open(filename, 'r')
    # creating a csv reader object
    csvreader = csv.reader(csvfile)     
    # extracting each data row one by one
    for row in csvreader:
        MQ_peptide.append(row)
    csvfile.close() 

    feature_list=np.zeros((len(MQ_peptide), len(MQ_peptide[0]) ))    
    for i in range (0, len(MQ_peptide)):
        for j in range (0, len(MQ_peptide[0])):
            try:
                feature_list[i, j]=MQ_peptide[i][j]
            except:
                feature_list[i, j]=0
    
    #both list are sorted on m/z in asc order
    mz_tolerance=0.01
    RT_tolerance=0.03
    mz_resolution=3
    j=0
    count=0
    common_set=[]
    for i in range (0, total_feature):
        if peptide_feature[i, 7]<RT_min:
            continue
        ftr_mz=peptide_feature[i, 0]
        ftr_charge=peptide_feature[i, 3]
        ftr_meanRT=peptide_feature[i, 13]
        while (j<feature_list.shape[0] and round(feature_list[j, 1], mz_resolution)<round(ftr_mz-mz_tolerance, mz_resolution)): 
            j=j+1
            
        found=0
        j2=j
        while (j2<feature_list.shape[0] and round(feature_list[j2, 1], mz_resolution)<=round(ftr_mz+mz_tolerance, mz_resolution)):
            ftr_mz_mq=round(feature_list[j2, 1], mz_resolution)
            d_1=abs(ftr_mz_mq-ftr_mz)
            ftr_mz_avg=(ftr_mz_mq+ftr_mz)/2.0
            d_2=(ftr_mz_avg*10)/10**6
#            print(d_2)
            if d_1<d_2: # 10ppm error tolerance accepted
                mq_rt=round(feature_list[j2, 4], 2)
                if round(ftr_meanRT-RT_tolerance, 2) <= mq_rt and mq_rt <= round(ftr_meanRT+RT_tolerance, 2):
                    if ftr_charge==int(feature_list[j2, 0]):
                        found=1
                        break
            
            j2=j2+1
        
        if found==1:
            common_set.append((i, j2))
            count=count+1
            
    print(count)


    peptide_feature[:, 15]=-1
    for i in range (0, len(common_set)):
        peptide_feature[common_set[i][0], 15]=common_set[i][1] #indexing starts from 0
    

    logfile=open(path+'feature_list/'+dataname[data_index]+'_combineIsotopes_featureList.csv', 'wb')
    np.savetxt(logfile, peptide_feature, delimiter=',')
    logfile.close()


    f=open(path+'common_set/'+dataname[data_index]+'_common_set', 'wb')
    pickle.dump(common_set,  f, protocol=2)
    f.close()


