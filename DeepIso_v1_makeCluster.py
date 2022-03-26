'''Example: 
You have a file ABC.ms1 at /DeepIsoV1/rawdata/ABC.ms1
You are done with running IsoDetecting on this file. Now you want to run this clustering.  
So put the command as below:
nohup python -u DeepIso_v1_makeCluster.py ABC > output_makeCluster.log &
'''

from __future__ import division
from __future__ import print_function
import pickle
import numpy as np
from collections import deque
from collections import defaultdict
import copy
import sys
import os
########## file run parameters #################################
current_path=os.system("pwd")
datapath=current_path+'/DeepIsoV1/data/'  
modelpath=current_path+'/DeepIsoV1/model/'
file_name=sys.arg[1]


##############################################################
isotope_gap=np.zeros((10))
isotope_gap[0]=0.01
isotope_gap[1]=1.00
isotope_gap[2]=0.50
isotope_gap[3]=0.33
isotope_gap[4]=0.25
isotope_gap[5]=0.20
isotope_gap[6]=0.17
isotope_gap[7]=0.14
isotope_gap[8]=0.13
isotope_gap[9]=0.11

RT_window=15
mz_window=211
total_class=10
RT_unit=0.01
mz_unit=0.01
num_class=10
####################################################################
print('scanning test ms: '+file_name)
print('reading dictionary record from disk, you will get a message after its done')
f=open(datapath+file_name+'_ms1_record', 'rb')
RT_mz_I_dict, maxI=pickle.load(f)
f.close()   
print('disk read done')
print('reading ms1 record from disk, you will get a message after its done')
f=open(datapath+file_name+'_consecutive_scan_MS1_1', 'rb') 
ms1=pickle.load(f)
f.close()
f=open(datapath+file_name+'_consecutive_scan_MS1_2', 'rb') 
ms1_next=pickle.load(f)
f.close()    
print('disk read done')
 
ms1=np.concatenate((ms1, np.copy(ms1_next)), axis=1)
temp_ms1=np.zeros((ms1.shape[0]+RT_window, ms1.shape[1]+mz_window))
temp_ms1[0:ms1.shape[0], 0:ms1.shape[1]]=np.copy(ms1[:, :])
ms1=np.copy(temp_ms1)
temp_ms1=0


###########################
#scan ms1_block and record the cnn outputs in list_dict[z]: hash table based on m/z
#for each m/z
mz_resolution=2
RT_list = np.sort(list(RT_mz_I_dict.keys()))
max_RT=RT_list[len(RT_list)-1]
min_RT=RT_list[0]

sorted_mz_list=[]
RT_index=dict()
for i in range(0, len(RT_list)):
    RT_index[round(RT_list[i], 2)]=i
    sorted_mz_list.append(sorted(RT_mz_I_dict[RT_list[i]]))   
    
max_mz=0
min_mz=1000
for i in range (0, len(sorted_mz_list)):
    mz_I_list=sorted_mz_list[i]
    mz=mz_I_list[len(mz_I_list)-1][0]
    if mz>max_mz:
        max_mz=mz
    mz=mz_I_list[0][0]
    if mz<min_mz:
        min_mz=mz

rt_search_index=0
while(RT_list[rt_search_index]<=min_RT):
    if RT_list[rt_search_index]==min_RT:
        break
    rt_search_index=rt_search_index+1 

total_mz=int(round((max_mz-min_mz+mz_unit)/mz_unit, mz_resolution)) 
total_RT=len(RT_list)-rt_search_index

#############################
print('reading the scanning results, you will get a message once its done')
f=open(datapath+file_name+'_scanning_result'+'_seg_0', 'rb')    
list_dict, stripe_index = pickle.load(f) #all mz_done

f=open(datapath+file_name+'_scanning_result'+'_seg_1', 'rb')     
list_dict_next, stripe_index = pickle.load(f) #all mz_done

for z in range (1, 10):
    list_dict[z].update(list_dict_next[z])

f=open(datapath+file_name+'_scanning_result'+'_seg_2', 'rb')    
list_dict_next, stripe_index = pickle.load(f) #all mz_done

for z in range (1, 10):
    list_dict[z].update(list_dict_next[z])

print('done')
print('starting cluster preparation. You will see results from charge z = 1 to 9')
isotope_cluster=defaultdict(list)
for z in range (1, 10):
    print('charge z=%d'%z)
    list_mz=np.sort(list(list_dict[z].keys()))
    max_dict=len(list_mz)
    for i in range (0, max_dict):#
        mz=round(list_mz[i], 2)
        list_RT_range=list_dict[z][mz] # get list of RT range 
        if len(list_RT_range)==0: #remove the empty list
            list_dict[z].pop(mz)
            continue
        list_dict[z][mz] = deque()
        limit=len(list_RT_range)
        seq_running=0
        rt_pred=list_RT_range.popleft()
        j=1
        while j < limit-1:
            rt_current=list_RT_range.popleft()
            j=j+1
            if rt_current==-1:
                rt_next=list_RT_range.popleft()
                j=j+1
                if RT_index[rt_next]-RT_index[rt_pred]<=1:  #A 
                    if seq_running==0:
                        list_dict[z][mz].append(rt_pred)
                        rt_pred=rt_next
                        list_dict[z][mz].append(rt_pred)
                        seq_running=1
                    else:        
                        list_dict[z][mz].pop()
                        rt_pred=rt_next
                        list_dict[z][mz].append(rt_pred)
                elif seq_running==1:
                    seq_running=0
                    list_dict[z][mz].append(-1)
                    rt_pred=rt_next
                else:
                    rt_pred=rt_next
                    
            elif seq_running==0:
                list_dict[z][mz].append(rt_pred)
                rt_pred=rt_current
                list_dict[z][mz].append(rt_pred)
                seq_running=1
                
            elif seq_running==1:
                list_dict[z][mz].pop()
                rt_pred=rt_current
                list_dict[z][mz].append(rt_pred)
                
        if seq_running==1:
            list_dict[z][mz].append(-1)
        
        
    # Remove the false detections caused by saying YES ahead of time
    # Enclose the ranges in a [start,end,-1] format
    count=0
    list_keys=np.sort(list(list_dict[z].keys()))
    max_dict=len(list_keys)
    for i in range (0, max_dict):#
        mz=round(list_keys[i], 2)
        list_RT_range=list_dict[z][mz] # get list of RT range 
        if len(list_RT_range)==0:
            list_dict[z].pop(mz)
            continue
        
        list_dict[z][mz] = deque()
        limit=len(list_RT_range)
        j=0
        while j < limit:
            rt_st=round(list_RT_range.popleft(), 2)
            rt_end=round(list_RT_range.popleft() , 2)       
            list_RT_range.popleft() #remove the -1 sign
            # B
            if RT_index[rt_end]-RT_index[rt_st]>=2 and np.amax(ms1[RT_index[rt_st]-rt_search_index: RT_index[rt_end]-rt_search_index+1, int((mz-min_mz)/mz_unit)])>0: 
                list_dict[z][mz].append([rt_st, rt_end, -1])
            else:
                count=count+1 #just for debug to see how many traces were false detections like that
            j=j+3
    
    
    merge_isotopes=dict() #based on id 
    list_keys=np.sort(list(list_dict[z].keys()))
    
    if len(list_keys)==1:
        list_dict[z].pop(round(list_keys[0], 2))
    list_keys=np.sort(list(list_dict[z].keys())) 
    max_dict=len(list_keys)-1
    i=0
    j=0
    k=0
    for i in range (0, max_dict):
        mz_pred=round(list_keys[i], mz_resolution)
        mz=round(list_keys[i+1], mz_resolution)
        if round(mz_pred+mz_unit, mz_resolution)==mz:
            mz_pred_RT_list=list(list_dict[z][mz_pred])
            list_dict[z][mz_pred]=mz_pred_RT_list #it has made list from dict
            mz_RT_list=list(list_dict[z][mz])
            list_dict[z][mz]=mz_RT_list #it has made list from dict
            k=0
            for j in range (0, len(mz_pred_RT_list)):
                a=round(mz_pred_RT_list[j][0], 2)
                b=round(mz_pred_RT_list[j][1], 2)
                id=mz_pred_RT_list[j][2]
                
                mz_point1=int(round((mz_pred-min_mz)/mz_unit))
                rt_1_s=RT_index[a]-rt_search_index 
                rt_1_e=RT_index[b]-rt_search_index 
                y=np.copy(ms1[rt_1_s:rt_1_e+1, mz_point1])
                weight_pred_mz=np.sum(y)                    
                peak_RT_1=RT_list[(np.argmax(y)+rt_1_s+rt_search_index)] #peak_x#       
                #find the next overlapped 
                p=k
                max_overlapped_area=-1
                max_overlapped_index=-1
                while p < len(mz_RT_list):
                    c=round(mz_RT_list[p][0], 2)
                    d=round(mz_RT_list[p][1], 2)
                    #check overlapping: if (RectA.Left < RectB.Right && RectA.Right > RectB.Left..)
                    if c>=b:
                        break
                    elif a<d and b>c: #overlap                        
                        mz_point2=int(round((mz-min_mz)/mz_unit))
                        rt_2_s=RT_index[c]-rt_search_index 
                        rt_2_e=RT_index[d]-rt_search_index 
                        y=np.copy(ms1[rt_2_s:rt_2_e+1, mz_point2])
                        peak_RT_2=RT_list[(np.argmax(y)+rt_2_s+rt_search_index)]
                        # C
                        if  abs(RT_index[peak_RT_1]-RT_index[peak_RT_2])<=2: 
                            overlapped_area=min(b, d)-max(a, c)
                            if overlapped_area>max_overlapped_area:
                                max_overlapped_area=overlapped_area
                                max_overlapped_index=p
                    p=p+1
                        
                if max_overlapped_index==-1: #no match 
                    if id==-1:
                        new_id=len(merge_isotopes)
                        mz_weight=[weight_pred_mz]
                        peak_RT_list=[peak_RT_1]
                        merge_isotopes[new_id]=[mz_weight, a, b, -1, mz_weight, [mz_pred], peak_RT_list]  
                        list_dict[z][mz_pred][j][2]=[]
                        list_dict[z][mz_pred][j][2].append(new_id)                      
                    k=p
                    continue
                # else 
                c=round(mz_RT_list[max_overlapped_index][0], 2)
                d=round(mz_RT_list[max_overlapped_index][1], 2)                    
                mz_point2=int(round((mz-min_mz)/mz_unit, mz_resolution))                            
                rt_2_s=RT_index[c]-rt_search_index
                rt_2_e=RT_index[d]-rt_search_index 
                y=np.copy(ms1[rt_2_s:rt_2_e+1, mz_point2])
                
                peak_RT_2=RT_list[(np.argmax(y)+rt_2_s+rt_search_index)]  
                weight_mz=np.sum(y)
                intensity_2=weight_mz
                ################################
                                                
                if id==-1:
                    intensity_1=weight_pred_mz

                    #########################
                    if intensity_1>intensity_2:
                        grp_rt_st=a
                        grp_rt_end=b
                        auc=intensity_1
                    else:
                        grp_rt_st=c
                        grp_rt_end=d
                        auc=intensity_2
                        
                    new_id=len(merge_isotopes)
                    mz_weight=[weight_pred_mz, weight_mz]
                    peak_RT_list=[peak_RT_1, peak_RT_2]
                    merge_isotopes[new_id]=[mz_weight, grp_rt_st, grp_rt_end, auc, intensity_1+intensity_2, [mz_pred, mz], peak_RT_list]
                    if list_dict[z][mz][max_overlapped_index][2]==-1:
                        list_dict[z][mz][max_overlapped_index][2]=[]
                        
                    list_dict[z][mz][max_overlapped_index][2].append(new_id)
                    list_dict[z][mz_pred][j][2]=[]
                    list_dict[z][mz_pred][j][2].append(new_id)
                else: #this might need to run a loop over ids. do this for all ids
                    for pred_id in id:
                        get_current_intensity=merge_isotopes[pred_id][3]
                        if get_current_intensity<=intensity_2:
                            merge_isotopes[pred_id][1]=c
                            merge_isotopes[pred_id][2]=d
                            merge_isotopes[pred_id][3]=intensity_2
                            
                        # add new intensity and weight to the existing one
                        merge_isotopes[pred_id][4]=merge_isotopes[pred_id][4]+intensity_2
                        merge_isotopes[pred_id][5].append(mz)
                        merge_isotopes[pred_id][0].append(weight_mz)
                        merge_isotopes[pred_id][6].append(peak_RT_2)
                        if list_dict[z][mz][max_overlapped_index][2]==-1:
                            list_dict[z][mz][max_overlapped_index][2]=[]
                        
                        list_dict[z][mz][max_overlapped_index][2].append(pred_id)
                 
                if max_overlapped_index==-1:
                    k=p
                else:
                    k=max_overlapped_index
        elif i==0 or round(list_keys[i-1]+mz_unit, 2)!=mz_pred:
            list_dict[z].pop(mz_pred)


    if len(list_keys)!=0:
        i=i+1                        
        mz=round(list_keys[i], 2)
        mz_RT_list=list(list_dict[z][mz])
        list_dict[z][mz]=mz_RT_list
        for j in range (0, len(mz_RT_list)):
            if mz_RT_list[j][2]==-1:
                a=round(mz_RT_list[j][0], 2)
                b=round(mz_RT_list[j][1], 2)                
                mz_point1=int(round((mz-min_mz)/mz_unit, mz_resolution))
                rt_1_s=RT_index[a]-rt_search_index 
                rt_1_e=RT_index[b]-rt_search_index 
                y=np.copy(ms1[rt_1_s:rt_1_e+1, mz_point1])
                weight_mz=np.sum(y)                
                peak_RT_1=RT_list[(np.argmax(y)+rt_1_s+rt_search_index)] 
                
                new_id=len(merge_isotopes)
                mz_weight=[weight_mz]
                peak_RT_list=[peak_RT_1]
                merge_isotopes[new_id]=[mz_weight, a, b, -1, mz_weight, [mz], peak_RT_list]  
                list_dict[z][mz][j][2]=[]
                list_dict[z][mz][j][2].append(new_id)                                      
#                list_dict[z][mz][j]=[0, 0, -1]

    print('merge isotopes done')

    isotope_table=defaultdict(list)
    for i in range (0, len(merge_isotopes)):
        mz_weight_list=merge_isotopes[i][0]
        max_weight=-1
        mz_index=-1
        for j in range(0, len(mz_weight_list)):
            if mz_weight_list[j]>=max_weight:
                max_weight=mz_weight_list[j]
                mz_index=j
        isotope_table[round(merge_isotopes[i][5][mz_index], mz_resolution)].append([merge_isotopes[i][6][mz_index], merge_isotopes[i][1],merge_isotopes[i][2],merge_isotopes[i][4]])

    isotope_mz_list=sorted(isotope_table.keys())

    isotope_table_temp=defaultdict(list)
    for i in isotope_mz_list:
        isotope_table[i]=sorted(isotope_table[i])
        j=0
        while (j<len(isotope_table[i])):
            isotope_table_temp[i].append(isotope_table[i][j])
            if j+1>=len(isotope_table[i]):
                break
            for k in range (j+1,  len(isotope_table[i])):
                if (isotope_table[i][j][0]!=isotope_table[i][k][0]):
                    break
            j=k
            
            
    isotope_table=copy.deepcopy(isotope_table_temp)
    isotope_table_temp=0

    print('form cluster of isotopes to feed to the IsoGrouping module')
    DEBUG=0
    mz_list=sorted(isotope_table.keys())
    tolerance_RT=2 #D
    for mz in mz_list:
        iso_list_mz=isotope_table[mz]
        for i in range (0, len(iso_list_mz)):
            current_iso=iso_list_mz[i]
            current_mz=mz
            if current_iso[0]==-1:
                continue
            current_peak=current_iso[0]
            found1=0
            id=len(isotope_cluster)
            next_mz_exact=round(current_mz+isotope_gap[z], mz_resolution)
            next_mz_range=[]
            next_mz_range.append(next_mz_exact)
            mz_tolerance_10ppm=round((next_mz_exact*10.0)/10**6, mz_resolution)
            mz_tolerance=int(round(mz_tolerance_10ppm/mz_unit, mz_resolution))
            for tolerance_mz in range (1, mz_tolerance+1):
                next_mz_range.append(round(next_mz_exact-mz_unit*tolerance_mz, mz_resolution))
                next_mz_range.append(round(next_mz_exact+mz_unit*tolerance_mz, mz_resolution))
            # next_mz might be a range
            k=0
            while(k<len(next_mz_range)):
                next_mz= next_mz_range[k]            
                if next_mz in isotope_table:
                    found2=0
                    iso_list_next_mz=isotope_table[next_mz]
                    
                    for j in range (0, len(iso_list_next_mz)):
                        next_iso=iso_list_next_mz[j]
                        if next_iso[0]==-1:
                            continue
                        if RT_index[next_iso[0]]>RT_index[current_peak]+tolerance_RT:
                            break
                        if RT_index[current_peak]-tolerance_RT<=RT_index[next_iso[0]] and RT_index[next_iso[0]]<=RT_index[current_peak]+tolerance_RT:
                           # within tolerance. Check RT range
                            a=current_iso[1]
                            b=current_iso[2]
                            c=next_iso[1]
                            d=next_iso[2]
                            if a<=d and b>=c: #overlapped
                                found2=1
                                break
                            
                    if found2==1:
                        found1=1
                        isotope_table[next_mz][j]=[-1] #remove it
                        # add pred_iso to cluster
                        isotope_cluster[id].append([current_mz, current_iso])
                        current_iso=next_iso
                        current_peak=current_iso[0]
                        current_mz=next_mz
                        ############
                        next_mz_exact=round(current_mz+isotope_gap[z], mz_resolution)
                        next_mz_range=[]
                        next_mz_range.append(next_mz_exact)
                        mz_tolerance_10ppm=round((next_mz_exact*10.0)/10**6, mz_resolution)
                        mz_tolerance=int(round(mz_tolerance_10ppm/mz_unit, mz_resolution))                        
                        for tolerance_mz in range (1, mz_tolerance+1):
                            next_mz_range.append(round(next_mz_exact-mz_unit*tolerance_mz, mz_resolution))
                            next_mz_range.append(round(next_mz_exact+mz_unit*tolerance_mz, mz_resolution))
                        ############
                        k=0
                    else:    
                        k=k+1
                else:
                    k=k+1
            if found1==1:
                # add pred_iso to cluster
                isotope_cluster[id].append([current_mz, current_iso])
                isotope_cluster[id].append([z]) # charge
            else: #else: insert them in to the single iso table
#                id=len(isotope_cluster)
                isotope_cluster[id].append([current_mz, current_iso])    
                isotope_cluster[id].append([z])
                
            isotope_table[mz][i]=[-1] #remove it
#        if DEBUG==1:
#            break


#########################################
print(len(isotope_cluster.keys()))
total_cluster=len(isotope_cluster.keys())
temp_isotope_cluster=copy.deepcopy(isotope_cluster)
isotope_cluster=defaultdict(list)
total_clusters=len(temp_isotope_cluster.keys())

for i in range (0, total_clusters):
    ftr=copy.deepcopy(temp_isotope_cluster[i])
    isotope_cluster[round(ftr[0][0], mz_resolution)].append(ftr) # starting m/z of the 1st isotope

temp_isotope_cluster=0

keys_list=sorted(isotope_cluster.keys())
max_num_iso=0
for mz in keys_list:
    ftr_list=isotope_cluster[mz]
    for i in range (0,  len(ftr_list)):
        ftr=ftr_list[i]
        if (len(ftr)-1)>max_num_iso:
            max_num_iso=(len(ftr)-1)        
    
print("max number of isotopes in the cluster is %d "%max_num_iso) 

f=open(datapath+file_name+'_clusters', 'wb') 
pickle.dump([isotope_cluster, max_num_iso, total_cluster], f, protocol=2)
f.close()
print('cluster write done')

