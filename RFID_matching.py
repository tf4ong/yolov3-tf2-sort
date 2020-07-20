import pandas as pd
import numpy as np
import math
import itertools
from scipy.optimize import linear_sum_assignment
import cv2
import os
from tqdm import tqdm
import itertools
pd.options.mode.chained_assignment = None
# RFID reader coodinates
RFID_coords={0:[269, 186, 345, 260],1:[36, 254, 96, 323],2:[39, 65, 96, 130],3:[166, 34, 250, 101],
             4:[416, 51, 490, 128],5:[397, 189, 511, 321]}
#max distance before centroid ID association is not conducted
max_distance=150
#max frames to look back into 
max_frames=30
# the overlap between bbox and entrance point to do nothing
ent_thres=50
#rework split entry
overlap_thres=0.25
#frames to look backwards and forwards for entry rfid (3)
entr_frames=15
# distance to the RFID reader for a matching to occur 
RFID_dist=50
frame_int=876
"""
Loads to RFID csv file in to pandas dataframe
returns:
    1.the dataframe of the original RFID csv file
"""
def load_RFID(path):
    df_RFID=pd.read_csv(path,index_col=False,dtype=str)
    df_RFID.Frame=df_RFID.Frame.astype(int)
    df_RFID=df_RFID.set_index('Frame')
    return df_RFID
'''
calculates the area of the bbself.df_track_temp
Intake bb:x1,y1,x2,y2
'''            
def bbox_area(bbox):
    w=bbox[0]+bbox[2]
    h=bbox[1]+bbox[3]
    area=w*h
    return area
'''
returns the centroid of the bbox
'''
def bbox_to_centroid(bbox):
    cX=int((bbox[0]+bbox[2])/2)
    cY=int((bbox[1]+bbox[3])/2)
    return [cX,cY]
'''    index_list=df[df.RFID_tracks.map(len)==3].index
calculates the centronoid distances between bbs
intake centronoid
'''
def Distance(centroid1,centroid2):  
     dist = math.sqrt((centroid2[0] - centroid1[0])**2 + (centroid2[1] - centroid1[1])**2)  
     return dist
'''
Gets the centroid distance between RFID reader and bbox
 Input bbox
 Out distancebbox_to_centroid(bbox)
'''
def distance_box_RFID(RFID,bbox):
    bbox_1_centroid=bbox_to_centroid(RFID_coords[RFID])
    bbox_2_centroid=bbox_to_centroid(bbox)
    return Distance(bbox_1_centroid,bbox_2_centroid)
'''
Gets the distance of the bb to RFIDs
'''
def distance_to_entrance(bbox2):
    bbox_1_centroid=bbox_to_centroid(RFID_coords[5])
    bbox_2_centroid=bbox_to_centroid(bbox2)
    return Distance(bbox_1_centroid,bbox_2_centroid)
'''
Removes duplicate loggins in the dataframe
'''
def duplicate_remove(list_dup):
    l1=[]
    [l1.append(i) for i in list_dup if i not in l1]
    return l1
'''
find the intersection over union between bb
''' 
def iou(bb_test,bb_gt):
    xx1 = np.maximum(bb_test[0], bb_gt[0])
    yy1 = np.maximum(bb_test[1], bb_gt[1])
    xx2 = np.minimum(bb_test[2], bb_gt[2])
    yy2 = np.minimum(bb_test[3], bb_gt[3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    o = wh / ((bb_test[2]-bb_test[0])*(bb_test[3]-bb_test[1])
      + (bb_gt[2]-bb_gt[0])*(bb_gt[3]-bb_gt[1]) - wh)
    return(o)
'''
splits the sort_tracks into ones in cage vs near entrance based on ent_thres
'''
def track_splitter(sort_tracks):
    sort_entrance_distance=[distance_to_entrance(i) for i in sort_tracks]
    sort_entrance_tracks=[i for i,v in enumerate(sort_entrance_distance) if v <=ent_thres]
    sort_entrance_tracks=[v for i,v in enumerate(sort_tracks) if i in sort_entrance_tracks]
    sort_cage_tracks=[i for i,v in enumerate(sort_entrance_distance) if v>ent_thres]
    sort_cage_tracks=[v for i, v in enumerate(sort_tracks) if i in sort_cage_tracks]
    return sort_entrance_tracks, sort_cage_tracks
'''
Reads the tracks processed into a dataframe
'''
def df_tracks_read(path):
    columns=['frame_count','Sort_dets','Sort_track_ids','Sort_tracks','sort_cage_dets',
                                 'sort_cage_ids','sort_cage_tracks','sort_entrance_dets','sort_entrance_ids',
                                 'sort_entrance_tracks','iou','track_id','tracks','RFID_readings']
    dics={i: eval for i in columns}
    df = pd.read_csv(path, converters=dics)
    df=df.set_index('Unnamed: 0')
    return df
'''
reconnect sort ids in the cage
works based on past frames
intake: 1.the current frame, 2. the sort tracks, 3 the dataframe holding the data
'''
def reconnect_tracks_main(frame_count,sort_tracks,df_trackout):
    track_dict={}
    if sort_tracks != []:
        sort_entrance_tracks, sort_cage_tracks= track_splitter(sort_tracks)
    else:
        return {}
    sort_entrance_tracks=sorted(sort_entrance_tracks,key=lambda x:x[4])
    sort_cage_tracks=sorted(sort_cage_tracks,key=lambda x:x[4])
    if frame_count-1==0:
        return {}
    else:
        previous_cage_sort=df_trackout.loc[frame_count-1]['sort_cage_tracks']
        previous_cage_sort_index=df_trackout.loc[frame_count-1]['sort_cage_ids']
        current_cage_index=[i[4] for i in sort_cage_tracks]
    if current_cage_index == previous_cage_sort_index:
        return {}
    if  len(df_trackout)<max_frames+1:
        max_detections = max(df_trackout['sort_cage_dets'])
    else: 
        max_detections=max(df_trackout.loc[frame_count-max_frames:frame_count]['sort_cage_dets'])
    if len(current_cage_index)>0 and len(current_cage_index) <= max_detections:
        if len(current_cage_index)<len(previous_cage_sort_index):
            if frame_count ==frame_int:
                print(frame_count)
                print('loop 2')
                print(max_detections)
            track_dict=reconnect_tracks_1(frame_count,sort_cage_tracks,current_cage_index,previous_cage_sort,
                                          previous_cage_sort_index,df_trackout)
        else:
            track_dict=reconnect_tracks_2(frame_count,sort_cage_tracks,current_cage_index,
                                          previous_cage_sort,previous_cage_sort_index,df_trackout)
            if frame_count ==frame_int:
                print(frame_count)
                print('loop 3')
                print(track_dict)
                print(max_detections)
    elif len(current_cage_index)>0 and len(current_cage_index) > max_detections:
        if frame_count ==frame_int:
            print(frame_count)
            print('loop 4')
            print(max_detections)
        track_dict=reconnect_tracks_3(frame_count,sort_cage_tracks,current_cage_index,
                                      previous_cage_sort,previous_cage_sort_index,df_trackout)
    return track_dict

def iou_tracks(sort_track_nums,sort_tracks):
    iou_index=[]
    iou_area=[]
    for combinations in itertools.combinations(sort_tracks,2):
        iou_index.append((combinations[0][4],combinations[1][4]))
        iou_area.append(iou(combinations[0],combinations[1]))
    iou_dictionary= {i:v for i,v in zip(iou_index,iou_area)}
    return iou_dictionary
'''
function in the reconnect_tracks_main
'''
def reconnect_tracks_1(frame_count,sort_cage_tracks,current_cage_index,previous_cage_sort,previous_cage_sort_index,df_trackout):
    if len(previous_cage_sort)==0:
        if len(df_trackout)<max_frames+1:
            start_frame=1
        else: 
            start_frame=frame_count-max_frames
        try:
            index_n= df_trackout.loc[start_frame:frame_count-1].query('sort_cage_dets!=0').index[-1]
            previous_cage_sort_index=df_trackout.loc[index_n]['sort_cage_track_ids']
            previous_cage_sort_index.sort()
            previous_cage_sort= df_trackout.loc[index_n]['Sort_cage_tracks']
            previous_cage_sort=sorted(previous_cage_sort,key=lambda x:x[4])
        except Exception: 
            return {}
    track_dict={}
    new_index=[i for i in current_cage_index if i not in previous_cage_sort_index]
    tracked_index=[i for i in current_cage_index if i in previous_cage_sort_index]
    lost_tracks=[i for i in previous_cage_sort_index if i not in tracked_index]
    old_centroids=[bbox_to_centroid(z) for z in previous_cage_sort if z[4] in lost_tracks]

    if old_centroids ==[]:
        return {}
    if len(new_index) ==0:
        return {}
    else:
        if len(new_index)==1 or len(lost_tracks)==0:
            for i in new_index:
                new_centroid=bbox_to_centroid([z for z in sort_cage_tracks if z[4]==i][0])
                distances=[Distance(new_centroid,z) for z in old_centroids]
                if min(distances) <max_distance:
                    track_dict[lost_tracks[distances.index(min(distances))]]=i
                else:
                    pass
        elif len(new_index)>1:
            distance_matrix=[]
            for i in new_index:
                new_centroid=bbox_to_centroid([z for z in sort_cage_tracks if z[4]==i][0])
                distances=[Distance(new_centroid,z) for z in old_centroids]
                distance_matrix.append(distances)
            distance_matrix=np.array(distance_matrix)
            _,col = linear_sum_assignment(distance_matrix)
            for i,z in zip(new_index,col):
                if distance_matrix[new_index.index(i)][z]<max_distance:
                    track_dict[lost_tracks[z]]=i
                else:
                    pass
    return track_dict
""""
function in the reconnect_tracks_main
"""
def reconnect_tracks_2(frame_count,sort_cage_tracks,current_cage_index,previous_cage_sort,previous_cage_sort_index,df_trackout):
    '''
    Process to assign the tracks if the current track was not
    in the previous frame and the number of current tracks is equal to or larger than
    the previous frame
    '''
    #number of detections in the current frame
    N_detections=len(current_cage_index)
    # finds the previous frame which there are the same number of tracks, does nothing if same n tracks not found in
    if len(df_trackout)<max_frames+1:
        start_frame=1
    else:
        start_frame=frame_count-max_frames
    try:
        index_n= df_trackout.loc[start_frame:frame_count-1].query(f'sort_cage_dets=={str(N_detections)}').index[-1]
    except Exception:
        return {}
    if len(previous_cage_sort_index)==0:
        try:
            index_n2= df_trackout.loc[start_frame:frame_count-1].query('sort_cage_dets!=0').index[-1]
            previous_cage_sort_index=df_trackout.loc[index_n2]['sort_cage_ids']
            previous_cage_sort_index.sort()
            previous_cage_sort=df_trackout.loc[index_n2]['sort_cage_tracks']
            previous_cage_sort=sorted(previous_cage_sort,key=lambda x:x[4])
        except Exception:
            return {}
    previous_track_num_matched=df_trackout.loc[index_n]['sort_cage_ids']
    previous_track_num_matched.sort()
    previous_track_num_matched_SORT=df_trackout.loc[index_n]['sort_cage_tracks']
    previous_track_num_matched_SORT=sorted(previous_track_num_matched_SORT,key=lambda x:x[4])
    new_index=[i for i in current_cage_index if i not in previous_cage_sort_index and i not in previous_track_num_matched]
    tracked_index=[i for i in current_cage_index if i in previous_cage_sort_index or i in previous_track_num_matched]
    lost_tracks=[i for i in list(set(previous_cage_sort_index +previous_track_num_matched)) if i not in tracked_index]
    previous_SORT_lost=[z for z in previous_cage_sort if z[4] in lost_tracks]
    previous_track_num_matched_SORT_lost=[z for z in previous_track_num_matched_SORT if z[4] in lost_tracks and z[4] not in previous_cage_sort_index]
    combined_lost_sort=previous_SORT_lost+previous_track_num_matched_SORT_lost
    combined_lost_sort=sorted(combined_lost_sort,key=lambda x:x[4])
    lost_tracks.sort()
    old_centroids=[bbox_to_centroid(z) for z in combined_lost_sort if z[4] in lost_tracks]
    if frame_count ==frame_int:
        print(frame_count)
        print('loop reconnect 2')
        print(previous_track_num_matched)
        print(previous_cage_sort_index)
        print(current_cage_index)
        print(new_index)
        print(tracked_index)
        print(lost_tracks)
    if old_centroids ==[]:
        return {}
    track_dict={}
    if len(new_index) ==0 or len(lost_tracks)==0:
        return {}
    elif len(new_index)==1:
    #else:
        for i in new_index:
            new_centroid=bbox_to_centroid([z for z in sort_cage_tracks if z[4]==i][0])
            distances=[Distance(new_centroid,z) for z in old_centroids]
            if min(distances) <max_distance:
                track_dict[lost_tracks[distances.index(min(distances))]]=i
    elif len(new_index)>1 and len(lost_tracks)>1:
        distance_matrix=[]
        for i in new_index:
            new_centroid=bbox_to_centroid([z for z in sort_cage_tracks if z[4]==i][0])
            distances=[Distance(new_centroid,z) for z in old_centroids]
            distance_matrix.append(distances)
        distance_matrix=np.array(distance_matrix)
        rnd,col = linear_sum_assignment(distance_matrix)
        try:
            for i,z in zip(new_index,col):
                if distance_matrix[new_index.index(i)][z]<max_distance:
                    track_dict[lost_tracks[z]]=i
                else:
                    pass
        except Exception as e:
            print(e)
            print(combined_lost_sort)
            print(lost_tracks)
            print(new_centroid)
            print(old_centroids)
    return track_dict


""""
function 3 tempin the reconnect_tracks_main
"""
def reconnect_tracks_3(frame_count,sort_cage_tracks,current_cage_index,previous_cage_sort,previous_cage_sort_index,df_trackout):
    '''
    Process to assign the tracks if the current track was not
    in the previous frame and the number of current tracks is equal to or larger than
    the previous frame
    '''
    #low numbers to look back to prevent quick changes
    entrance_id_list=[i for i in df_trackout.loc[frame_count-4:frame_count].sort_entrance_ids]
    try:
        entrance_id_list=list(itertools.chain(*entrance_id_list))
    except Exception:
        print(entrance_id_list)
        print(frame_count)
        return {}
    current_cage_index=[i for i in current_cage_index if i not in entrance_id_list]
    sort_cage_tracks=[v for v in sort_cage_tracks if v[4] not in entrance_id_list]
    #number of detections in the current frame
    N_detections=len(current_cage_index)
    # finds the previous frame which there are the same number of tracks, does nothing if same n tracks not found in
    if len(df_trackout)<max_frames+1:
        start_frame=1
    else:
        start_frame=frame_count-max_frames
    if frame_count ==frame_int:
        print(frame_count)
        print('loop 5')
    try:
        indexes= df_trackout.loc[start_frame:frame_count-1].query(f'sort_cage_dets=={str(N_detections)}').index
    except Exception:
        return {}
    entrance_ids=list(set(entrance_id_list))
    index_ns=[]
    for i in indexes: 
        id_check= df_trackout.loc[i].sort_cage_ids
        check=any(i for i in id_check if i in entrance_ids)
        if check: 
            pass
        else: 
            index_ns.append(i)
    if frame_count ==frame_int:
        print(frame_count)
        print('loop 6')
        print(indexes)
        print(entrance_ids)
    if len(index_ns) ==0:
        return {}
    index_n=index_ns[-1]
    previous_track_num_matched=df_trackout.loc[index_n]['sort_cage_ids']
    previous_track_num_matched.sort()
    previous_track_num_matched_SORT=df_trackout.loc[index_n]['sort_cage_tracks']
    previous_track_num_matched_SORT=sorted(previous_track_num_matched_SORT,key=lambda x:x[4])
    new_index=[i for i in current_cage_index if i not in previous_cage_sort_index and i not in previous_track_num_matched]
    tracked_index=[i for i in current_cage_index if i in previous_cage_sort_index or i in previous_track_num_matched]
    lost_tracks=[i for i in list(set(previous_cage_sort_index +previous_track_num_matched)) if i not in tracked_index]
    previous_SORT_lost=[z for z in previous_cage_sort if z[4] in lost_tracks]
    previous_track_num_matched_SORT_lost=[z for z in previous_track_num_matched_SORT if z[4] in lost_tracks and z[4] not in previous_cage_sort_index]
    combined_lost_sort=previous_SORT_lost+previous_track_num_matched_SORT_lost
    combined_lost_sort=sorted(combined_lost_sort,key=lambda x:x[4])
    lost_tracks.sort()
    old_centroids=[bbox_to_centroid(z) for z in combined_lost_sort if z[4] in lost_tracks]
    if frame_count ==frame_int:
        print('loop reconnect 3')
        print(previous_track_num_matched)
        print(previous_cage_sort_index)
        print(current_cage_index)
        print(new_index)
        print(tracked_index)
        print(lost_tracks)
    if old_centroids ==[]:
        return {}
    track_dict={}
    if len(new_index) ==0 or len(lost_tracks)==0:
        return {}
    elif len(new_index)==1:
    #else:
        for i in new_index:
            new_centroid=bbox_to_centroid([z for z in sort_cage_tracks if z[4]==i][0])
            distances=[Distance(new_centroid,z) for z in old_centroids]
            if min(distances) <max_distance:
                track_dict[lost_tracks[distances.index(min(distances))]]=i
    elif len(new_index)>1 and len(lost_tracks)>1:
        distance_matrix=[]
        for i in new_index:
            new_centroid=bbox_to_centroid([z for z in sort_cage_tracks if z[4]==i][0])
            distances=[Distance(new_centroid,z) for z in old_centroids]
            distance_matrix.append(distances)
        distance_matrix=np.array(distance_matrix)
        rnd,col = linear_sum_assignment(distance_matrix)
        try:
            for i,z in zip(new_index,col):
                if distance_matrix[new_index.index(i)][z]<max_distance:
                    track_dict[lost_tracks[z]]=i
                else:
                    pass
        except Exception as e:
            print(e)
            print(combined_lost_sort)
            print(lost_tracks)
            print(new_centroid)
            print(old_centroids)
    if frame_count ==frame_int:
        print(frame_count)
        print('loop 2')
        print(track_dict)
    return track_dict

"""
Function to implement changes processed by reconnect main  
"""
def connect_tracts(track_dict,sort_index,sort_index_entrance,sort_tracks):
    sort_index.sort()
    sort_tracks=sorted(sort_tracks,key=lambda x:x[4])
    if track_dict =={} or track_dict == None:
        return sort_index,sort_tracks
    else:
         for k,v in track_dict.items():
             if k>v: 
                 pass
             elif k in sort_index or k in sort_index_entrance:
                 pass
             else:
                 sort_tracks[sort_index.index(v)][4]=k
                 sort_index[sort_index.index(v)]=k
         sort_index.sort()
         sort_tracks=sorted(sort_tracks,key=lambda x:x[4])
    return sort_index,sort_tracks


def recconnect_sort_tracts(frame_count,sort_tracks,df_RFID_cage,df_track_temp):
    RFID_readings=df_RFID_cage.loc[frame_count].tolist()[2:] # change to 2 for testing samples
    RFID_readings={i:v for i,v in enumerate(RFID_readings) if v != '0'}
    if sort_tracks !=[]:
        sort_tracks_ids=[i[4] for i in sort_tracks]
        sort_entrance_tracks, sort_cage_tracks= track_splitter(sort_tracks)
        sort_cage_ids=[i[4] for i in sort_cage_tracks]
        sort_entrance_ids=[i[4] for i in sort_entrance_tracks]
        track_reassignment_index=reconnect_tracks_main(frame_count,sort_tracks,df_track_temp)
        if frame_count ==frame_int:
            print(frame_count)
            print(sort_cage_ids)
            print(track_reassignment_index)
        sort_cage_ids, sort_cage_tracks=connect_tracts(track_reassignment_index,
                                                       sort_cage_ids,sort_entrance_ids,sort_cage_tracks)
        if frame_count ==frame_int:
            print('final results')
            print(frame_count)
            print(sort_cage_ids)
            print(sort_cage_tracks)
            print(sort_entrance_tracks)
        tracks_temp= sort_cage_tracks+sort_entrance_tracks
        track_id=[i[4] for i in tracks_temp]
        if len(sort_tracks) >1:
            iou_dic=iou_tracks(sort_tracks_ids,sort_tracks)
        else:
            iou_dic={}
        df_track_temp.loc[frame_count]=[frame_count,len(sort_tracks_ids),sort_tracks_ids, sort_tracks, len(sort_cage_ids),
                                        sort_cage_ids, sort_cage_tracks, len(sort_entrance_ids),sort_entrance_ids,
                                        sort_entrance_tracks,iou_dic,track_id,tracks_temp,RFID_readings]
    else:
        df_track_temp.loc[frame_count]=[frame_count,0,[],[],0,[],[],0,[],[],{},[],[],RFID_readings]
    return df_track_temp


def RFID_matching(df):
    df1=pd.DataFrame(columns=['frame','RFID','Track_matched'])
    df['RFID_tracks']=[list() for i in range(len(df.index))]
    df['RFID_matched']=[list() for i in range(len(df.index))]
    validation_frames=[i-1 for i in df.loc[df.RFID_readings !={}].index]
    '''
    index_delete=[]
    index_seq=[]
    for i,v in enumerate(validation_frames):
        if i<len(validation_frames)-1:
            if v+1 ==validation_frames[i+1]:
                index_seq.append(i)
    for i in index_seq:
        index1=validation_frames[i]
        index2=validation_frames[i+1]
        try:
            if df.loc[index1]['RFID_readings'] == df.loc[index2]['RFID_readings']:
                index_delete.append(i)
            else:
                pass
        except Exception: 
            print(i)
            print(df.loc[index1]['RFID_readings'])
    validation_frames=[v-1 for i,v in enumerate(validation_frames) if i not in index_delete]
    '''
    for z in validation_frames:
        for i,v in df.iloc[z]['RFID_readings'].items():
            if v != 'None':
                track_distances=[distance_box_RFID(i,k) for k in df.iloc[z]['tracks']]
                track_distances= {t:k for t,k in enumerate(track_distances) if k<RFID_dist}
                if len(track_distances) >0:
                    index=min(track_distances,key=lambda k:track_distances[k])
                    sort_id=df.iloc[z]['tracks'][index][4]
                    df1.loc[z]= [z,df.iloc[z]['RFID_readings'],sort_id]
                    ious=[v for i,v in df.iloc[z]['iou'].items() if sort_id in i and v >overlap_thres]
                else: 
                    ious=[1,1,1,1]
                if len(ious) ==0 and len(df.iloc[z]['tracks']) !=0:
                    a=[k for k in df.iloc[z]['tracks'] if k[4]==sort_id]
                    matched=[[sort_id,v]]
                    track=[[a[0][0],a[0][1],a[0][2],a[0][3],v]]
                    df.iloc[z]['RFID_tracks']+=track
                    df.iloc[z]['RFID_matched']+= matched
                    floating_index_f=1
                    floating_index_b=1
                    skip_counter_b=0
                    skip_counter_f=0
                    iou_foward=[i for i,v in df.iloc[z+floating_index_f]['iou'].items() if sort_id in i and v >overlap_thres]
                    iou_backward=[i for i,v in df.iloc[z-floating_index_b]['iou'].items() if sort_id in i and v >overlap_thres]
                else:
                    iou_foward=[1,1,1,1]
                    iou_backward=[1,1,1,1]
                while len(iou_backward)<1:
                    if sort_id in df.iloc[z-floating_index_b]['track_id']:
                        a=[k for k in df.iloc[z-floating_index_b]['tracks'] if k[4]==sort_id]
                        matched=[[sort_id,v]]
                        track=[[a[0][0],a[0][1],a[0][2],a[0][3],v]]
                        if track not in df.iloc[z-floating_index_b]['RFID_tracks']:
                            df.iloc[z-floating_index_b]['RFID_tracks']+=track
                        df.iloc[z-floating_index_b]['RFID_matched']+= matched
                        #df.iloc[z-floating_index_b]['RFID_tracks']+=a
                        floating_index_b+=1
                        iou_backward=[i for i,v in df.iloc[z-floating_index_b]['iou'].items() if sort_id in i and v >overlap_thres]
                    else:
                         skip_counter_b+=1
                         floating_index_b+=1
                    if z-floating_index_b ==-1 or skip_counter_b ==30:
                        iou_backward=[1,1,1,1]
                    else:
                        pass
                else:
                    #print(z-floating_index_b+1)
                    #print(iou_backward)
                    pass
                while len(iou_foward)<1:
                    if sort_id in df.iloc[z+floating_index_f]['track_id']:
                        a=[k for k in df.iloc[z+floating_index_f]['tracks'] if k[4]==sort_id]
                        track=[[a[0][0],a[0][1],a[0][2],a[0][3],v]]
                        matched=[[sort_id,v]]
                        if track not in  df.iloc[z+floating_index_f]['RFID_tracks']:
                            df.iloc[z+floating_index_f]['RFID_tracks']+=track
                        df.iloc[z+floating_index_f]['RFID_matched']+= matched
                        floating_index_f+=1
                        iou_foward=[i for i,v in df.iloc[z+floating_index_f]['iou'].items() if sort_id in i and v >overlap_thres]
                    else:
                         skip_counter_f+=1
                         floating_index_f+=1
                    if skip_counter_f ==30 or z+floating_index_f==len(df)-1:
                        iou_foward=[1,1,1,1,1]
                    else:
                        pass
                else:
                    pass
    df['RFID_tracks']= df['RFID_tracks'].map(lambda x: duplicate_remove(x))
    df['RFID_matched']= df['RFID_matched'].map(lambda x: duplicate_remove(x))
    return df,df1,validation_frames


# tries to match the last RFID tag (when the other 3 tags are matched)
# the process repeats until all frames with 3 tags matched (4 animals detected) are completely matched
def last_tag_match(df,tags):
    #gets index of RFID tracks where 3 tags have been matched
    index_list=df[df.RFID_tracks.map(len)==3].index
    index_list=[i for i in index_list if len(df.loc[i].track_id) ==4]
    print(len(index_list))
    print(index_list)
    #checks if there are any duplicated of the tags (same mouse being taged twice)
    #loop=0
    #while len(index_list) != 0:
    for i in index_list: 
        RFID_list=[i[4] for i in df.loc[i].RFID_tracks]
        if len(set(RFID_list)) !=3:
            print(i)
            print(df.loc[i].RFID_tracks)
            print('loop1')
            pass
        else:
            #checks if there are any unmatched track_id 
            tag_left=list(set(tags)-set(RFID_list))[0]
            matched=[z[0] for z in df.loc[i].RFID_matched]
            tracks=[z for z in df.loc[i].track_id]
            unmatched_id=list(set(tracks)-set(matched))
            if len(unmatched_id) == 0:
                print('loop2')
                pass
            elif len(unmatched_id) !=1:
                pass
            elif tag_left not in [i[4] for i in df.loc[i]['RFID_tracks']]:
                pass
            else:
                a=[k for k in df.loc[i]['tracks'] if k[4]==unmatched_id[0]]
                track=[[a[0][0],a[0][1],a[0][2],a[0][3],tag_left]]
                track_f=df.loc[i,'RFID_tracks']+track
                df.at[i,'RFID_tracks']=track_f
                RFID_matched=df.loc[i,'RFID_matched']+[[unmatched_id[0],tag_left]]
                df.at[i,'RFID_matched']=RFID_matched
                floating_index_f=1
                floating_index_b=1
                skip_counter_b=0
                skip_counter_f=0
                iou_foward=[z for z,v in df.loc[i+floating_index_f]['iou'].items() if unmatched_id[0] in z and v >overlap_thres]
                iou_backward=[z for z,v in df.loc[i-floating_index_b]['iou'].items() if unmatched_id[0] in z and v >overlap_thres]
            while len(iou_backward)<1 and tag_left not in [i[4] for i in df.loc[i-floating_index_b]['RFID_tracks']]:
                if unmatched_id[0] in df.loc[i-floating_index_b]['track_id']:
                    a=[k for k in df.loc[i-floating_index_b]['tracks'] if k[4]==unmatched_id[0]]
                    track=[[a[0][0],a[0][1],a[0][2],a[0][3],tag_left]]
                    track_f=df.loc[i-floating_index_b,'RFID_tracks']+track
                    df.at[i-floating_index_b,'RFID_tracks']=track_f
                    RFID_matched= df.loc[i-floating_index_b,'RFID_matched']+[[unmatched_id[0],tag_left]]
                    df.at[i-floating_index_b,'RFID_matched']=RFID_matched
                    floating_index_b+=1
                    iou_backward=[z for z,v in df.loc[i-floating_index_b]['iou'].items() if unmatched_id[0 in z and v >overlap_thres]]
                else: 
                    skip_counter_b+=1
                    floating_index_b+=1
                if i-floating_index_b ==-1 or skip_counter_b ==30:
                    iou_backward=[1,1,1,1]
            else:
                pass
            while len(iou_foward)<1 and tag_left not in [i[4] for i in df.loc[i-floating_index_b]['RFID_tracks']]:
                if unmatched_id[0] in df.loc[i+floating_index_f]['track_id']:
                    a=[k for k in df.loc[i+floating_index_f]['tracks'] if k[4]==unmatched_id[0]]
                    track=[[a[0][0],a[0][1],a[0][2],a[0][3],tag_left]]
                    track_f=df.loc[i+floating_index_f,'RFID_tracks']+track
                    df.at[i+floating_index_f,'RFID_tracks']=track_f
                    RFID_matched=df.loc[i+floating_index_f,'RFID_matched']+[[unmatched_id[0],tag_left]]
                    df.at[i+floating_index_f,'RFID_matched']=RFID_matched
                    floating_index_f+=1
                    iou_foward=[z for z,v in df.loc[i+floating_index_f]['iou'].items() if unmatched_id[0] in z and v >overlap_thres]
                else:
                     skip_counter_f+=1
                     floating_index_f+=1
                if skip_counter_f ==30 or i+floating_index_f==len(df)-1:
                    iou_foward=[1,1,1,1,1]
                else:
                    pass
            #else:
            #    pass
    df['RFID_tracks']= df['RFID_tracks'].map(lambda x: duplicate_remove(x))
    df['RFID_matched']= df['RFID_matched'].map(lambda x: duplicate_remove(x))
        #index_list=df[df.RFID_tracks.map(len)==3].index
        #index_list=[i for i in index_list if len(df.loc[i].track_id) ==4]
        #loop+=1
    '''
        print(f'Finsihed loop1 {str(loop)}')
        print(len(index_list))
        print(index_list)
    '''
   # else:
   #     pass
    return df
def get_lost_tracks(df,dfr):
    df['lost_tracks']=[list() for i in range(len(df))]
    for i in range(len(df)):
        if len(df.iloc[i]['tracks']) ==0:
            pass
        elif len(df.iloc[i]['RFID_tracks']) ==0 and len(df.iloc[i]['Sort_tracks']) !=0:
            df.iloc[i]['lost_tracks']+=df.iloc[i]['tracks']
        else:
            temp_to_comapre=[z[:4] for z in df.iloc[i]['RFID_tracks']]
            temp_sort= [z[:4] for z in df.iloc[i]['tracks']]
            index_lost_tracks=[z for z,v in enumerate(temp_sort) if v not in temp_to_comapre]
            lost_tracks=[v for z,v in enumerate(df.iloc[i]['tracks']) if z in index_lost_tracks]
            df.iloc[i]['lost_tracks']+=lost_tracks
    df['Timestamp']=dfr['Time']
    return df




"""
Class for mouse tracking
"""
class mouse_tracker:
    def __init__(self,tags,path,df_RFID_cage,writer,vid_length,batch=300):
        self.tags={i:'NA' for i in tags} #starts a dictionary for the tags about their where abouts
        self.df_RFID_cage = df_RFID_cage
        self.df_tracks_temp=pd.DataFrame(columns=['frame_count','Sort_dets','Sort_track_ids','Sort_tracks','sort_cage_dets',
                                     'sort_cage_ids','sort_cage_tracks','sort_entrance_dets','sort_entrance_ids',
                                     'sort_entrance_tracks','iou','track_id','tracks','RFID_readings'])
        self.writer= cv2.VideoWriter(writer[0], writer[1],writer[2], (800,800))
        self.colours=np.random.rand(300, 3) * 255
        self.batch=0
        self.path=path
        self.batch_size=batch
        self.vid_length=vid_length
        if not os.path.exists(f'{self.path}/tracks'):
            os.mkdir(f'{self.path}/tracks')
    def update(self, frame_count, sort_tracks):
        self.df_tracks_temp=recconnect_sort_tracts(frame_count,sort_tracks,self.df_RFID_cage,self.df_tracks_temp)
        if len(self.df_tracks_temp) == self.batch_size:
            self.batch +=1
            df_RFID_tracks_save=self.df_tracks_temp[:self.batch_size-100]
            df_RFID_tracks_save.to_csv(f'{self.path}/tracks/tracks_{str(self.batch)}.csv')
            self.df_tracks_temp=self.df_tracks_temp[self.batch_size-100:]
        elif self.vid_length ==frame_count:
            self.batch +=1
            self.df_tracks_temp.to_csv(f'{self.path}/tracks/tracks_{str(self.batch)}.csv')
        else:
            pass
    def evaulate(self):
        path_list=os.listdir(f'{self.path}/tracks')
        self.df_tracks_temp=pd.concat([df_tracks_read(f'{self.path}/tracks/'+i) for i in path_list])
        self.df_tracks_temp=self.df_tracks_temp.sort_index()
        self.df_tracks_temp,df_matches,validation_frames=RFID_matching(self.df_tracks_temp)
        self.validation_frames=validation_frames
        '''
        try:
            self.df_tracks_temp=last_tag_match(self.df_tracks_temp,self.tags)
        except KeyboardInterrupt:
            self.df_tracks_temp.to_csv('tesst1221.csv')
        '''
        self.df_tracks_temp=last_tag_match(self.df_tracks_temp,self.tags)
        self.df_tracks_temp=get_lost_tracks(self.df_tracks_temp,self.df_RFID_cage)
        return self.df_tracks_temp, df_matches
    def write_video(self,frame_count,img,Vid_type):
        if Vid_type == 'RFID':
            tracker=self.df_tracks_temp.loc[frame_count]['RFID_tracks']
        elif Vid_type == 'Sort':
            tracker=self.df_tracks_temp.loc[frame_count]['tracks']
        elif Vid_type =='Original':
            tracker= self.df_tracks_temp.loc[frame_count]['Sort_tracks']
        RFID_frame=[i for i in self.validation_frames if i>frame_count]
        if len(RFID_frame) >0:
            RFID_frame=min(RFID_frame)
        blankimg=250*np.ones(shape=[800,800,3],dtype=np.uint8)
        blankimg[200:600,144:656]=img
        for object in tracker:
            xmin, ymin, xmax, ymax, index = int(object[0]), int(object[1]), int(object[2]), int(object[3]), int(object[4])
            cx,cy=bbox_to_centroid([xmin,ymin,xmax,ymax])
            color = (int(self.colours[index % 300, 0]), int(self.colours[index % 300, 1]), int(self.colours[index % 300, 2]))
            #cv2.circle(blankimg, (cx+144, cy+180), 10, (0, 255, 0), -1) 
            cv2.rectangle(blankimg, (xmin+144, ymin+200), (xmax+144, ymax+200), color, 3)
            #cv2.putText(blankimg, str(index), (cx+164, cy+200), 0, 5e-3 * 200, color, 3)
            cv2.putText(blankimg, str(index), (xmin+144, ymin+180), 0, 5e-3 * 200, color, 3)
        cv2.putText(blankimg,f"Frame: {str(frame_count)}",(320,150),0, 5e-3 * 200,(0,0,255),5)
        spacer=0
        if RFID_frame != []:
            for i,v in self.df_tracks_temp.iloc[RFID_frame]['RFID_readings'].items():
                cv2.putText(blankimg,f'Frame {RFID_frame}: reader {i} , tag read {v}', (20,650+spacer),0,5e-3 * 200,(0,0,255),2)
                spacer+=100
        self.writer.write(blankimg)
        return blankimg
######save center
######get raw values
######trajectory 
#######before reassigning, find rfid tracker 
#######a dictionary :key tag, value(tuple): a list of sort_ids
