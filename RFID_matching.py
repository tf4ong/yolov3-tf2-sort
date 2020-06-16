import pandas as pd
#import modin.pandas as pd
import numpy as np
import math
import itertools
from scipy.optimize import linear_sum_assignment
import cv2
import os
from tqdm import tqdm
pd.options.mode.chained_assignment = None
# RFID reader coodinates
RFID_coords={0:[43, 87, 80, 125],1:[46, 273, 78, 313],2:[431, 248, 482, 298],3:[411, 209, 507, 311],4:[455, 86, 495, 124]}
#max distance before centroid ID association is not conducted
max_distance=250
#max frames to look back into 
max_frames=60
# the overlap between bbox and entrance point to do nothing
ent_thres=35
#rework split entry
overlap_thres=0.15
#frames to look backwards and forwards for entry rfid (3)
entr_frames=60
RFID_dist=50
"""
Loads to RFID csv file in to pandas dataframe
returns:
    1.the dataframe of the original RFID csv file
    2. df containing RFID readings in the cage
    3. df contatining RFID readings at the entry tunnel
"""
def load_RFID(path):
    df_RFID=pd.read_csv(path,index_col=False,dtype=str)
    df_RFID.Frame=df_RFID.Frame.astype(int)
    df_RFID=df_RFID.set_index('Frame')
    return df_RFID#,df_RFID_cage, df_RFID_entry
'''
calculates the centronoid distances between bb                        print(sort_id)
intake centronoid
'''
def Distance(centroid1,centroid2):  
     dist = math.sqrt((centroid2[0] - centroid1[0])**2 + (centroid2[1] - centroid1[1])**2)  
     return dist
'''
calculates the area of the bb
Intake bb:x1,y1,x2,y2
'''
def bbox_area(bbox):
    w=bbox[0]+bbox[2]
    h=bbox[1]+bbox[3]
    area=w*h
    return area
"""
                        else:
                            pass
                elif max(df.iloc[z:z+entr_frames]['Num_detections_entrance']) ==0 and max(df.iloc[z-entr_frames:z]['Num_detections_entrance']) ==1:
                    index_n=df.iloc[z-entr_frames:z].query('Num_detections_entrance ==1')[-1]
finds the centr
"""
def bbox_to_centroid(bbox):
    cX=int((bbox[0]+bbox[2])/2)
    cY=int((bbox[1]+bbox[3])/2)
    return [cX,cY]

def distance_to_entrance(bbox2):
    bbox_1_centroid=bbox_to_centroid(RFID_coords[3])
    bbox_2_centroid=bbox_to_centroid(bbox2)
    return Distance(bbox_1_centroid,bbox_2_centroid)

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

def iou_tracks(sort_track_nums,sort_tracks):
    iou_index=[]
    iou_area=[]
    for combinations in itertools.combinations(sort_tracks,2):
        iou_index.append((combinations[0][4],combinations[1][4]))
        iou_area.append(iou(combinations[0],combinations[1]))
    iou_dictionary= {i:v for i,v in zip(iou_index,iou_area)}
    return iou_dictionary

"""
Checks if the mutlipel bbs are found even thou there should only be one
"""
def spontaneous_BB_checker(frame_count,sort_index,sort_tracks,df_trackout):
    sort_index.sort()
    sort_tracks=sorted(sort_tracks,key=lambda x:x[4])
    if len(df_trackout) ==0:
        max_detections =9999
    elif len(df_trackout)>0 and len(df_trackout)<max_frames+1:
        max_detections=max(df_trackout['Num_detections'])
    else:
        max_detections=max(df_trackout.loc[frame_count-max_frames:frame_count]['Num_detections'])
    if len(sort_index) <= max_detections:
        return  sort_index,sort_tracks
    else:
        list_iou_entry=[iou(RFID_coords[3],i) for i in sort_tracks]
        list_iou_entry2=[list_iou_entry.index(i) for i in list_iou_entry if i>0.5]
        iou_list=[]
        iou_area=[]
        if len(list_iou_entry2) !=0:
            return sort_index,sort_tracks
        else:
            for combinations in itertools.combinations(sort_tracks,2):
                iou_list.append([combinations[0][4],combinations[1][4]])
                iou_area.append(iou(combinations[0],combinations[1]))
            list
            merge_tracks=iou_list[iou_area.index(max(iou_area))]
            temp_track={i[4]:bbox_area(i) for i in sort_tracks if i[4] in merge_tracks}
            #print(temp_track)
            track_delete=min(temp_track,key=temp_track.get)
            sort_tracks=[i for i in sort_tracks if i[4] != track_delete]
            sort_index=[i for i in sort_index if i!=track_delete]
        return sort_index, sort_tracks
'''
splits the sort_tracks into ones in cage vs near entrance
'''
def track_splitter(sort_index,sort_tracks):
    sort_entrance_distance=[distance_to_entrance(i) for i in sort_tracks]
    sort_entrance_index1=[i for i,v in enumerate(sort_entrance_distance) if v<ent_thres]
    sort_entrance_index2=[v for i,v in enumerate(sort_index) if i in sort_entrance_index1]
    sort_cage_index=[v for i, v in enumerate(sort_index) if i not in sort_entrance_index1]
    sort_cage_tracks=[v for i, v in enumerate(sort_tracks) if i not in sort_entrance_index1]
    sort_entrance_tracks=[v for i,v in enumerate(sort_tracks) if i in sort_entrance_index1]
    return sort_cage_index,sort_entrance_index2,sort_cage_tracks, sort_entrance_tracks

def distance_box_RFID(RFID,bbox2):
    bbox_1_centroid=bbox_to_centroid(RFID_coords[RFID])
    bbox_2_centroid=bbox_to_centroid(bbox2)
    return Distance(bbox_1_centroid,bbox_2_centroid)


def df_tracks_read(path):
    columns=['frame','Num_detections','Sort_nums','Sort_tracks',
             'iou_tracks','Num_detections_cage','Sort_nums_cage',
             'Sort_tracks_cage','reass','Num_detections_entrance',
             'Sort_nums_entrance','Sort_tracks_entrance','RFID_readings']
    dics={i: eval for i in columns}
    df = pd.read_csv(path, converters=dics)
    df=df.set_index('Unnamed: 0')
    return df





"""
main fucntion to reconnect track id generated from SORT 
works based on past frames 
intake: 1.the current frame coutn, 2. the sort tracks (bboxes) generated from SORT, 
3. the sort ID, 4. the dataframe holding the sort information 
"""
def reconnect_tracks_main(frame_count,sort_tracks,sort_index,df_trackout):
    '''
    takes in the current frame count, tr mentioned above are installed properly.
Traceback (most recent call last):
  File "detect_sort.py", line 9, in <module>
    from RFID_matching import *
  File "/home/tony/yolov3-tf2/RFID_matching.py", line 21
    return [cX,cY]def reconnect_tracks_main(frame_count,sort_tracks,sort_index,df_trackout):
ackers produced by SORT, 
    the SORT track indexes, and the dataframe for storing the detection information
    return a dictionary in which the SORT tracks should be reassigned to a previous SORT track
    Uses the function reconnect_tracks_1 for abd reconnect_tracks_2
    '''
    #only start the reaasociation process of the tracks 
    #where there are more than 10 frames which is approximately 1 second
    # don't do anything if there isn't more than the number of frames the function can look up
    track_dict={}
    #gets the track id
    #need to remove any tracks near entrance to prevent complications
    if sort_tracks != []:
        sort_entrance_iou=[distance_to_entrance(i) for i in sort_tracks]
        sort_delete_index=[i for i,v in enumerate(sort_entrance_iou) if v<ent_thres]
        sort_tracks=[v for i, v in enumerate(sort_tracks) if i not in sort_delete_index]
        sort_index=[v for i, v in enumerate(sort_index) if i not in sort_delete_index]
    else:
        pass
    sort_index.sort()
    sort_tracks=sorted(sort_tracks,key=lambda x:x[4])
    if frame_count-1==0:
        return {}
    else:
    #get the previous SORT tracks and indexes
        previous_SORT=df_trackout.loc[frame_count-1]['Sort_tracks']
        #print(previous_SORT)
        previous_SORT=sorted(previous_SORT,key=lambda x:x[4])
        previous_SORT_index=df_trackout.loc[frame_count-1]['Sort_nums']
        previous_SORT_index.sort()
    # if the current SORT index is equal to the previous SORT indexe then return empty dict
    # there are no tracks to reassociate
        if sort_index == previous_SORT_index:
            #print('same as before')
            return {}
        else:
    # only starts the process if the number of current SORT track index is larger than 0,does nothing if there is no tracks this frame
    # frame rate is approximately 10 fps
    # assumes if that there is something in the previous 25 frames that can be referenced
    # depends on how well SORT handles occulsions
            if len(df_trackout)<max_frames+1:
                max_detections=max(df_trackout['Num_detections'])
            else:
                max_detections=max(df_trackout.loc[frame_count-max_frames:frame_count]['Num_detections'])
            '''
            if frame_count == 402:
                print(frame_count)
                print(previous_SORT)
                print(max_detections)
            '''
            if len(sort_index)>0 and len(sort_index) <= max_detections:
                #no new mice have appeared
                if len(sort_index)<len(previous_SORT_index):
                    track_dict=reconnect_tracks_1(frame_count,sort_tracks,sort_index,previous_SORT_index,previous_SORT,df_trackout)
                else:
                    track_dict=reconnect_tracks_2(frame_count,sort_tracks,sort_index,previous_SORT_index,previous_SORT,df_trackout)
            else:
                #if frame_count ==402:
                #    print('loop 3')
                #print('increased n')
                return {}
        return track_dict
"""
function in recconnect_tracks_main
"""
def reconnect_tracks_1(frame_count,sort_tracks,sort_index,previous_SORT_index,previous_SORT,df_trackout):
    '''
    Process to assign the tracks if the current track was not
    in the previous frame and number of current tracks 
    are smaller than the previous frame 
    '''
    if len(previous_SORT_index)==0:
        # gets a new previous frame to reference because there was no tracks in the previous frame
        # returns {} if no detections were found in the previous 10 frames
        if len(df_trackout)<max_frames+1:
            start_frame=1
        else:
            start_frame=frame_count-max_frames
        try:
            index_n= df_trackout.loc[start_frame:frame_count-1].query('Num_detections!=0').index[-1]
            previous_SORT_index=df_trackout.loc[index_n]['Sort_nums']
            previous_SORT_index.sort()
            previous_SORT=df_trackout.loc[index_n]['Sort_tracks']
            previous_SORT=sorted(previous_SORT,key=lambda x:x[4])
        except Exception:
            return {}
    #new index: new track idea; tracked_index:tracks in the previous and current; lost tracks: tracks in the previous but not current frame
    # old centroids: centroids of lost tracks
    track_dict={}
    new_index=[i for i in sort_index if i not in previous_SORT_index]
    tracked_index=[i for i in sort_index if i in previous_SORT_index]
    lost_tracks=[i for i in previous_SORT_index if i not in tracked_index]
    #checks if the new index and lost index have been previously registered in tracks_history
    old_centroids=[bbox_to_centroid(z) for z in previous_SORT if z[4] in lost_tracks]
    track_dict={}
    #if no new tracks appear do nothing
    if len(new_index) ==0:
        #does nothing beacuse there was no new tracks
        return {}
    #if there is only one new track match it to the closes object in the old_centroid dictionary
    else:
        if len(new_index)==1 or len(lost_tracks)==0:
            for i in new_index:
                new_centroid=bbox_to_centroid([z for z in sort_tracks if z[4]==i][0])
                distances=[Distance(new_centroid,z) for z in old_centroids]
                if min(distances) <max_distance:
                    track_dict[lost_tracks[distances.index(min(distances))]]=i
                else:
                    pass
    #a little more complex if there is more than one new index
    #the new track maybe closest to mulitple previous tracks
    #used linear sum assignment to solve the issue
    # also discards the result if the distance to of the new track to old centroid assigned is larger than max distance 
    #( can't associate to a matrix too far), better to have it as a new track to verify
    # essentially this process can return nothing 
    # potential for identity swaps
        elif len(new_index)>1:
            distance_matrix=[]
            for i in new_index:
                new_centroid=bbox_to_centroid([z for z in sort_tracks if z[4]==i][0])
                distances=[Distance(new_centroid,z) for z in old_centroids]
                distance_matrix.append(distances)
            distance_matrix=np.array(distance_matrix)
            _,col = linear_sum_assignment(distance_matrix)
            for i,z in zip(new_index,col):
                if distance_matrix[new_index.index(i)][z]<max_distance:
                    track_dict[lost_tracks[z]]=i
                else:
                    pass
    #print(track_dict)
    return track_dict
""""
function in the reconnect_tracks_main
"""
def reconnect_tracks_2(frame_count,sort_tracks,sort_index,previous_SORT_index,previous_SORT,df_trackout):
    '''
    Process to assign the tracks if the current track was not
    in the previous frame and the number of current tracks is equal to or larger than
    the previous frame
    '''
    #number of detections in the current frame
    N_detections=len(sort_index)
    # finds the previous frame which there are the same number of tracks, does nothing if same n tracks not found in
    #the previous frames (max_frames) 
    if len(df_trackout)<max_frames+1:
        start_frame=1
    else:
        start_frame=frame_count-max_frames
    try: 
        index_n= df_trackout.loc[start_frame:frame_count-1].query(f'Num_detections=={str(N_detections)}').index[-1]
    except Exception:
        return {}
    # gets the another previous frame to reference if the previous frame grabbed has zero tracks
    if len(previous_SORT_index)==0:
        index_n2= df_trackout.loc[start_frame:frame_count-1].query('Num_detections!=0').index[-1]
        previous_SORT_index=df_trackout.loc[index_n2]['Sort_nums']
        previous_SORT_index.sort()
        previous_SORT=df_trackout.loc[index_n2]['Sort_tracks']
        previous_SORT=sorted(previous_SORT,key=lambda x:x[4])
    previous_track_num_matched=df_trackout.loc[index_n]['Sort_nums']
    previous_track_num_matched.sort()
    previous_track_num_matched_SORT=df_trackout.loc[index_n]['Sort_tracks']
    previous_track_num_matched_SORT=sorted(previous_track_num_matched_SORT,key=lambda x:x[4])
    new_index=[i for i in sort_index if i not in previous_SORT_index and i not in previous_track_num_matched]
    tracked_index=[i for i in sort_index if i in previous_SORT_index or i in previous_track_num_matched]
    lost_tracks=[i for i in list(set(previous_SORT_index +previous_track_num_matched)) if i not in tracked_index]
    previous_SORT_lost=[z for z in previous_SORT if z[4] in lost_tracks]
    previous_track_num_matched_SORT_lost=[z for z in previous_track_num_matched_SORT if z[4] in lost_tracks and z[4] not in previous_SORT_index]
    combined_lost_sort=previous_SORT_lost+previous_track_num_matched_SORT_lost
    combined_lost_sort=sorted(combined_lost_sort,key=lambda x:x[4])
    lost_tracks.sort()
    old_centroids=[bbox_to_centroid(z) for z in combined_lost_sort if z[4] in lost_tracks]
    track_dict={}
    if len(new_index) ==0 or len(lost_tracks)==0:
        return {}
    elif len(new_index)==1:
    #else:
        for i in new_index:
            new_centroid=bbox_to_centroid([z for z in sort_tracks if z[4]==i][0])
            distances=[Distance(new_centroid,z) for z in old_centroids]
            if min(distances) <max_distance:
                track_dict[lost_tracks[distances.index(min(distances))]]=i
    elif len(new_index)>1 and len(lost_tracks)>1:
        distance_matrix=[]
        for i in new_index:
            new_centroid=bbox_to_centroid([z for z in sort_tracks if z[4]==i][0])
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

"""
function in the reconnect_tracks_main
"""
def reconnect_tracks_cage(frame_count,sort_tracks_cage,sort_index_cage,df_trackout):
    track_dict={}
    sort_index_cage.sort()
    sort_tracks_cage=sorted(sort_tracks_cage,key=lambda x:x[4])
    if frame_count-1==0:
        return {}
    else:
        previous_SORT_cage=df_trackout.loc[frame_count-1]['Sort_tracks_cage']
        previous_SORT_cage=sorted(previous_SORT_cage,key=lambda x:x[4])
        previous_SORT_index_cage=df_trackout.loc[frame_count-1]['Sort_nums_cage']
        previous_SORT_index_cage.sort()
    if sort_index_cage == previous_SORT_index_cage:
        return {}
    else:
        if len(df_trackout)<max_frames+1:
            max_detections=max(df_trackout['Num_detections_cage'])
        else:
            max_detections=max(df_trackout.loc[frame_count-max_frames:frame_count]['Num_detections_cage'])
        if len(sort_index_cage)>0 and len(sort_index_cage) <= max_detections:
            if len(sort_index_cage)<len(previous_SORT_index_cage):
                track_dict=reconnect_tracks_cage_1(frame_count,sort_tracks_cage,sort_index_cage,previous_SORT_index_cage,
                                                   previous_SORT_cage,df_trackout)
            else:
                track_dict=reconnect_tracks_cage_2(frame_count,sort_tracks_cage,sort_index_cage,previous_SORT_index_cage,
                                                   previous_SORT_cage,df_trackout)
        else:
            pass
    return track_dict

def reconnect_tracks_cage_1(frame_count,sort_tracks_cage,sort_index_cage,previous_SORT_index_cage,previous_SORT_cage,df_trackout):
    if len(previous_SORT_index_cage)==0:
        if len(df_trackout)<max_frames+1:
            start_frame=1
        else:
            start_frame=frame_count-max_frames
    if frame_count ==65:
        try:
            index_n= df_trackout.loc[start_frame:frame_count-1].query('Num_detections_cage!=0').index[-1]
            previous_SORT_index_cage=df_trackout.loc[index_n]['Sort_nums_cage']
            previous_SORT_index_cage.sort()
            previous_SORT_cage=df_trackout.loc[index_n]['Sort_tracks_cage']
            previous_SORT_cage=sorted(previous_SORT_cage,key=lambda x:x[4])
        except Exception:
            return {}
    track_dict={}
    new_index=[i for i in sort_index_cage if i not in previous_SORT_index_cage]
    tracked_index=[i for i in sort_index_cage if i in previous_SORT_index_cage]
    lost_tracks=[i for i in previous_SORT_index_cage if i not in tracked_index]
    #checks if the new index and lost index have been previously registered in tracks_history
    old_centroids=[bbox_to_centroid(z) for z in previous_SORT_cage if z[4] in lost_tracks]
    if len(new_index) ==0 or len(lost_tracks)==0:
        #does nothing beacuse there was no new tracks
        return {}
    else:
        if len(new_index)==1:
            for i in new_index:
                new_centroid=bbox_to_centroid([z for z in sort_tracks_cage if z[4]==i][0])
                distances=[Distance(new_centroid,z) for z in old_centroids]
                if min(distances) <max_distance:
                    track_dict[lost_tracks[distances.index(min(distances))]]=i
                else:
                    pass
        #a little more complex if there is more than one new index
        #the new track maybe closest to mulitple previous tracks
        #used linear sum assignment to solve the issue
        # also discards the result if the distance to of the new track to old centroid assigned is larger than max distance 
        #( can't associate to a matrix too far), better to have it as a new track to verify
        # essentially this process can return nothing 
        # potential for identity swaps
        elif len(new_index)>1:
            distance_matrix=[]
            for i in new_index:
                new_centroid=bbox_to_centroid([z for z in sort_tracks_cage if z[4]==i][0])
                distances=[Distance(new_centroid,z) for z in old_centroids]
                distance_matrix.append(distances)
            distance_matrix=np.array(distance_matrix)
            _,col = linear_sum_assignment(distance_matrix)
            for i,z in zip(new_index,col):
                if distance_matrix[new_index.index(i)][z]<max_distance:
                    track_dict[lost_tracks[z]]=i
                else:
                    pass
    #print(track_dict)
    return track_dict

def reconnect_tracks_cage_2(frame_count,sort_tracks_cage,sort_index_cage,previous_SORT_index_cage,previous_SORT_cage,df_trackout):
    #number of detections in the current frame
    N_detections=len(sort_index_cage)
    # finds the previous frame which there are the same number of tracks, does nothing if same n tracks not found in
    #the previous frames (max_frames) 
    if len(df_trackout)<max_frames+1:
        start_frame=1
    else:
        start_frame=frame_count-max_frames
    try: 
        index_n= df_trackout.loc[start_frame:frame_count-1].query(f'Num_detections_cage=={str(N_detections)}').index[-1]
    except Exception:
        return {}
    # gets the another previous frame to reference if the previous frame grabbed has zero tracks
    if len(previous_SORT_index_cage)==0:
        index_n2= df_trackout.loc[start_frame:frame_count-1].query('Num_detections_cage!=0').index[-1]
        previous_SORT_index_cage=df_trackout.loc[index_n2]['Sort_nums_cage']
        previous_SORT_index_cage.sort()
        previous_SORT_cage=df_trackout.loc[index_n2]['Sort_tracks_cage']
        previous_SORT_cage=sorted(previous_SORT_cage,key=lambda x:x[4])
    previous_track_num_matched=df_trackout.loc[index_n]['Sort_nums_cage']
    previous_track_num_matched.sort()
    previous_track_num_matched_SORT=df_trackout.loc[index_n]['Sort_tracks_cage']
    previous_track_num_matched_SORT=sorted(previous_track_num_matched_SORT,key=lambda x:x[4])
    new_index=[i for i in sort_index_cage if i not in previous_SORT_index_cage and i not in previous_track_num_matched]
    tracked_index=[i for i in sort_index_cage if i in previous_SORT_index_cage or i in previous_track_num_matched]
    lost_tracks=[i for i in list(set(previous_SORT_index_cage +previous_track_num_matched)) if i not in tracked_index]
    previous_SORT_lost=[z for z in previous_SORT_cage if z[4] in lost_tracks]
    previous_track_num_matched_SORT_lost=[z for z in previous_track_num_matched_SORT if z[4] in lost_tracks and z[4] not in previous_SORT_index_cage]
    combined_lost_sort=previous_SORT_lost+previous_track_num_matched_SORT_lost
    combined_lost_sort=sorted(combined_lost_sort,key=lambda x:x[4])
    lost_tracks.sort()
    old_centroids=[bbox_to_centroid(z) for z in combined_lost_sort if z[4] in lost_tracks]
    track_dict={}
    if len(new_index) ==0 or len(lost_tracks)==0:
        return {}
    elif len(new_index)==1:
    #else:
        for i in new_index:
            new_centroid=bbox_to_centroid([z for z in sort_tracks_cage if z[4]==i][0])
            distances=[Distance(new_centroid,z) for z in old_centroids]
            if min(distances) <max_distance:
                track_dict[lost_tracks[distances.index(min(distances))]]=i
    elif len(new_index)>1:
        distance_matrix=[]
        for i in new_index:
            new_centroid=bbox_to_centroid([z for z in sort_tracks_cage if z[4]==i][0])
            distances=[Distance(new_centroid,z) for z in old_centroids]
            distance_matrix.append(distances)
        distance_matrix=np.array(distance_matrix)
        rnd,col = linear_sum_assignment(distance_matrix)
        for i,z in zip(new_index,col):
            if distance_matrix[new_index.index(i)][z]<max_distance:
                track_dict[lost_tracks[z]]=i
            else:
                pass
    return track_dict


def connect_tracts_cage(track_dict_cage,sort_index_cage,sort_tracks_cage):
    sort_index_cage.sort()
    sort_tracks_cage=sorted(sort_tracks_cage,key=lambda x:x[4])
    if len(track_dict_cage) ==0:
        return sort_index_cage,sort_tracks_cage
    else:
         for k,v in track_dict_cage.items():
             if k>v: 
                 pass
             elif k in sort_index_cage:
                 pass
             else:
                 sort_tracks_cage[sort_index_cage.index(v)][4]=k
                 sort_index_cage[sort_index_cage.index(v)]=k
         sort_index_cage.sort()
         sort_tracks_cage=sorted(sort_tracks_cage,key=lambda x:x[4])
    return sort_index_cage,sort_tracks_cage


"""
Function to implement changes processed by reconnect main  
"""
def connect_tracts(track_dict,sort_index,sort_tracks):
    sort_index.sort()
    sort_tracks=sorted(sort_tracks,key=lambda x:x[4])
    if len(track_dict) ==0:
        return sort_index,sort_tracks
    else:
         for k,v in track_dict.items():
             if k>v: 
                 pass
             elif k in sort_index:
                 pass
             else:
                 sort_tracks[sort_index.index(v)][4]=k
                 sort_index[sort_index.index(v)]=k
         sort_index.sort()
         sort_tracks=sorted(sort_tracks,key=lambda x:x[4])
    return sort_index,sort_tracks


def recconnect_sort_tracts(frame_count,sort_tracks,df_RFID_cage,df_track_temp):
    if sort_tracks !=[]:
        sort_index=[i[4] for i in sort_tracks]
    else:
        sort_index=[]
   # sort_index,sort_tracks =spontaneous_BB_checker(frame_count,sort_index,sort_tracks,df_track_temp)
    track_reassignment_index=reconnect_tracks_main(frame_count,sort_tracks,sort_index,df_track_temp)
    sort_index,sort_tracks=connect_tracts(track_reassignment_index,sort_index,sort_tracks)
    sort_cage_index,sort_entrance_index,sort_cage_tracks, sort_entrance_tracks = track_splitter(sort_index,sort_tracks)
    track_reassignment_index_cage=reconnect_tracks_cage(frame_count,sort_cage_tracks,sort_cage_index,df_track_temp)
    sort_cage_index,sort_cage_tracks=connect_tracts_cage(track_reassignment_index_cage,sort_cage_index,sort_cage_tracks)
    sort_index=sort_cage_index + sort_entrance_index
    sort_tracks=sort_cage_tracks+sort_entrance_tracks
    RFID_readings=df_RFID_cage.loc[frame_count].tolist()[2:]
    RFID_readings={i:v for i,v in enumerate(RFID_readings) if v != '0'}
    if len(sort_tracks) >1:
        iou_dic=iou_tracks(sort_index,sort_tracks)
    else:
        iou_dic={}

    df_track_temp.loc[frame_count]=[frame_count,len(sort_index),sort_index,sort_tracks,iou_dic,
                                         len(sort_cage_index),sort_cage_index,sort_cage_tracks,
                                         len(sort_entrance_index),sort_entrance_index,
                                         sort_entrance_tracks,RFID_readings]
    return df_track_temp


def RFID_matching(df):
    df1=pd.DataFrame(columns=['frame','RFID','Track_matched'])
    df['RFID_tracks']=[list() for i in range(len(df.index))]
    validation_frames=[i for i in df.loc[df.RFID_readings !={}].index]
    index_delete=[]
    index_seq=[]
    for i,v in enumerate(validation_frames):
        if i<len(validation_frames)-1:
            if v+1 ==validation_frames[i+1]:
                index_seq.append(i)
    for i in index_seq:
        index1=validation_frames[i]
        index2=validation_frames[i+1]
        if df.loc[index1]['RFID_readings'] == df.loc[index2]['RFID_readings']:
            index_delete.append(i)
        else:
            pass
    validation_frames=[v-1 for i,v in enumerate(validation_frames) if i not in index_delete]
    print(validation_frames)
    #pbar = tqdm(total=len(validation_frames))
    for z in validation_frames:
        for i,v in df.iloc[z]['RFID_readings'].items():
            if v != 'None' and i !=3:
                track_distances=[distance_box_RFID(i,k) for k in df.iloc[z]['Sort_tracks']]
                track_distances= {t:k for t,k in enumerate(track_distances) if k<RFID_dist}
                if len(track_distances) >0:
                    index=min(track_distances,key=lambda k:track_distances[k])
                    sort_id=df.iloc[z]['Sort_tracks'][index][4]
                    df1.loc[z]= [z,df.iloc[z]['RFID_readings'],sort_id]
                    ious=[v for i,v in df.iloc[z]['iou_tracks'].items() if sort_id in i and v >overlap_thres]
                else: 
                    ious=[1,1,1,1]
                if len(ious) ==0 and len(df.iloc[z]['Sort_tracks']) !=0:
                    a=[k for k in df.iloc[z]['Sort_tracks'] if k[4]==sort_id]
                    track=[[a[0][0],a[0][1],a[0][2],a[0][3],v]]
                    df.iloc[z]['RFID_tracks']+=track
                    floating_index_f=1
                    floating_index_b=1
                    skip_counter_b=0
                    skip_counter_f=0
                    iou_foward=[i for i,v in df.iloc[z+floating_index_f]['iou_tracks'].items() if sort_id in i and v >overlap_thres]
                    iou_backward=[i for i,v in df.iloc[z-floating_index_b]['iou_tracks'].items() if sort_id in i and v >overlap_thres]
                else:
                    iou_foward=[1,1,1,1]
                    iou_backward=[1,1,1,1]
                while len(iou_backward)<1:
                    if sort_id in df.iloc[z-floating_index_b]['Sort_nums']:
                        a=[k for k in df.iloc[z-floating_index_b]['Sort_tracks'] if k[4]==sort_id]
                        track=[[a[0][0],a[0][1],a[0][2],a[0][3],v]]
                        if track not in df.iloc[z-floating_index_b]['RFID_tracks']:
                            df.iloc[z-floating_index_b]['RFID_tracks']+=track
                        #df.iloc[z-floating_index_b]['RFID_tracks']+=a
                        floating_index_b+=1
                        iou_backward=[i for i,v in df.iloc[z-floating_index_b]['iou_tracks'].items() if sort_id in i and v >overlap_thres]
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
                    if sort_id in df.iloc[z+floating_index_f]['Sort_nums']:
                        a=[k for k in df.iloc[z+floating_index_f]['Sort_tracks'] if k[4]==sort_id]
                        track=[[a[0][0],a[0][1],a[0][2],a[0][3],v]]
                        if track not in  df.iloc[z+floating_index_f]['RFID_tracks']:
                            df.iloc[z+floating_index_f]['RFID_tracks']+=track
                        floating_index_f+=1
                        iou_foward=[i for i,v in df.iloc[z+floating_index_f]['iou_tracks'].items() if sort_id in i and v >overlap_thres]
                    else:
                         skip_counter_f+=1
                         floating_index_f+=1
                    if skip_counter_f ==30 or z+floating_index_f==len(df)-1:
                        iou_foward=[1,1,1,1,1]
                    else:
                        pass
                else:
                    #print(floating_index_f+z+1)
                    #print(iou_foward)
                    pass
            if v != 'None' and i ==3:
                if max(df.iloc[z-entr_frames:z+entr_frames]['Num_detections_entrance'])>1:
                    pass
                elif max(df.iloc[z-entr_frames:z]['Num_detections_entrance']) ==1 and max(df.iloc[z:z+entr_frames]['Num_detections_entrance'])==1:
                    pass
                elif max(df.iloc[z-entr_frames:z]['Num_detections_entrance']) ==0 and max(df.iloc[z:z+entr_frames]['Num_detections_entrance']) ==1:
                    index_n=df.iloc[z+1:z+entr_frames].query('Num_detections_entrance ==1').index[0]
                    sort_id_entrance=df.iloc[index_n-1]['Sort_tracks_entrance'][0]
                    df1.loc[z]= [z,df.iloc[z]['RFID_readings'],sort_id_entrance]
                    floating_index_f=1
                    skip_counter_f=0
                    iou_foward=[i for i,v in df.iloc[index_n+floating_index_f]['iou_tracks'].items() if sort_id_entrance[4] in i and v >overlap_thres]
                    while len(iou_foward)<1:
                        if sort_id_entrance[4] in df.iloc[index_n+floating_index_f]['Sort_nums']:
                            a=[k for k in df.iloc[index_n+floating_index_f]['Sort_tracks'] if k[4]==sort_id_entrance[4]]
                            track=[[a[0][0],a[0][1],a[0][2],a[0][3],v]]
                            if track not in  df.iloc[index_n+floating_index_f]['RFID_tracks']:
                                df.iloc[index_n+floating_index_f]['RFID_tracks']+=track
                            floating_index_f+=1
                            iou_foward=[i for i,v in df.iloc[index_n+floating_index_f]['iou_tracks'].items() if sort_id_entrance[4] 
                                        in i and v >0.17]
                        else:
                            skip_counter_f+=1
                            floating_index_f+=1
                        if skip_counter_f ==30 or index_n+floating_index_f==len(df)-1:
                            iou_foward=[1,1,1,1,1]
                        else:
                            pass
                elif max(df.iloc[z:z+entr_frames]['Num_detections_entrance']) ==0 and max(df.iloc[z-entr_frames:z]['Num_detections_entrance']) ==1:
                    if z ==1544:
                        a=df.iloc[z-entr_frames:z-1]
                    #print(df.iloc[z-entr_frames:z].query('Num_detections_entrance ==1'))
                    index_n=df.iloc[z-entr_frames:z].query('Num_detections_entrance ==1').index[-1]
                    sort_id_entrance=df.iloc[index_n-1]['Sort_tracks_entrance'][0]
                    track=[[sort_id_entrance[0],sort_id_entrance[1],sort_id_entrance[2],sort_id_entrance[3],v]]
                    df.iloc[z]['RFID_tracks']+=track
                    floating_index_b=1
                    skip_counter_b=0
                    iou_backward=[i for i,v in df.iloc[index_n-floating_index_b]['iou_tracks'].items() if sort_id_entrance[4] in i and v >overlap_thres]
                    while len(iou_backward)<1:
                        if sort_id_entrance[4] in df.iloc[index_n-floating_index_b]['Sort_nums']:
                            a=[k for k in df.iloc[index_n-floating_index_b]['Sort_tracks'] if k[4]==sort_id_entrance[4]]
                            track=[[a[0][0],a[0][1],a[0][2],a[0][3],v]]
                            if track not in  df.iloc[index_n-floating_index_b]['RFID_tracks']:
                                df.iloc[index_n-floating_index_b]['RFID_tracks']+=track
                            floating_index_b+=1
                            iou_foward=[i for i,v in df.iloc[index_n-floating_index_b]['iou_tracks'].items() if sort_id_entrance[4]
                                        in i and v >0.17]
                        else:
                            skip_counter_b+=1
                            floating_index_b+=1
                        if skip_counter_b ==30 or index_n+floating_index_b==len(df)-1:
                            iou_backward=[1,1,1,1,1]
                        else:
                            pass
    #pbar.update(1)
    df['RFID_tracks']= df['RFID_tracks'].map(lambda x: duplicate_remove(x))
    return df,df1,validation_frames


def get_lost_tracks(df,dfr):
    df['lost_tracks']=[list() for i in range(len(df))]
    for i in range(len(df)):
        if len(df.iloc[i]['Sort_tracks']) ==0:
            pass
        elif len(df.iloc[i]['RFID_tracks']) ==0 and len(df.iloc[i]['Sort_tracks']) !=0:
            df.iloc[i]['lost_tracks']+=df.iloc[i]['Sort_tracks']
        else:
            temp_to_comapre=[z[:4] for z in df.iloc[i]['RFID_tracks']]
            temp_sort= [z[:4] for z in df.iloc[i]['Sort_tracks']]
            index_lost_tracks=[z for z,v in enumerate(temp_sort) if v not in temp_to_comapre]
            lost_tracks=[v for z,v in enumerate(df.iloc[i]['Sort_tracks']) if z in index_lost_tracks]
            df.iloc[i]['lost_tracks']+=lost_tracks
    df['Timestamp']=dfr['Time']
    return df




"""
Class for Mouse tracking
"""
class mouse_tracker:
    def __init__(self,tags,path,df_RFID_cage,writer,vid_length,batch=300):
        self.tags={i:'NA' for i in tags} #starts a dictionary for the tags about their where abouts
        self.df_RFID_cage = df_RFID_cage
        self.df_track_temp=pd.DataFrame(columns=['frame','Num_detections','Sort_nums','Sort_tracks','iou_tracks',
                                                 'Num_detections_cage','Sort_nums_cage','Sort_tracks_cage',
                                                 'Num_detections_entrance','Sort_nums_entrance',
                                                 'Sort_tracks_entrance','RFID_readings'])
        self.writer= cv2.VideoWriter(writer[0], writer[1],writer[2], (800,800))
        self.colours=np.random.rand(300, 3) * 255
        self.batch=0
        self.path=path
        self.batch_size=batch
        self.vid_length=vid_length
        if not os.path.exists(f'{self.path}/tracks'):
            os.mkdir(f'{self.path}/tracks')
    def update(self, frame_count, sort_tracks):
        self.df_track_temp=recconnect_sort_tracts(frame_count,sort_tracks,self.df_RFID_cage,self.df_track_temp)
        if len(self.df_track_temp) == self.batch_size:
            self.batch +=1
            df_RFID_tracks_save=self.df_track_temp[:self.batch_size-100]
            df_RFID_tracks_save.to_csv(f'{self.path}/tracks/tracks_{str(self.batch)}.csv')
            self.df_track_temp=self.df_track_temp[self.batch_size-100:]
        elif self.vid_length ==frame_count:
            self.batch +=1
            self.df_track_temp.to_csv(f'{self.path}/tracks/tracks_{str(self.batch)}.csv')
        else:
            pass
    def evaulate(self):
        path_list=os.listdir(f'{self.path}/tracks')
        self.df_track_temp=pd.concat([df_tracks_read(f'{self.path}/tracks/'+i) for i in path_list])
        self.df_track_temp=self.df_track_temp.sort_index()
        self.df_track_temp,df_matches,validation_frames=RFID_matching(self.df_track_temp)
        self.validation_frames=validation_frames
        self.df_track_temp=get_lost_tracks(self.df_track_temp,self.df_RFID_cage)
        return self.df_track_temp, df_matches
    def write_video(self,frame_count,img,Vid_type):
        if Vid_type == 'RFID':
            tracker=self.df_track_temp.loc[frame_count]['RFID_tracks']
        elif Vid_type == 'Sort':
            tracker=self.df_track_temp.loc[frame_count]['Sort_tracks']
        RFID_frame=[i for i in self.validation_frames if i>frame_count]
        if len(RFID_frame) >0:
            RFID_frame=min(RFID_frame)
        blankimg=180*np.ones(shape=[800,800,3],dtype=np.uint8)
        blankimg[200:600,144:656]=img
        for object in tracker:
            xmin, ymin, xmax, ymax, index = int(object[0]), int(object[1]), int(object[2]), int(object[3]), int(object[4])
            color = (int(self.colours[index % 300, 0]), int(self.colours[index % 300, 1]), int(self.colours[index % 300, 2]))
            cv2.rectangle(blankimg, (xmin+144, ymin+200), (xmax+144, ymax+200), color, 3)
            cv2.putText(blankimg, str(index), (xmin+144, ymin+180), 0, 5e-3 * 200, color, 3)
        cv2.putText(blankimg,f"Frame: {str(frame_count)}",(320,150),0, 5e-3 * 200,(0,0,255),5)
        spacer=0
        if RFID_frame != []:
            for i,v in self.df_track_temp.iloc[RFID_frame]['RFID_readings'].items():
                cv2.putText(blankimg,f'Frame {RFID_frame}: reader {i} , tag read {v}', (20,650+spacer),0,5e-3 * 200,(0,0,255),2)
                spacer+=100
        self.writer.write(blankimg)
        return blankimg
    def write_out(self):
        self.df_track_temp.to_csv('test_output.csv')

