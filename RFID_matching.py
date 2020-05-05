import pandas as pd
import numpy as np
import math
import itertools
from scipy.optimize import linear_sum_assignment

RFID_coords={0:[9, 40, 81, 128],1:[14, 244, 90, 341],2:[473, 228, 512, 306],3:[473, 228, 512, 306],4:[432, 50, 511, 129]}
max_distance=170
max_frames=35
def Distance(centroid1,centroid2):  
     dist = math.sqrt((centroid2[0] - centroid1[0])**2 + (centroid2[1] - centroid1[1])**2)  
     return dist
def bbox_area(bbox):
    w=bbox[0]+bbox[2]
    h=bbox[1]+bbox[3]
    area=w*h
    return area
def bbox_to_centroid(bbox):
    cX=int((bbox[0]+bbox[2])/2)
    cY=int((bbox[1]+bbox[3])/2)
    return [cX,cY]def reconnect_tracks_main(frame_count,sort_tracks,sort_index,df_trackout):
    '''
    takes in the current frame count, trackers produced by SORT, 
    the SORT track indexes, and the dataframe for storing the detection information
    return a dictionary in which the SORT tracks should be reassigned to a previous SORT track
    Uses the function reconnect_tracks_1 for abd reconnect_tracks_2
    '''
    #only start the reaasociation process of the tracks 
    #where there are more than 10 frames which is approximately 1 second
    # don't do anything if there isn't more than the number of frames the function can look up
    track_dict={}
    if frame_count<max_frames:
        return {}
    else:
        sort_index.sort()
        sort_tracks=sorted(sort_tracks,key=lambda x:x[4])
    #get the previous SORT tracks and indexes
        previous_SORT=df_trackout.loc[frame_count-1]['sort_tracks']
        previous_SORT=sorted(previous_SORT,key=lambda x:x[4])
        previous_SORT_index=df_trackout.loc[frame_count-1]['sort_track_nums']
        previous_SORT_index.sort()
    # if the current SORT index is equal to the previous SORT indexe then return empty dict
    # there are no tracks to reassociate
        if sort_index == previous_SORT_index:
            print('same as before')
            return {}
        else:
    # only starts the process if the number of current SORT track index is larger than 0,does nothing if there is no tracks this frame
    # frame rate is approximately 10 fps
    # assumes if that there is something in the previous 25 frames that can be referenced
    # depends on how well SORT handles occulsions
            if len(sort_index)>0 and len(sort_index) <= max(df_trackout.loc[frame_count-max_frames:frame_count]['Num_detections']):
                #no new mice have appeared
                if len(sort_index)<len(previous_SORT_index):
                    # mouse disappeared
                    print('reconnect1')
                    track_dict=reconnect_tracks_1(frame_count,sort_tracks,sort_index,previous_SORT_index,previous_SORT,df_trackout)
                else:
                    print('connect2')
                    track_dict=reconnect_tracks_2(frame_count,sort_tracks,sort_index,previous_SORT_index,previous_SORT,df_trackout)
            #elif len(sort_index)>0 and len(sort_index) > max(df_trackout.loc[frame_count-20:frame_count]['Num_detections']):
            #    print('reconnect3')
            #    track_dict=reconnect_tracks_3(frame_count,sort_tracks,sort_index,previous_SORT_index,previous_SORT,df_trackout,threshold=0.8)
            else:
                print('increased n')
                return {}
        return track_dict
def load_RFID(path):
    df_RFID=pd.read_csv(path,index_col=False,dtype=str)
    df_RFID.Frame=df_RFID.Frame.astype(int)
    df_RFID=df_RFID.set_index('Frame')
    return df_RFID
def RFID_detection(df_RFID,frame_count):
    RFID_detections={}
    if df_RFID.loc[frame_count]['RFID_0'] !='0':
        RFID_detections[0]=df_RFID.loc[frame_count]['RFID_0']
    elif df_RFID.loc[frame_count]['RFID_1'] !='0':
        RFID_detections[1]=df_RFID.loc[frame_count]['RFID_1']
    elif df_RFID.loc[frame_count]['RFID_2'] !='0':
        RFID_detections[2]=df_RFID.loc[frame_count]['RFID_2']
    elif df_RFID.loc[frame_count]['RFID_3'] !='0':    sort_index.sort()
    sort_tracks=sorted(sort_tracks,key=lambda x:x[4])
        RFID_detections[3]=df_RFID.loc[frame_count]['RFID_3']
    elif df_RFID.loc[frame_count]['RFID_4'] !='0':
        RFID_detections[4]=df_RFID.loc[frame_count]['RFID_4']
    return RFID_detections
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


def rank_IOU(RFID_detections, sort_tracks):
    global RFID_coords
    iou_ranks={}
    reader_tags={}
    if len(RFID_detections) == 1:
        for reader, tag in RFID_detections.items():
            if tag !='None' and reader !=3:
                iou_rank=[iou(i,RFID_coords[int(reader)]) for i in sort_tracks]
                iou_ranks[reader]=iou_rank
                reader_tags[reader]=tag
    else:
        pass
    return iou_ranks,reader_tags

def iou_intersection(tracks,threshold=0.3):
    #function to determine if there are overlaps of bb in the frame
    #if there is overlap then gives the tracks nums of interacting pairs
    interaction_pair=[]
    for combinations in itertools.combinations(tracks,2):
        f=iou(combinations[0],combinations[1])
        if f >threshold:
            interaction_pair=[combinations[0][4],combinations[1][4]]

    return interaction_pair



def reconnect_tracks_main(frame_count,sort_tracks,sort_index,df_trackout):
    '''
    takes in the current frame count, trackers produced by SORT, 
    the SORT track indexes, and the dataframe for storing the detection information
    return a dictionary in which the SORT tracks should be reassigned to a previous SORT track
    Uses the function reconnect_tracks_1 for abd reconnect_tracks_2
    '''
    #only start the reaasociation process of the tracks 
    #where there are more than 10 frames which is approximately 1 second
    # don't do anything if there isn't more than the number of frames the function can look up
    track_dict={}
    if frame_count<max_frames:
        return {}
    else:
        sort_index.sort()
        sort_tracks=sorted(sort_tracks,key=lambda x:x[4])
    #get the previous SORT tracks and indexes
        previous_SORT=df_trackout.loc[frame_count-1]['sort_tracks']
        previous_SORT=sorted(previous_SORT,key=lambda x:x[4])
        previous_SORT_index=df_trackout.loc[frame_count-1]['sort_track_nums']
        previous_SORT_index.sort()
    # if the current SORT index is equal to the previous SORT indexe then return empty dict
    # there are no tracks to reassociate
        if sort_index == previous_SORT_index:
            print('same as before')
            return {}
        else:
    # only starts the process if the number of current SORT track index is larger than 0,does nothing if there is no tracks this frame
    # frame rate is approximately 10 fps
    # assumes if that there is something in the previous 25 frames that can be referenced
    # depends on how well SORT handles occulsions
            if len(sort_index)>0 and len(sort_index) <= max(df_trackout.loc[frame_count-max_frames:frame_count]['Num_detections']):
                #no new mice have appeared
                if len(sort_index)<len(previous_SORT_index):
                    # mouse disappeared
                    print('reconnect1')
                    track_dict=reconnect_tracks_1(frame_count,sort_tracks,sort_index,previous_SORT_index,previous_SORT,df_trackout)
                else:
                    print('connect2')
                    track_dict=reconnect_tracks_2(frame_count,sort_tracks,sort_index,previous_SORT_index,previous_SORT,df_trackout)
            #elif len(sort_index)>0 and len(sort_index) > max(df_trackout.loc[frame_count-20:frame_count]['Num_detections']):
            #    print('reconnect3')
            #    track_dict=reconnect_tracks_3(frame_count,sort_tracks,sort_index,previous_SORT_index,previous_SORT,df_trackout,threshold=0.8)
            else:
                print('increased n')
                return {}
        return track_dict

def reconnect_tracks_1(frame_count,sort_tracks,sort_index,previous_SORT_index,previous_SORT,df_trackout):
    '''
    Process to assign the tracks if the current track was not
    in the previous frame and number of current tracks 
    are smaller than the previous frame 
    '''
    if len(previous_SORT_index)==0:
        # gets a new previous frame to reference because there was no tracks in the previous frame
        # returns {} if no detections were found in the previous 10 frames
        try:
            index_n= df_trackout.iloc[frame_count-max_frames:frame_count-1].query('Num_detections!=0').index[-1]
            previous_SORT_index=df_trackout.loc[index_n]['sort_track_nums']
            previous_SORT_index.sort()
            previous_SORT=df_trackout.loc[index_n]['sort_tracks']
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
    print(sort_index)
    print(sort_tracks)
    print(previous_SORT_index)
    print(previous_SORT)
    print(new_index)
    print(tracked_index)
    print(lost_tracks)
    #if no new tracks appear do nothing
    if len(new_index) ==0:
        #does nothing beacuse there was no new tracks
        return {}
    #if there is only one new track match it to the closes object in the old_centroid dictionary
    else:
        if len(new_index)==1:
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
    print(track_dict)
    return track_dict

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
    
    try: 
        index_n= df_trackout.iloc[frame_count-max_frames:frame_count-1].query(f'Num_detections=={str(N_detections)}').index[-1]
    except Exception:
        return {}
    # gets the another previous frame to reference if the previous frame grabbed has zero tracks
    if len(previous_SORT_index)==0:
        index_n2= df_trackout.iloc[frame_count-max_frames:frame_count-1].query('Num_detections!=0').index[-1]
        previous_SORT_index=df_trackout.loc[index_n2]['sort_track_nums']
        previous_SORT_index.sort()
        previous_SORT=df_trackout.loc[index_n2]['sort_tracks']
        previous_SORT=sorted(previous_SORT,key=lambda x:x[4])
    previous_track_num_matched=df_trackout.loc[index_n]['sort_track_nums']
    previous_track_num_matched.sort()
    previous_track_num_matched_SORT=df_trackout.loc[index_n]['sort_tracks']
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
    print(sort_index)
    print(sort_tracks)
    print(previous_SORT_index)
    print(previous_SORT)
    print(previous_track_num_matched)
    print(previous_track_num_matched_SORT)
    print(combined_lost_sort)
    print(new_index)
    print(lost_tracks)
    print(tracked_index)
    print(old_centroids)
    if len(new_index) ==0:
        return {}
    elif len(new_index)==1:
    #else:
        print('Starting')
        for i in new_index:
            new_centroid=bbox_to_centroid([z for z in sort_tracks if z[4]==i][0])
            distances=[Distance(new_centroid,z) for z in old_centroids]
            print(i)
            print(distances)
            if min(distances) <max_distance:
                track_dict[lost_tracks[distances.index(min(distances))]]=i
    elif len(new_index)>1:
        print('STARTTING CONNECT @ DISTACNM')
        distance_matrix=[]
        for i in new_index:
            new_centroid=bbox_to_centroid([z for z in sort_tracks if z[4]==i][0])
            distances=[Distance(new_centroid,z) for z in old_centroids]
            distance_matrix.append(distances)
        distance_matrix=np.array(distance_matrix)
        rnd,col = linear_sum_assignment(distance_matrix)
        for i,z in zip(new_index,col):
            if distance_matrix[new_index.index(i)][z]<max_distance:
                track_dict[lost_tracks[z]]=i
            else:
                pass
    print(track_dict)
    return track_dict

def reconnect_tracks_3(frame_count,sort_tracks,sort_index,previous_SORT_index,previous_SORT,df_trackout,threshold=0.8):
    list_iou_entry=[iou(RFID_coords[2],i) for i in sort_tracks]
    list_iou_entry2=[list_iou_entry.index(i) for i in list_iou_entry if i>0.3]
    if len(list_iou_entry2) !=1:
        return {}
    else:
        sort_tracks=[i for i in sort_tracks if sort_tracks.index(i) not in list_iou_entry2]
        N_detections=len(sort_index)-1
        try: 
            index_n= df_trackout.iloc[frame_count-max_frames:frame_count-1].query(f'Num_detections=={str(N_detections)}').index[-1]
        except Exception:
            return {}
        previous_track_num_matched=df_trackout.loc[index_n]['sort_track_nums']
        previous_track_num_matched_SORT=df_trackout.loc[index_n]['sort_tracks']
        new_index=[i for i in sort_index if i not in previous_SORT_index and i not in previous_track_num_matched]
        tracked_index=[i for i in sort_index if i in previous_SORT_index or i in previous_track_num_matched]
        lost_tracks=[i for i in list(set(previous_SORT_index+ previous_track_num_matched)) if i not in tracked_index]
        old_centroids=[bbox_to_centroid(z) for z in previous_SORT+previous_track_num_matched_SORT if z[4] in lost_tracks]
        track_dict={}
        print(previous_track_num_matched)
        print(new_index)
        print(tracked_index)
        print(previous_SORT)
        print(previous_track_num_matched_SORT)
        print(previous_SORT+previous_track_num_matched_SORT)
        print(lost_tracks)
        if len(new_index) ==0:
            return {}
        else:
            for i in new_index:
                new_centroid=bbox_to_centroid([z for z in sort_tracks if z[4]==i][0])
                distances=[Distance(new_centroid,z) for z in old_centroids]
                if min(distances) <max_distance:
                    track_dict[lost_tracks[distances.index(min(distances))]]=i
                else:
                    pass
        return track_dict

def connect_tracts(track_dict,sort_index,sort_tracks):
    sort_index.sort()
    sort_tracks=sorted(sort_tracks,key=lambda x:x[4])
    if len(track_dict) ==0:
        return sort_index,sort_tracks
    else:
         for k,v in track_dict.items():
             if k>v: 
                 pass
             else:
                 sort_tracks[sort_index.index(v)][4]=k
                 sort_index[sort_index.index(v)]=k
                 #if k not in track_history:
                     #track_history[k]=[]
                 #if v not in tracks_history[k]:
                     #track_history[k].append(v)
    return sort_index,sort_tracks

def spontaneous_BB_checker(frame_count,sort_index,sort_tracks,df_trackout):
    sort_index.sort()
    sort_tracks=sorted(sort_tracks,key=lambda x:x[4])
    if len(sort_index) <= max(df_trackout.loc[frame_count-max_frames:frame_count-1]['Num_detections']):
        return  sort_index,sort_tracks
    else:
        list_iou_entry=[iou(RFID_coords[2],i) for i in sort_tracks]
        list_iou_entry2=[list_iou_entry.index(i) for i in list_iou_entry if i>0.3]
        iou_list=[]
        iou_area=[]
        if len(list_iou_entry2) !=0:
            return sort_index,sort_tracks
        else:
            for combinations in itertools.combinations(sort_tracks,2):
                iou_list.append([combinations[0][4],combinations[1][4]])
                iou_area.append(iou(combinations[0],combinations[1]))
            merge_tracks=iou_list[iou_area.index(max(iou_area))]
            temp_track={i[4]:bbox_area(i) for i in sort_tracks if i[4] in merge_tracks}
            print(temp_track)
            track_delete=min(temp_track,key=temp_track.get)
            sort_tracks=[i for i in sort_tracks if i[4] != track_delete]
            sort_index=[i for i in sort_index if i!=track_delete]
        return sort_index, sort_tracks
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        

