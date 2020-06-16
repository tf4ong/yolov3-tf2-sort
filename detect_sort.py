#modified detect video from yolov3-tf2
from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import tensorflow as tf
from yolov3_tf2.models import YoloV3
from yolov3_tf2.dataset import transform_images
from yolov3_tf2.utils import get_object_details,get_center
import RFID_matching as rm
import pandas as pd
#sort
from sort import Sort
import numpy as np
from tqdm import tqdm
############################
import os
import logging
import time
#import pandas as pd
#flags to work in codes 
flags.DEFINE_string('classes', './data/mouse.names', 'path to classes file')
flags.DEFINE_string('weights', './checkpoints/yolov3_train_44.tf',
                    'path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_string('data','./data','path to video and RFID csv file for tracker cage')
flags.DEFINE_string('output', 'None', 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_integer('num_classes', 1, 'number of classes in the model')
flags.DEFINE_string('Video_type','RFID','Sort tracks or RFID matched Video')
flags.DEFINE_string('showvid','True','Show video during process')

tags=[2018121360,2018121255, 801010273,2018121290]
####surprise tensorflow2 loggin information
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.FATAL)




def main(_argv):
    ##progress bar
    #Iniating sort tracker 
    mot_tracker = Sort()
    #starting yolov3 and related processes    delete_index=[]
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    yolo = YoloV3(classes=FLAGS.num_classes)
    yolo.load_weights(FLAGS.weights)
    class_names = [c.strip() for c in open(FLAGS.classes).readlines()]
    vid = cv2.VideoCapture(FLAGS.data+'/raw.avi')
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
    if FLAGS.output == 'None':
        FLAGS.output=f'{FLAGS.data}/Track_output.avi'
    writer=[FLAGS.output,codec,fps*2]
    #out = cv2.VideoWriter(FLAGS.output, codec, fps, (1000, 1000))
    df_RFID_cage=rm.load_RFID(FLAGS.data+'/RFID_data_all.csv')
    #starting mouse tracker processing
    vid_length=len(df_RFID_cage)
    pbar = tqdm(total=vid_length)
    mouse_tracks=rm.mouse_tracker(tags,FLAGS.data,df_RFID_cage,writer,vid_length)
    #starting detection and sort loop
    frame_count=0
    t1=time.time()
    while vid.isOpened():#reading frames
        ret,img=vid .read()
        #yolo process
        if ret:
            frame_count+=1
            img_in = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
            img_in = tf.expand_dims(img_in, 0)
            img_in = transform_images(img_in, FLAGS.size)
            boxes, scores, classes, nums = yolo.predict(img_in)
            #print(scores)
            objects, bb_start, bb_end, probability= get_object_details(img, (boxes, scores, classes, nums),class_names)
            ds_boxes=[] # array to feed into sort tracker
            if nums[0] != 0:
                for i in range(nums[0]):
                    center_coords, standard_cords= get_center(bb_start[i],bb_end[i])
                    standard_cords.append(round(probability[i],4))
                    ds_boxes.append(standard_cords)
            ds_boxes_array=np.asarray(ds_boxes)
            #updating sort tracker
            trackers = mot_tracker.update(ds_boxes_array)
            sort_tracks=[]
            for object in trackers:
                xmin, ymin, xmax, ymax, index = int(object[0]), int(object[1]), int(object[2]), int(object[3]), int(object[4])
                sort_tracker=[xmin, ymin, xmax, ymax, index]
                sort_tracks.append(sort_tracker)
            sort_tracks=sorted(sort_tracks,key=lambda x:x[4])
            #
            #updating mouse tracker
            mouse_tracks.update(frame_count, sort_tracks)
            pbar.update(1)
        else:
            if frame_count ==0:
                
                print('Unable to open video, please check video path')
                break
            else:

                vid.release()
                break
    t2=time.time()
    time_yolo_sort=t2-t1
    print(f'time taken for yolo_sort on {str(frame_count)}: {str(time_yolo_sort)} seconds')
    #associating RFID tag
    print('Associating RFID with Sort_ID')
    t3=time.time()
    df_RFID_tracks,df_matchs=mouse_tracks.evaulate()
    t4=time.time()
    RFID_matching_time=t4-t3
    print(f'RFID Matching of {str(frame_count)} took {str(RFID_matching_time)} seconds')
    print('Writing Video')
    vid = cv2.VideoCapture(FLAGS.data+'/raw.avi')
    frame_count=0
    pbar = tqdm(total=len(df_RFID_cage))
    df_RFID_tracks.to_csv(f'{FLAGS.data}/RFID_tracks.csv')
    df_matchs.to_csv(f'{FLAGS.data}/RFID_matches.csv')
    t5=time.time()
    while vid.isOpened():
        ret,img=vid .read()
        if ret:
            frame_count+=1
            edited_img=mouse_tracks.write_video(frame_count,img,FLAGS.Video_type)
            if eval(FLAGS.showvid):
                cv2.imshow('output', edited_img)
                if cv2.waitKey(1) == ord('q'):
                    break
            else:
                pass
            pbar.update(1)
        else:
            break
    t6=time.time()
    write_time=t6-t5
    print('Wrting {str(frame_count)} took {str(write_time) seconds}')
    print('All processes completed')
if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass










