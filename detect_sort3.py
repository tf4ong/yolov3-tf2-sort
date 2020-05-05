#modified detect video from yolov3-tf2
from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import tensorflow as tf
from yolov3_tf2.models import YoloV3
from yolov3_tf2.dataset import transform_images
from yolov3_tf2.utils import get_object_details,get_center
from RFID_matching import *
#sort
from sort import Sort
import numpy as np
import pandas as pd
#flags to work in codes 
flags.DEFINE_string('classes', './data/mouse.names', 'path to classes file')
flags.DEFINE_string('weights', './checkpoints/yolov3.tf',
                    'path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_string('data','./data','path to video and RFID csv file for tracker cage')
flags.DEFINE_string('output', None, 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_integer('num_classes', 1, 'number of classes in the model')


def main(_argv):
    global df_trackout
    #Iniating deepsort
    mot_tracker = Sort()
    track_history={}
    #starting yolov3
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    yolo = YoloV3(classes=FLAGS.num_classes)
    yolo.load_weights(FLAGS.weights)
    logging.info('weights loaded')
    class_names = [c.strip() for c in open(FLAGS.classes).readlines()]
    logging.info('classes loaded')
    vid = cv2.VideoCapture(FLAGS.data+'/raw.avi')    
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
    out = cv2.VideoWriter(FLAGS.output, codec, fps, (1000, 1000))
    frame_count=0
    colours = np.random.rand(300, 3) * 255
    #vid.set(cv2.CAP_PROP_POS_FRAMES,470)
    df_trackout=pd.DataFrame(columns=['Frame','Num_detections','Yolo_det','Sort_tracks'])
    df_RFID=load_RFID(FLAGS.data+'/RFID_data_all.csv')
    yolo_detection={}
    df_trackout=pd.DataFrame(columns=['Frame','Num_detections','Yolo_det','sort_tracks','sort_track_nums'])
    tracker_index={}
    while vid.isOpened():
        ret,img=vid .read()
        frame_count+=1
        if ret:
            img_in = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
            img_in = tf.expand_dims(img_in, 0)
            img_in = transform_images(img_in, FLAGS.size)
            boxes, scores, classes, nums = yolo.predict(img_in)
            if len(class_names) ==1:
                scores=scores*2
            #scores= scores*100
            objects, bb_start, bb_end, probability= get_object_details(img, (boxes, scores, classes, nums),class_names)
            #print(f'Frame {frame_count} analysis completed')
            #print(f'{str(nums[0])} mice detected')
            ds_boxes=[]
            if nums[0] != 0:
                for i in range(nums[0]):
                    center_coords, standard_cords= get_center(bb_start[i],bb_end[i])
                    #print(f'Mouse {str(i+1)}: coordinates({str(center_coords[0]),str(center_coords[1])}), {str(probability[i])}')
                    #print('')
                    standard_cords.append(round(probability[i],4))
                    ds_boxes.append(standard_cords)
            yolo_detection.update({frame_count:[len(ds_boxes),ds_boxes]})
            ds_boxes_array=np.asarray(ds_boxes)
            trackers = mot_tracker.update(ds_boxes_array)    
            sort_tracks=[]
            sort_tracker_nums=[]
            for object in trackers:
                xmin, ymin, xmax, ymax, index = int(object[0]), int(object[1]), int(object[2]), int(object[3]), int(object[4])
                sort_tracker=[xmin, ymin, xmax, ymax, index]
                color = (int(colours[index % 300, 0]), int(colours[index % 300, 1]), int(colours[index % 300, 2]))
                sort_tracks.append(sort_tracker)
                sort_tracker_nums.append(index)
            print(frame_count)
            if frame_count >50:
                sort_tracker_nums,sort_tracks =spontaneous_BB_checker(frame_count,sort_tracker_nums,sort_tracks,df_trackout)
            track_reassignment_index=reconnect_tracks_main(frame_count,sort_tracks,sort_tracker_nums,df_trackout)
            sort_tracker_nums ,sort_tracks=connect_tracts(track_reassignment_index,sort_tracker_nums,sort_tracks)
            RFID_detections=RFID_detection(df_RFID,frame_count)
            #if len(RFID_detections) != 0:
            #    iou_bb_RFID,reader_tags=rank_IOU(RFID_detections,sort_tracks)
            #    for i in  iou_bb_RFID.keys():
            #        index_key=iou_bb_RFID[i].index(max(iou_bb_RFID[i]))
            #        tracker_index[sort_tracks[index_key][4]]=reader_tags[i]
            #else:
            #    iou_bb_RFID,reader_tags={},{}
            blankimg=180*np.ones(shape=[1000,1000,3],dtype=np.uint8)
            blankimg[250:650,250:762]=img
            img=blankimg
            '''
            for object in sort_tracks:
                xmin, ymin, xmax, ymax, index = int(object[0]), int(object[1]), int(object[2]), int(object[3]), int(object[4])
                color = (int(colours[index % 300, 0]), int(colours[index % 300, 1]), int(colours[index % 300, 2]))
                #try: 
                #    label=tracker_index[int(index)]
                #except Exception:
                #    label='Dummy Mice'
                cv2.rectangle(img, (xmin+250, ymin+250), (xmax+250, ymax+250), color, 3)
                cv2.putText(img, str(index), (xmin+250, ymin+250), 0, 5e-3 * 200, color, 3)
            df_trackout.loc[frame_count]=[frame_count,len(sort_tracks),ds_boxes,sort_tracks,sort_tracker_nums]
            cv2.putText(img,str(frame_count),(500,800), 0, 5e-3 * 200,(255,255,0),4)
            '''
            out.write(img)
            cv2.imshow('output', img)
            if cv2.waitKey(1) == ord('q'):
                break
        else:
            if frame_count ==0:
                print('Unable to open video, please check video path')
                break
            else:
                cv2.destroyAllWindows()
                trackout.to_csv('Mice_detections.csv')
                print('All analysis complete')
                break
if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        df_trackout.to_csv('detection.csv')
        pass