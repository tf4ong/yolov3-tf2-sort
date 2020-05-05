#modified detect video from yolov3-tf2
from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import tensorflow as tf
from yolov3_tf2.models import YoloV3
from yolov3_tf2.dataset import transform_images
from yolov3_tf2.utils import draw_outputs,get_object_details,get_center
#sort
from sort import Sort
import numpy as np
#flags to work in codes 
flags.DEFINE_string('classes', './data/mouse.names', 'path to classes file')
flags.DEFINE_string('weights', './checkpoints/yolov3.tf',
                    'path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_string('video', './data/video.mp4',
                    'path to video file or number for webcam)')
flags.DEFINE_string('output', None, 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_integer('num_classes', 1, 'number of classes in the model')



def main(_argv):
    #Iniating deepsort
    mot_tracker = Sort()
    #starting yolov3
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    yolo = YoloV3(classes=FLAGS.num_classes)
    yolo.load_weights(FLAGS.weights)
    logging.info('weights loaded')
    class_names = [c.strip() for c in open(FLAGS.classes).readlines()]
    logging.info('classes loaded')
    vid = cv2.VideoCapture(FLAGS.video)    
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
    out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))
    frame_count=0
    appear={}
    number=0
    colours = np.random.rand(300, 3) * 255
    vid.set(cv2.CAP_PROP_POS_FRAMES,470)
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
            print(f'Frame {frame_count} analysis completed')
            print(f'{str(nums[0])} mice detected')
            ds_boxes=[]
            if nums[0] != 0:
                for i in range(nums[0]):
                    center_coords, standard_cords= get_center(bb_start[i],bb_end[i])
                    print(f'Mouse {str(i+1)}: coordinates({str(center_coords[0]),str(center_coords[1])}), {str(probability[i])}')
                    print('')
                    standard_cords.append(probability[i])
                    ds_boxes.append(standard_cords)

            ds_boxes=np.asarray(ds_boxes)
            print(ds_boxes)
            trackers = mot_tracker.update(ds_boxes)    
            print(trackers)
            for object in trackers:
                xmin, ymin, xmax, ymax, index = int(object[0]), int(object[1]), int(object[2]), int(object[3]), int(object[4])
                color = (int(colours[index % 300, 0]), int(colours[index % 300, 1]), int(colours[index % 300, 2]))
                cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 2)
                cv2.putText(img, str(index), (xmin, ymin), 0, 5e-3 * 200, color, 2)
                print(xmin,ymin,xmax,ymax)
                if index in appear.keys():
                    appear[index] += 1
                else:
                    number += 1
                    appear[index] = 1
            '''
            if len(ds_boxes)!=0:
                for i in range(len(ds_boxes)):
                    cv2.rectangle(img, (bb_start[i][0], bb_start[i][1]), (bb_end[i][0], bb_end[i][1]), (255, 0, 0), 2)
            #img = draw_outputs(img, (boxes, scores, classes, nums), class_names)
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
                print('All analysis complete')
                break
if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass     