from __future__ import division
import torch 
from torch.autograd import Variable
import cv2 
from util import *
import argparse
from darknet import Darknet
import pickle as pkl
import random
import imutils
import time
from PyQt5 import QtGui

def arg_parse():
    '''
    Argumentos que pasarle al detector
    
    '''
    
    parser = argparse.ArgumentParser(description='YOLOv3 detector and tracker parameters')
   
    parser.add_argument('--source', dest = 'source', help = 'Name of the video or port to the camara to collect (0: camera, 1:webcam)', default = '0', type = str)
    parser.add_argument('--confidence', dest = 'confidence', help = 'Object Confidence to filter predictions', default = 0.5)
    parser.add_argument('--nms_thresh', dest = 'nms_thresh', help = 'NMS Threshhold', default = 0.4)
    parser.add_argument('--model', dest = 'model', help = 'folder with .cfg and .weights', default = 'COCO', type = str)
    parser.add_argument('--reso_det', dest = 'reso_det', help = 'Input resolution of the network. Increase to increase accuracy. Decrease to increase speed', default = '416', type = str)
    parser.add_argument('--reso_track', dest = 'reso_track', help = 'Input resolution for the tracking phase. Increase to increase accuracy. Decrease to increase speed', default = '320', type = str)
    parser.add_argument('--class', dest = 'filter', help = 'classes to filter', nargs = '*',default = [], type = str)
    parser.add_argument('--tracker', dest = 'tracker', help = 'OpenCV object tracker type', default = 'kcf', type = str)

    return parser.parse_args()


def write(x, img):
    global BBox  
    cls = int(x[-1])
    label = '{0}'.format(classes[cls])    
    
    if (true_class_filter and (label in true_class_filter)) or not true_class_filter:
        c1 = tuple(x[1:3].int())
        c2 = tuple(x[3:5].int())
        if BBox == []:
            BBox = [(c1,c2)]
        else:
            BBox.append((c1,c2))
        color = random.choice(colors)
        cv2.rectangle(img, c1, c2, color, 1)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
        c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
        cv2.rectangle(img, c1, c2, color, -1)
        cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1)
    

def click_det2track(event, x, y, flags, param):
    global phase, track_rect
    global frame, initBBox
    color = (0,255,0)
    rect = []
    if event == cv2.EVENT_LBUTTONDOWN:
        if phase == 'det':
            for c1,c2 in BBox:
                if c1 < (x,y) and c2 > (x,y):
                    rect.append((c1,c2))
            if len(rect) >= 1:  
                area = []
                for c1,c2 in rect:
                    x, y = c1  # esquina 1
                    x2, y2 = c2 # esquina 2
                    w, h = x2-x, y2-y
                    area.append(w*h)
                idx = area.index(min(area))
                x, y = rect[idx][0]   # esquina 1
                x2, y2 = rect[idx][1] # esquina 2
                w, h = x2-x, y2-y
                track_rect = (x,y,w,h)
                cv2.rectangle(frame,(x,y),(x+w,y+h),color,2) 
                phase = 'track'
                return 0
            else:
                return 0
        elif phase == 'track':
            phase = 'det'
            initBBox = []
            return 0


def prep_rect(x, y, w, h, ratio):
    x, y, w, h = x*ratio, y*ratio, w*ratio, h*ratio
    return int(x), int(y), int(w), int(h)

class DetectorTracker():
    def __init__(self, source, model_folder, class_filter, reso_det, tracker_alg, reso_track, confidence, nms_thresh, label_info=False):
        self.source = source
        self.model_folder = model_folder
        self.class_filter = class_filter
        self.reso_det = reso_det
        self.tracker_alg = tracker_alg
        self.reso_track = reso_track
        self.confidence = confidence
        self.nms_thresh = nms_thresh
        self.label_info = label_info
    
    def start(self):
        
        # Inicializacion de variables globales
        global classes, BBox, colors, phase, frame, initBBox, true_class_filter

        # PREPARACION DE LA FASE DE DETECCION

        CUDA = torch.cuda.is_available()

        text = 'No class filter selected'

        classes = load_classes('model/{}/model.names'.format(self.model_folder))
        colors = pkl.load(open('pallete', 'rb'))
        
        num_classes = len(classes)    
        if [i for i in self.class_filter if not (i in classes)]:
            if self.label_info:
                text = 'WARNING: {} class/classes are not included in the selected model. Updating the searching list...'.format([i for i in self.class_filter if not (i in classes)]) 
                self.label_info.setText(text)
            else:
                print('WARNING: {} class/classes are not included in the selected model. Updating the searching list...'.format([i for i in self.class_filter if not (i in classes)]) )
        true_class_filter=[i for i in self.class_filter if (i in classes)]

        # Configuracion de la red
        if self.label_info:
            text += '\nLoading network...'
            self.label_info.setText(text)
        else:
            print('Loading network.....')

        model = Darknet('model/{}/model.cfg'.format(self.model_folder))
        model.load_weights('model/{}/model.weights'.format(self.model_folder))

        if self.label_info:
            text += '\nNetwork succesfully loaded'
            self.label_info.setText(text)
        else:
            print('Network successfully loaded')

        model.net_info['height'] = self.reso_det
        inp_dim_det = int(model.net_info['height'])
        assert inp_dim_det % 32 == 0 
        assert inp_dim_det > 32 

        # Si hay un dispositivo CUDA se carga en el el modelo
        if CUDA:
            model.cuda()

        # Modelo en modo de evaluacion
        model.eval()

        # PREPARACION DE LA FASE DE TRACKING

        inp_dim_track = int(self.reso_track)

        OPENCV_OBJECT_TRACKERS = {
                'csrt': cv2.TrackerCSRT_create,
                'kcf': cv2.TrackerKCF_create,
                'boosting': cv2.TrackerBoosting_create,
                'mil': cv2.TrackerMIL_create,
                'tld': cv2.TrackerTLD_create,
                'medianflow': cv2.TrackerMedianFlow_create,
                'mosse': cv2.TrackerMOSSE_create
            }


        # INICIALIZACION DE LA FUENTE
        
        if self.source == '0' or self.source == '1':
            self.cap = cv2.VideoCapture(int(self.source))
            mode = 'cam'  
            self.window_name = 'Camera ' + self.source
        else:
            if self.label_info:     # via GUI se obtiene el path completo
                self.cap = cv2.VideoCapture(self.source)  
            else:                   # via terminal solo escribimos el nombre del archivo
                self.cap = cv2.VideoCapture('videos/{}'.format(self.source))  
            mode = 'file'
            self.window_name = self.source
        assert self.cap.isOpened(), 'Cannot capture source'
        

        phase = 'det'
        initBBox = []
        cont = 0
        frames = 0
        
        
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, click_det2track)
        
        while self.cap.isOpened():
            grab, frame = self.cap.read()

            start = time.time()

            if grab:    
                # Fase de deteccion
                if phase == 'det':

                    if mode == 'cam':
                        img = prep_image_c(frame, inp_dim_det)
                    elif mode == 'file':
                        img = prep_image_f(frame, inp_dim_det)
                    
                    im_dim = frame.shape[1], frame.shape[0]
                    im_dim = torch.FloatTensor(im_dim).repeat(1,2)   
                                
                    if CUDA:
                        im_dim = im_dim.cuda()
                        img = img.cuda()
                    

                    # Inicializacion la lista de BBox detectadas
                    BBox = []

                    output = model.forward(Variable(img), CUDA)
                    output = write_results(output, self.confidence, num_classes, nms_conf = self.nms_thresh)

                    if type(output) == int:
                        frames += 1
                        cv2.imshow(self.window_name, frame)
                        key = cv2.waitKey(1)
                        if key & 0xFF == ord('q'):
                            break
                        continue
                    
                    if mode == 'cam':

                        output[:,1:5] = torch.clamp(output[:,1:5], 0.0, float(inp_dim_det))

                        im_dim = im_dim.repeat(output.size(0), 1)/inp_dim_det
                        output[:,1:5] *= im_dim
                    
                    elif mode == 'file':

                        im_dim = im_dim.repeat(output.size(0), 1)
                        scaling_factor = torch.min(inp_dim_det/im_dim,1)[0].view(-1,1)
                        
                        output[:,[1,3]] -= (inp_dim_det - scaling_factor*im_dim[:,0].view(-1,1))/2
                        output[:,[2,4]] -= (inp_dim_det - scaling_factor*im_dim[:,1].view(-1,1))/2
                        
                        output[:,1:5] /= scaling_factor
                
                        for i in range(output.shape[0]):
                            output[i, [1,3]] = torch.clamp(output[i, [1,3]], 0.0, im_dim[i,0])
                            output[i, [2,4]] = torch.clamp(output[i, [2,4]], 0.0, im_dim[i,1])
                    
                    list(map(lambda x: write(x, frame), output))
                    
                    cv2.imshow(self.window_name, frame)                    
                    key = cv2.waitKey(1)
                    if key & 0xFF == ord('q'):
                        break
                    frames += 1
                    
                    if self.label_info:
                        self.label_info.setText(text + '\nDETECTION PHASE:' + '\n   {0: .2f} fps'.format(float(1/(time.time()-start))))
                
                
                # Fase de tracking
                elif phase == 'track':

                    ratio = frame.shape[0]/inp_dim_track

                    img = imutils.resize(frame, height=inp_dim_track)
                    
                    if initBBox:
                        (success, box) = tracker.update(img)

                        if success:
                            cont = 0
                            (x, y, w, h) = [int(v) for v in box]
                            x, y, w, h = prep_rect(x, y, w, h, ratio)
                            cv2.rectangle(frame, (x, y), (x + w, y + h),(0, 255, 0), 2)
                        
                        else:
                            cont += 1
                            if self.label_info:
                                self.label_info.setText(text + '\nTRACKING PHASE' + '\nObject lost ({})'.format(cont))
                            else:
                                print('Object lost ', cont)

                    else:
                        (x, y, w, h) = [int(v) for v in track_rect]
                        initBBox = (prep_rect(x, y, w, h, float(1/ratio)))
                        tracker = OPENCV_OBJECT_TRACKERS[self.tracker_alg]()
                        tracker.init(img, initBBox)


                    if cont > 100:
                        phase = 'det'
                        cont = 0
                        initBBox = []
                        
                    cv2.imshow(self.window_name, frame)                    
                    key = cv2.waitKey(1)
                    if key & 0xFF == ord('q'):
                        break
                    frames += 1
                    
                    if self.label_info:
                        self.label_info.setText(text + '\nTRACKING PHASE:' + '\n   {0: .2f} fps'.format(float(1/(time.time()-start))))          
                
                else:
                    break

            else: 
                break

        
        if not self.label_info:
            cv2.destroyWindow(self.window_name)
        
        self.cap.release()
        
        torch.cuda.empty_cache()


    def stop(self):
        cv2.destroyWindow(self.window_name)
        self.cap.release()

        torch.cuda.empty_cache()



if __name__ == '__main__':

    args = arg_parse()                          
    confidence = float(args.confidence)         
    nms_thresh = float(args.nms_thresh)         
    model_folder = args.model                 
    source = args.source                        
    class_filter = args.filter                 
    reso_det = args.reso_det                   
    tracker_alg = args.tracker                 
    reso_track = args.reso_track              

    det_track = DetectorTracker(source,model_folder,class_filter,reso_det,tracker_alg,reso_track,confidence,nms_thresh)
    det_track.start()
    
