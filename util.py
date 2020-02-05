from __future__ import division

import torch 
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
import numpy as np
import cv2 


def predict_transform(prediction, inp_dim, anchors, num_classes, CUDA = True):
    """
    Transforma un mapa de caracteristicas en un tensor 2D con las caracteristicas de las BBox
    """
    batch_size = prediction.size(0)
    stride =  inp_dim // prediction.size(2)
    grid_size = inp_dim // stride
    bbox_attrs = 5 + num_classes
    num_anchors = len(anchors)
    
    prediction = prediction.view(batch_size, bbox_attrs*num_anchors, grid_size*grid_size)
    prediction = prediction.transpose(1,2).contiguous()
    prediction = prediction.view(batch_size, grid_size*grid_size*num_anchors, bbox_attrs)

    anchors = [(a[0]/stride, a[1]/stride) for a in anchors]

    # Aplica la sigmoide a los centros (x,y) y a la 'confianza' del objeto (las capas 2 y 3 son las dimensiones de la caja), el resto de capas son probabilidad de que sea un tipo de clase (depende del numero de clases)
    prediction[:,:,0] = torch.sigmoid(prediction[:,:,0])    # x
    prediction[:,:,1] = torch.sigmoid(prediction[:,:,1])    # y
    prediction[:,:,4] = torch.sigmoid(prediction[:,:,4])    # confianza con la que contiene un objeto

    grid = np.arange(grid_size)
    a,b = np.meshgrid(grid, grid)

    x_offset = torch.FloatTensor(a).view(-1,1)
    y_offset = torch.FloatTensor(b).view(-1,1)

    if CUDA:
        x_offset = x_offset.cuda(0)
        y_offset = y_offset.cuda(0)

    x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1,num_anchors).view(-1,2).unsqueeze(0)

    # Centro de las BBox en referencia global (referencia de imagen en lugar de casilla)  
    prediction[:,:,:2] += x_y_offset

    # Anchors con la dimension de la BBox
    anchors = torch.FloatTensor(anchors)

    if CUDA:
        anchors = anchors.cuda(0)

    anchors = anchors.repeat(grid_size*grid_size, 1).unsqueeze(0)
    prediction[:,:,2:4] = torch.exp(prediction[:,:,2:4])*anchors	#cmultiplica ancho y alto de BBox por la de las anchor boxes

    # Aplica sigmoide a la probabilidad de cada clase
    prediction[:,:,5: 5 + num_classes] = torch.sigmoid((prediction[:,:, 5 : 5 + num_classes]))

    prediction[:,:,:4] *= stride    # reescalado de las predicciones para que coincidan con las dimensiones del input

    return prediction


def write_results(prediction, confidence, num_classes, nms_conf = 0.4):
    """
    La prediccion tiene dimensiones de B*10647*(5+num_classes). B es el numero de imagenes por batch, 10647 son las BBox, (5+num_clases) contiene
    centro (x,y) de las BBox, dimensiones (w,h) de las BBox, confianza con la que contiene un objeto y las probabilidades de que sea de una clase 
    (num_classes).
    
    Devuelve un vector D*8, siendo D el numero de detecciones verificadas y 8 valores para cada una (indice de la imagen en el batch, 4 coordenadas
    de esquinas ((x1,y1) y (x2,y2)), confianza de objeto, confianza de la clase mas probable y el indice de la clase)
    """

    conf_mask = (prediction[:,:,4] > confidence).float().unsqueeze(2)   # filtro de confianza de objeto (los que no lo pasan se quedan como 0)
    prediction = prediction*conf_mask

    # Trabajamos con las coordenadas de las esquinas de las BBox en lugar de centro (x,y) y dimensiones (h,w)
    box_corner = prediction.new(prediction.shape)
    box_corner[:,:,0] = (prediction[:,:,0] - prediction[:,:,2]/2)   # x-w/2 (x esquina 1)
    box_corner[:,:,1] = (prediction[:,:,1] - prediction[:,:,3]/2)   # y-h/2 (y esquina 1)
    box_corner[:,:,2] = (prediction[:,:,0] + prediction[:,:,2]/2)   # x+w/2 (x esquina 2)
    box_corner[:,:,3] = (prediction[:,:,1] + prediction[:,:,3]/2)   # y+h/2 (y esquina 2)
    prediction[:,:,:4] = box_corner[:,:,:4]

    batch_size = prediction.size(0)

    write = False   # indica que todavia no se ha inicializado output, el tensor que guarda las true detections (confianza>threshhold)

    for ind in range(batch_size):   # bucle dentro de las imagenes de cada batch
        image_pred = prediction[ind]
        max_conf, max_conf_score = torch.max(image_pred[:,5:5+ num_classes], 1) # guarda la clase con mayor puntuacion y la puntuacion que tiene
        max_conf = max_conf.float().unsqueeze(1)
        max_conf_score = max_conf_score.float().unsqueeze(1)
        seq = (image_pred[:,:5], max_conf, max_conf_score)      # 4 valores de BBox, confianza de objeto, clase, confianza de clase
        image_pred = torch.cat(seq, 1)

        non_zero_ind =  (torch.nonzero(image_pred[:,4]))    # elimina las filas con ceros
        try:
            image_pred_ = image_pred[non_zero_ind.squeeze(),:].view(-1,7)
        except:
            continue    # sale del bucle en caso de que no haya ninguna BBox con confianza de objeto > threshhold en la imagen

        if image_pred_.shape[0] == 0:
            continue    # equivalente por temas de compatibilidad con pytorch 0.4
        

        # Coge las clases detectadas en la imagen    
        img_classes = unique(image_pred_[:,-1]) # -1 mantiene el indice de la clase
        
        for cls in img_classes:
            # Deteccion para una clase en particular
            cls_mask = image_pred_*(image_pred_[:,-1] == cls).float().unsqueeze(1)
            class_mask_ind = torch.nonzero(cls_mask[:,-2]).squeeze()
            image_pred_class = image_pred_[class_mask_ind].view(-1,7)

            # La entrada con la mayor confianza de objeto se pone en primer lugar
            conf_sort_index = torch.sort(image_pred_class[:,4], descending = True )[1]
            image_pred_class = image_pred_class[conf_sort_index]
            idx = image_pred_class.size(0)   # numero de detecciones

            # NMS
            for i in range(idx):
                # Coge los IoUs de todas las cajas que vienen despues de la que buscamos en el bucle
                try:
                    ious = bbox_iou(image_pred_class[i].unsqueeze(0), image_pred_class[i+1:])
                except ValueError:
                    break

                except IndexError:
                    break

                # Cero para todas las detecciones con IoU > treshhold
                iou_mask = (ious < nms_conf).float().unsqueeze(1)
                image_pred_class[i+1:] *= iou_mask       

                # Eliminacion de las filas con no-ceros
                non_zero_ind = torch.nonzero(image_pred_class[:,4]).squeeze()
                image_pred_class = image_pred_class[non_zero_ind].view(-1,7)
        
            batch_ind = image_pred_class.new(image_pred_class.size(0), 1).fill_(ind)      
            
            # Repite el batch_ind para tantas detecciones de la clase cls haya en la imagen
            seq = batch_ind, image_pred_class

            if not write:
                output = torch.cat(seq,1)
                write = True
            else:
                out = torch.cat(seq,1)
                output = torch.cat((output,out))

    try:
        return output
    except:
        return 0


def unique(tensor):
    """
    Tensor único en pytorch a partir de uno de numpy
    """

    tensor_np = tensor.cpu().numpy()
    unique_np = np.unique(tensor_np)
    unique_tensor = torch.from_numpy(unique_np)
    
    tensor_res = tensor.new(unique_tensor.shape)
    tensor_res.copy_(unique_tensor)
    return tensor_res


def bbox_iou(box1, box2):
    """
    Devuelve el IoU de dos BBox

    """
    # Coordenadas BBox
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:,0], box1[:,1], box1[:,2], box1[:,3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:,0], box2[:,1], box2[:,2], box2[:,3]
    
    # Coordenadas del rectangulo interseccion
    inter_rect_x1 =  torch.max(b1_x1, b2_x1)
    inter_rect_y1 =  torch.max(b1_y1, b2_y1)
    inter_rect_x2 =  torch.min(b1_x2, b2_x2)
    inter_rect_y2 =  torch.min(b1_y2, b2_y2)
    
    # Area de la interseccion
    if torch.cuda.is_available():
            inter_area = torch.max(inter_rect_x2 - inter_rect_x1 + 1,torch.zeros(inter_rect_x2.shape).cuda())*torch.max(inter_rect_y2 - inter_rect_y1 + 1, torch.zeros(inter_rect_x2.shape).cuda())
    else:
            inter_area = torch.max(inter_rect_x2 - inter_rect_x1 + 1,torch.zeros(inter_rect_x2.shape))*torch.max(inter_rect_y2 - inter_rect_y1 + 1, torch.zeros(inter_rect_x2.shape))
    
    # Area de la union
    b1_area = (b1_x2 - b1_x1 + 1)*(b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1)*(b2_y2 - b2_y1 + 1)
    
    iou = inter_area / (b1_area + b2_area - inter_area)
    
    return iou


def letterbox_image(img, inp_dim):
    """
    Modifica el tamaño de una imagen conservando el radio de aspecto
    """
    img_w, img_h = img.shape[1], img.shape[0]
    w, h = inp_dim
    new_w = int(img_w * min(w/img_w, h/img_h))
    new_h = int(img_h * min(w/img_w, h/img_h))
    resized_image = cv2.resize(img, (new_w,new_h), interpolation = cv2.INTER_CUBIC)
    
    canvas = np.full((inp_dim[1], inp_dim[0], 3), 128)

    canvas[(h-new_h)//2:(h-new_h)//2 + new_h,(w-new_w)//2:(w-new_w)//2 + new_w,  :] = resized_image
    
    return canvas


def load_classes(namesfile):
    fp = open(namesfile, "r")
    names = fp.read().split("\n")[:-1]
    return names


def prep_image_f(img, inp_dim):
    """
    Configura una imagen para introducirla en una red neuronal de Pytorch. 
    Devuelve una Variable.
    Version para archivo (imagenes o video)
    """

    orig_im = img
    dim = orig_im.shape[1], orig_im.shape[0]
    img = (letterbox_image(orig_im, (inp_dim, inp_dim)))
    img_ = img[:,:,::-1].transpose((2,0,1)).copy()
    img_ = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)
    return img_


def prep_image_c(img, inp_dim):
    """
    Configura una imagen para introducirla en una red neuronal de Pytorch. 
    Devuelve una Variable.
    Version para video en vivo
    """
    orig_im = img
    dim = orig_im.shape[1], orig_im.shape[0]
    img = cv2.resize(orig_im, (inp_dim, inp_dim))
    img_ = img[:,:,::-1].transpose((2,0,1)).copy()
    img_ = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)
    return img_
