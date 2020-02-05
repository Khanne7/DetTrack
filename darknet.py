from __future__ import division

import torch 
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
import numpy as np
from util import * 


def parse_cfg(cfgfile):
    """
    Convierte el archivo de configuracion de yolo en un diccionario de capas de la red.

    """

    file = open(cfgfile, 'r')
    lines = file.read().split('\n')                 # guarda el archivo de configuracion en una lista
    lines = [x for x in lines if len(x) > 0]        # obvia lineas en blanco 
    lines = [x for x in lines if x[0] != '#']       # obvia comentarios
    lines = [x.rstrip().lstrip() for x in lines]    # anula espacios en blanco a derecha e izquierda

    block = {}
    blocks = []

    for line in lines:
        if line[0] == "[":              # comienza un nuevo bloque
            if len(block) != 0:         # si no esta vacio, crea un bloque nuevo
                blocks.append(block)    # se añade el bloque a la lista de bloques
                block = {}              # reinicia el bloque
            block["type"] = line[1:-1].rstrip()     
        else:
            key,value = line.split("=") 
            block[key.rstrip()] = value.lstrip()
    
    blocks.append(block)

    return blocks


def create_modules(blocks):
    """
    Crea un modulo para cada uno de los bloques, diferente segun el tipo de capas que contenga

    """

    net_info = blocks[0]    # la primera capa ([net]) aporta informacion sobre el entrenamiento
    module_list = nn.ModuleList()
    prev_filters = 3    # inicializa los filtros previos con 3 porque se asume una imagen RGB
    output_filters = [] # guarda el numero de filtros de salida de cada capa

    for index, x in enumerate(blocks[1:]):
        module = nn.Sequential()

        if (x["type"] == "convolutional"):
            activation = x["activation"]
            try:
                batch_normalize = int(x["batch_normalize"])
                bias = False
            except:
                batch_normalize = 0
                bias = True

            filters= int(x["filters"])
            padding = int(x["pad"])
            kernel_size = int(x["size"])
            stride = int(x["stride"])

            if padding:
                pad = (kernel_size - 1) // 2
            else:
                pad = 0

            # Añade la capa convolucional
            conv = nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias = bias)
            module.add_module("conv_{0}".format(index), conv)

            # Añade la capa Batch Norm
            if batch_normalize:
                bn = nn.BatchNorm2d(filters)
                module.add_module("batch_norm_{0}".format(index), bn)

            # Comprueba el tipo de activacion 
            if activation == "leaky":
                activn = nn.LeakyReLU(0.1, inplace = True)
                module.add_module("leaky_{0}".format(index), activn)

        elif (x["type"] == "upsample"):
            stride = int(x["stride"])
            upsample = nn.Upsample(scale_factor = 2, mode = "nearest")
            module.add_module("upsample_{}".format(index), upsample)

        elif (x["type"] == "route"):
            x["layers"] = x["layers"].split(',')
            
            start = int(x["layers"][0])
            
            try:
                end = int(x["layers"][1])
            except:
                end = 0
            
            if start > 0: 
                start = start - index
            if end > 0:
                end = end - index
            route = EmptyLayer()    # este tipo de capa no esta pregenerada en pytorch por lo que se crea a partir de una vacia
            module.add_module("route_{0}".format(index), route)
            if end < 0:
                filters = output_filters[index + start] + output_filters[index + end]
            else:
                filters = output_filters[index + start]

        elif x["type"] == "shortcut":
            shortcut = EmptyLayer()
            module.add_module("shortcut_{}".format(index), shortcut)

        elif x["type"] == "yolo":
            mask = x["mask"].split(",")
            mask = [int(x) for x in mask]

            anchors = x["anchors"].split(",")
            anchors = [int(a) for a in anchors]
            anchors = [(anchors[i], anchors[i+1]) for i in range(0, len(anchors),2)]
            anchors = [anchors[i] for i in mask]

            detection = DetectionLayer(anchors)
            module.add_module("Detection_{}".format(index), detection)

        module_list.append(module)
        prev_filters = filters
        output_filters.append(filters)
    
    return (net_info, module_list)
    

class DetectionLayer(nn.Module):

    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors


class EmptyLayer(nn.Module):
    
    def __init__(self):
        super(EmptyLayer, self).__init__()


class Darknet(nn.Module):
    
    def __init__(self, cfgfile):
        super(Darknet, self).__init__()
        self.blocks = parse_cfg(cfgfile)
        self.net_info, self.module_list = create_modules(self.blocks)
    
    def forward(self, x, CUDA):
        modules = self.blocks[1:]
        outputs = {}   # guarda los outputs de cada capa para las capas shortcut y route
        write = 0     

        for i, module in enumerate(modules):        
            module_type = (module["type"])

            if module_type == "convolutional" or module_type == "upsample":                
                x = self.module_list[i](x)

            elif module_type == "route":
                layers = module["layers"]
                layers = [int(a) for a in layers]

                if (layers[0]) > 0:
                    layers[0] = layers[0] - i

                if len(layers) == 1:
                    x = outputs[i + (layers[0])]

                else:
                    if (layers[1]) > 0:
                        layers[1] = layers[1] - i

                    map1 = outputs[i + layers[0]]
                    map2 = outputs[i + layers[1]]

                    x = torch.cat((map1, map2), 1)

            elif  module_type == "shortcut":
                from_ = int(module["from"])
                x = outputs[i-1] + outputs[i+from_]

            elif module_type == 'yolo':       
                anchors = self.module_list[i][0].anchors

                # Dimension del input
                inp_dim = int (self.net_info["height"])

                # Numero de clases
                num_classes = int (module["classes"])

                # Aplicacion de la funcion predict_transform 
                x = x.data
                x = predict_transform(x, inp_dim, anchors, num_classes, CUDA)
                
                if not write:
                    detections = x
                    write = 1

                else:       
                    detections = torch.cat((detections, x), 1)

            outputs[i] = x

        return detections

    def load_weights(self, weightfile):
        # Abrir el archivo .weights
        fp = open(weightfile, "rb")

        # Los primeros 5 valores son informacion de la cabecera
        header = np.fromfile(fp, dtype = np.int32, count = 5)
        self.header = torch.from_numpy(header)
        self.seen = self.header[3]        
        
        weights = np.fromfile(fp, dtype = np.float32)

        # Solo tienen pesos las capas convolucional y batch_norm
        ptr = 0
        for i in range(len(self.module_list)):
            module_type = self.blocks[i + 1]["type"]

            # Si el bloque es convolucional, entonces carga pesos, si no, no
            if module_type == "convolutional":
                model = self.module_list[i]
                try:
                    batch_normalize = int(self.blocks[i+1]["batch_normalize"])
                except:
                    batch_normalize = 0

                conv = model[0]        

                # Se cargan los pesos de forma diferente dependiendo de si existe o no una capa batch_norm 
                if (batch_normalize):
                    bn = model[1]

                    # Numero de pesos
                    num_bn_biases = bn.bias.numel()

                    # Carga de los pesos
                    bn_biases = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
                    ptr += num_bn_biases

                    bn_weights = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr  += num_bn_biases

                    bn_running_mean = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr  += num_bn_biases

                    bn_running_var = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
                    ptr  += num_bn_biases

                    # Cambia las dimensiones de los pesos cargados a pesos del modelo
                    bn_biases = bn_biases.view_as(bn.bias.data)
                    bn_weights = bn_weights.view_as(bn.weight.data)
                    bn_running_mean = bn_running_mean.view_as(bn.running_mean)
                    bn_running_var = bn_running_var.view_as(bn.running_var)

                    # Copia los datos al modelo
                    bn.bias.data.copy_(bn_biases)
                    bn.weight.data.copy_(bn_weights)
                    bn.running_mean.copy_(bn_running_mean)
                    bn.running_var.copy_(bn_running_var)

                # Si no existe batch_norm, simplemente carga los bias de la capa convolucional
                else:
                    # Numero de bias
                    num_biases = conv.bias.numel()

                    # Carga de los pesos
                    conv_biases = torch.from_numpy(weights[ptr: ptr + num_biases])
                    ptr = ptr + num_biases

                    # Cambia las dimensiones de los pesos cargados a pesos del modelo
                    conv_biases = conv_biases.view_as(conv.bias.data)

                    # Copia los datos al modelo
                    conv.bias.data.copy_(conv_biases)

                # Carga de los pesos de la capa convolucional
                num_weights = conv.weight.numel()

                # Repetir el proceso de los otros pesos
                conv_weights = torch.from_numpy(weights[ptr:ptr+num_weights])
                ptr = ptr + num_weights

                conv_weights = conv_weights.view_as(conv.weight.data)
                conv.weight.data.copy_(conv_weights)

