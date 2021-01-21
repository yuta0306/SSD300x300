import torch
import torch.nn as nn
from torch.autograd import Function
import numpy as np
from functions import decode, nms

class Detect(Function):
    def forward(self, output, num_classes, 
                top_k=200, variance=[0.1,0.2], 
                conf_thresh=0.01, nms_thresh=0.45):    
        loc_data, conf_data, prior_data = output[0], output[1], output[2]
        softmax = nn.Softmax(dim=-1)
        conf_data = softmax(conf_data)       
        num = loc_data.size(0)  
        output = torch.zeros(num, num_classes, top_k, 5)
        conf_preds = conf_data.transpose(2, 1)
        # Decode predictions into bboxes.
        for i in range(num):
            decoded_boxes = decode(loc_data[i], prior_data, variance)
            conf_scores = conf_preds[i].clone()
            for cl in range(1, num_classes): 
                c_mask = conf_scores[cl].gt(conf_thresh)
                scores = conf_scores[cl][c_mask]
                if scores.size(0) == 0:
                    continue
                l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)
                boxes = decoded_boxes[l_mask].view(-1, 4)
                ids, count = nms(boxes, scores, nms_thresh, top_k)
                output[i, cl, :count] = \
                    torch.cat((scores[ids[:count]].unsqueeze(1),
                               boxes[ids[:count]]), 1)
        return output