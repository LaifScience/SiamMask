import uuid

from collections import namedtuple
from tools.test import *
from custom import Custom

torch.backends.cudnn.benchmark = True # better runtime execution - the input size will not vary


def bb_iou(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
 
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
 
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
 
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)
 
	# return the intersection over union value
	return iou

def scale_bbox(bbox, factor = 2):
    ini_w = bbox["right"] - bbox["left"]
    ini_h = bbox["bottom"] - bbox["top"]
    new_w = ini_w * factor
    new_h = ini_h * factor
    pos_x = bbox["left"] + ini_w / 2
    pos_y = bbox["top"] + ini_h / 2

    new_left = max(pos_x - new_w / 2, 0)
    new_top = max(pos_y - new_h / 2, 0)

    new_bbox = {
        "left": new_left,
        "top": new_top,
        "right": new_left + new_w, # no max width check
        "bottom": new_top + new_h, # no max height check
        "label": bbox["label"]
    }
    
    return new_bbox


TrackingResult = namedtuple("TrackingResult", "id bbox")

class SiamFaceTracker(object):
    def __init__(self, cfg, min_iou = 0.3, model="SiamMask_DAVIS.pth"):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # to do - check if this should be instantiated for multiple SiamMasks objects
        self.internal_id = uuid.uuid4() # object identity can also work
        
        self.cfg = cfg
        self.siammask = Custom(anchors=cfg['anchors'])
        self.siammask = load_pretrain(self.siammask, model)
        self.siammask.eval().to(device)
        self.state = None
        
        self.prev_bbox = None
        self.is_recruited = False
        self.class_id = None
        self.min_iou = min_iou
        self.iou = None
        self.frames_elapsed_from_set_state = 0
        self.last_tracking_result = None


    def set_state(self, im, detection): # we can adapt this input to match the object detector bbox output
        scaled_bbox = scale_bbox(detection)
        x = scaled_bbox["left"]
        y = scaled_bbox["top"]
        w = abs(x - scaled_bbox["right"])
        h = abs(y - scaled_bbox["bottom"])        
        target_pos = np.array([x + w / 2, y + h / 2])
        target_sz = np.array([w, h])
        self.state = siamese_init(im, target_pos, target_sz, self.siammask, self.cfg['hp'])     
        self.class_id = scaled_bbox["label"] 
        self.is_recruited = True  
        self.frames_elapsed_from_set_state = 0

    def update_state(self, im, detection):
        scaled_bbox = scale_bbox(detection)
        x = scaled_bbox["left"]
        y = scaled_bbox["top"]
        w = abs(x - scaled_bbox["right"])
        h = abs(y - scaled_bbox["bottom"])        
        target_pos = np.array([x + w / 2, y + h / 2])
        target_sz = np.array([w, h])
        self.state = siamese_init(im, target_pos, target_sz, self.siammask, self.cfg['hp']) 
        self.frames_elapsed_from_set_state = 0

    def invalidate(self):
        self.class_id = None
        self.is_recruited = False
        self.frames_elapsed_from_set_state = 0
        self.last_tracking_result = None
        self.prev_bbox = None
        self.iou = None

    def track_face(self, im):        
        if not self.is_recruited:
            return(None)

        self.state = siamese_track(self.state, im, mask_enable=False)  
        
        [x, y] = self.state["target_pos"]
        [w, h] = self.state["target_sz"]

        x = int(x - w / 2)
        y = int(y - h / 2)
        xw = int(x + w)
        yh = int(y + h)

        c_bbox = (x, y, xw, yh)

        if self.prev_bbox:
            self.iou = bb_iou(self.prev_bbbox, c_bbox)

        self.prev_bbox = c_bbox
        self.frames_elapsed_from_set_state += 1

        if self.iou:
            if iou > self.min_iou:
                self.last_tracking_result = (TrackingResult(self.class_id, c_bbox))                
            else:
                self.prev_bbox = None
                self.iou = None
                self.is_recruited = False
                self.last_tracking_result = None                
        else:
            self.last_tracking_result = (TrackingResult(self.class_id, c_bbox))


class MultiTracker(object):
    def __init__(self, fps, siams):
        self.siams = siams 
        self.fps = fps        
    
    def iou_siam(self, siam_x, siam_y):
        if siam_x.internal_id == siam_y.internal_id:
            return(False)
        if not (siam_x.last_tracking_result and siam_y.last_tracking_result):
            return(False)
        iou = bb_iou(siam_x.last_tracking_result.bbox, siam_y.last_tracking_result.bbox)
        return(iou > 0)
    
    def make_result_dict(self, siam):
        bbox = siam.last_tracking_result.bbox
        label = siam.last_tracking_result.id
        res_dict = {"left" : bbox[0], "top" : bbox[1], "right" : bbox[2], "bottom" : bbox[3], "label" : label }
        return(res_dict)
    
    def process_frame(self, detections,  frame):
        results = []
        
        for siam in self.siams:
            siam.track_face(frame)
        
        # invalidate intersected siam boxes
        for siam_x in self.siams:
            if not siam_x.is_recruited:
                continue                        
            for siam_y in self.siams:
                if self.iou_siam(siam_x, siam_y):
                    siam_x.invalidate()
                    siam_y.invalidate()
        
        detection_wo_siam_overlap = []
        
        for detection in detections:
            overlap = False
            for siam in self.siams:
                if not (siam.is_recruited and siam.last_tracking_result): # some redundant logic in some places, but better safe than sorry when hurry
                    continue            
            
                face_box = (detection["left"], detection["top"], detection["right"], detection["bottom"])
                iou = bb_iou(face_box, siam.last_tracking_result.bbox)                

                if iou > 0 and siam.frames_elapsed_from_set_state > self.fps:
                    siam.update_state(frame, detection)

                if siam.frames_elapsed_from_set_state > self.fps * 10:
                    siam.invalidate()

                if iou > 0:
                    overlap = True
                    break                    
            
            if not overlap:
                detection_wo_siam_overlap.append(detection)

        for siam in self.siams:
            if siam.is_recruited and siam.last_tracking_result:
                results.append(self.make_result_dict(siam))

        for detection in detection_wo_siam_overlap:
            next_recruit = next((siam for siam in self.siams if not siam.is_recruited), None)

            if next_recruit:
                next_recruit.set_state(frame, detection)

            results.append(scale_bbox(detection))

        return(results)

        

        

        

                
                



                


        



    
    