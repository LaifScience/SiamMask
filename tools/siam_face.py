from tools.test import *
from custom import Custom

torch.backends.cudnn.benchmark = True # better runtime execution - the input size will not vary

class SiamFaceTracker(object):
    def __init__(self, cfg, model="SiamMask_DAVIS.pth"):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.cfg = cfg
        self.siammask = Custom(anchors=cfg['anchors'])
        self.siammask = load_pretrain(self.siammask, model)
        self.siammask.eval().to(device)
        self.state = None

    def set_state(self, im, loc): # we can adapt this input to match the object detector bbox output
        (x, y, w, h) = loc
        target_pos = np.array([x + w / 2, y + h / 2])
        target_sz = np.array([w, h])
        self.state = siamese_init(im, target_pos, target_sz, self.siammask, self.cfg['hp'])        

    def track_face(self, im):
        self.state = siamese_track(self.state, im, mask_enable=False)  
        
        [x, y] = self.state["target_pos"]
        [w, h] = self.state["target_sz"]

        x = int(x - w / 2)
        y = int(y - h / 2)
        xw = int(x + w)
        yh = int(y + h)

        return((x, y, xw, yh))
        


