# --------------------------------------------------------
# SiamMask
# Licensed under The MIT License
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
# --------------------------------------------------------
from os import path
import glob
import json
from tools.test import *
from siam_face import *

parser = argparse.ArgumentParser(description='PyTorch Tracking Annotation')

parser.add_argument('--resume', default='', type=str, required=True,
                    metavar='PATH',help='path to latest checkpoint (default: none)')
parser.add_argument('--config', dest='config', default='config_davis.json',
                    help='hyper-parameter of SiamMask in json format')
parser.add_argument('--base_path_images', default='', help='datasets - images directory')
parser.add_argument('--base_path_video', default='', help='datasets - video path')
parser.add_argument('--save_dir', default='', help='output directory')
args = parser.parse_args()

if __name__ == '__main__':
    cfg = load_config(args)
    siam = SiamFaceTracker(cfg, model=args.resume)

    cv2.namedWindow("SiamMask", cv2.WND_PROP_FULLSCREEN)
    # cv2.setWindowProperty("SiamMask", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # Select ROI
    def select_region(im):
        return(cv2.selectROI('SiamMask', im, False, False))    

    def gen_from_images(ext_pattern):
        for imf in sorted(glob.glob(join(args.base_path_images, ext_pattern))): # '*.jp*'
            yield(cv2.imread(imf))

    def gen_from_video():
        vidcap = cv2.VideoCapture(args.base_path_video)
        while vidcap.isOpened():        
            has_frames, image = vidcap.read()
            if has_frames:
                yield(image)
            else:
                vidcap.release()

    data_gen = gen_from_video() if args.base_path_video else gen_from_images('*.jp*')

    toc = 0
    selected = False
    bboxes = {}   

    for f, im in enumerate(data_gen):
        print("processing frame " + str(f))
        tic = cv2.getTickCount()
        if not selected:
            cv2.imshow('SiamMask', im)
            key = cv2.waitKey(0)
            if key == ord('s'):  # init
                x, y, w, h = select_region(im)
                print("selected region in frame " + str(f))
                print((x, y, w, h))
                siam.set_state(im, (x, y, w, h))
                selected = True
                bboxes[f] = (int(x), int(y), int(x + w), int(y + h))
            elif key == ord('n'):
                print("skip frame " + str(f))
            elif key == ord('q'):
                print("exit at frame " + str(f))  
                break
                
        elif selected:  # tracking
            imcopy = im.copy() # for the case when the proposed box is rejected           

            (x, y, xw, yh) = siam.track_face(im)            
            
            cv2.rectangle(imcopy, (x, y), (xw, yh), (0, 255, 0), 2)
            cv2.imshow('SiamMask', imcopy)

            key = cv2.waitKey(0)
            
            if key == ord('r'):
                print("rejected proposal for frame " + str(f))
                cv2.imshow('SiamMask', im)
                x, y, w, h = select_region(im)
                siam.set_state(im, (x, y, w, h))
                bboxes[f] = (int(x), int(y), int(x + w), int(y + h))
                
            elif key == ord('a'):
                print("accepted proposal for frame " + str(f))                
                bboxes[f] = (x, y, xw, yh)
            elif key == ord('n'):
                print("reject and next frame in frame " + str(f))
                selected = False           
            

        toc += cv2.getTickCount() - tic
    toc /= cv2.getTickFrequency()
    fps = f / toc
    print('SiamMask Time: {:02.1f}s Speed: {:3.1f}fps (with visulization!)'.format(toc, fps))
    
    def get_save_path():
        from_path = args.base_path_video if args.base_path_video else args.base_path_images
        return(join(args.save_dir, path.splitext(path.basename(from_path))[0] + ".json"))

    with open(get_save_path(), 'w') as outfile:  
        json.dump(bboxes, outfile)