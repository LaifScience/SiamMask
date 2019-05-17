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
from citygroup_face_recognition import face_recognition_api

parser = argparse.ArgumentParser(description='PyTorch Tracking Annotation')

parser.add_argument('--resume', default='', type=str, required=True,
                    metavar='PATH',help='path to latest checkpoint (default: none)')
parser.add_argument('--config', dest='config', default='config_davis.json',
                    help='hyper-parameter of SiamMask in json format')
parser.add_argument('--base_path_images', default='', help='datasets - images directory')
parser.add_argument('--base_path_video', default='', help='datasets - video path')
parser.add_argument('--save_dir', default='', help='output directory')
args = parser.parse_args()

def bb_intersection_over_union(boxA, boxB):
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

if __name__ == '__main__':
    cfg = load_config(args)
    siam = SiamFaceTracker(cfg, model=args.resume)
    face_recognition_api.init_api("faceboxes", "facenet", "true")

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
    out = cv2.VideoWriter('outpy.mp4', cv2.VideoWriter_fourcc(*'XVID'), 25, (1920,1080))

    toc = 0
    selected = False
    bboxes = {}   

    colors = [(0, 255,0), (0,0,255), (0, 255, 255), (255,0, 255), ]
    siams = []
    last_bboxes = []
    for f, im in enumerate(data_gen):
        print("processing frame " + str(f))
        tic = cv2.getTickCount()
        # if not selected:
        #     cv2.imshow('SiamMask', im)
        #     key = cv2.waitKey(0)
        #     if key == ord('s'):  # init
        #         x, y, w, h = select_region(im)
        #         print("selected region in frame " + str(f))
        #         print((x, y, w, h))
        #         siam.set_state(im, (x, y, w, h))
        #
        #         x, y, w, h = select_region(im)
        #         print("selected region in frame " + str(f))
        #         print((x, y, w, h))
        #         siam2.set_state(im, (x, y, w, h))
        #         siams = [siam, siam2]
        #         selected = True
        #         bboxes[f] = (int(x), int(y), int(x + w), int(y + h))
        #     elif key == ord('n'):
        #         print("skip frame " + str(f))
        #     elif key == ord('q'):
        #         print("exit at frame " + str(f))
        #         out.release()
        #         break
        #
        # elif selected:  # tracking
        if True:
            imcopy = im.copy() # for the case when the proposed box is rejected           

            results = face_recognition_api.detect_and_classify_image(im)
            for r in results:
                left, top = r["left"], r["top"]
                right, bottom = r["right"], r["bottom"]
                cv2.putText(imcopy, r["label"],  (left, top-15), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, 125, 1, cv2.LINE_AA)
                cv2.rectangle(imcopy, (left, top), (right, bottom), (255,0,0), 2)

            siam_preds = []
            for i, el in enumerate(siams):
                label, siam = el
                (x, y, xw, yh) = siam.track_face(im)
                siam_preds.append((x, y, xw, yh))
                cv2.rectangle(imcopy, (x, y), (xw, yh), colors[i], 2)

            cp_bboxes = siam_preds[:]
            new_siams = []
            new_predictions = []
            for r in results:
                left, top = r["left"], r["top"]
                right, bottom = r["right"], r["bottom"]
                label = r["label"]
                maxInd, maxValue = -1, -1
                for i, prev_box in enumerate(cp_bboxes):
                    iou = bb_intersection_over_union((left, top, right, bottom), prev_box)
                    if iou > maxValue:
                        maxInd = i
                        maxValue = iou
                if maxValue > 0.5:
                    bbox = cp_bboxes.pop(maxInd)
                    label, siam = siams.pop(maxInd)
                    new_predictions.append((label, bbox))
                    new_siams.append((label, siam))
                else:
                    new_predictions.append((label, (left, top, right, bottom)))
                    siam = SiamFaceTracker(cfg, model=args.resume)
                    siam.set_state(im, (left, top, right - left, bottom - top))
                    new_siams.append((label, siam))

            siams = new_siams

            for label, bbox in new_predictions:
                left, top, right, bottom = bbox
                cv2.rectangle(imcopy, (left, top), (right, bottom), (255,255,255), 2)
                cv2.putText(imcopy, label,  (left, top-15), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255,255,255), 1, cv2.LINE_AA)


            cv2.imshow('SiamMask', imcopy)
            out.write(imcopy)
            key = cv2.waitKey(10)
            
            if key == ord('r'):
                print("rejected proposal for frame " + str(f))
                cv2.imshow('SiamMask', im)
                x, y, w, h = select_region(im)
                siam.set_state(im, (x, y, w, h))
                bboxes[f] = (int(x), int(y), int(x + w), int(y + h))
                
            elif key == ord('q'):
                out.release()
                break
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
