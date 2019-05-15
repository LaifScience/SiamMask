# --------------------------------------------------------
# SiamMask
# Licensed under The MIT License
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
# --------------------------------------------------------
from os import path
import glob
import json
from tools.test import *

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
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True

    # Setup Model
    cfg = load_config(args)
    from custom import Custom
    siammask = Custom(anchors=cfg['anchors'])
    if args.resume:
        assert isfile(args.resume), '{} is not a valid file'.format(args.resume)
        siammask = load_pretrain(siammask, args.resume)

    siammask.eval().to(device)

    
    cv2.namedWindow("SiamMask", cv2.WND_PROP_FULLSCREEN)
    # cv2.setWindowProperty("SiamMask", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # Select ROI
    def select_region(im):
        return(cv2.selectROI('SiamMask', im, False, False))
    

    def get_state(x, y, w, h):
        target_pos = np.array([x + w / 2, y + h / 2])
        target_sz = np.array([w, h])
        state = siamese_init(im, target_pos, target_sz, siammask, cfg['hp'])
        return(state)

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
            if cv2.waitKey(0) == ord('s'):  # init
                x, y, w, h = select_region(im)
                print("selected region in frame " + str(f))
                print((x, y, w, h))
                state = get_state(x, y, w, h)
                selected = True
                bboxes[f] = (int(x), int(y), int(x + w), int(y + h))
            elif cv2.waitKey(0) == ord('n'):
                print("skip frame " + str(f))
            elif cv2.waitKey(0) == ord('q'):
                print("exit at frame " + str(f))                
                break
                #cv2.imshow('SiamMask', im)
        elif selected:  # tracking
            state = siamese_track(state, im, mask_enable=True, refine_enable=True)  # track
            location = state['ploygon']            
            mask = state['mask'] > state['p'].seg_thr

            im[:, :, 2] = (mask > 0) * 255 + (mask == 0) * im[:, :, 2]
            #print(location)
            
            imcopy = im.copy()
            x = int(min([wi[0] for wi in location]))
            y = int(min([wi[1] for wi in location]))
            xw = int(max([wi[0] for wi in location]))
            yh = int(max([wi[1] for wi in location]))
            
            print((x,y,xw,yh))
            cv2.rectangle(imcopy, (x, y), (xw, yh), (0, 255, 0), 2)
            #cv2.polylines(imcopy, [np.int0(location).reshape((-1, 1, 2))], True, (0, 255, 0), 3)
            cv2.imshow('SiamMask', imcopy)

            if cv2.waitKey(0) == ord('r'):
                print("rejected proposal for frame " + str(f))
                cv2.imshow('SiamMask', im)
                x, y, w, h = select_region(im)
                state = get_state(x, y, w, h)
                bboxes[f] = (int(x), int(y), int(x + w), int(y + h))
                
            elif cv2.waitKey(0) == ord('a'):
                print("accepted proposal for frame " + str(f))                
                bboxes[f] = (x, y, xw, yh)
            elif cv2.waitKey(0) == ord('n'):
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
