# --------------------------------------------------------
# SiamMask
# Licensed under The MIT License
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
# --------------------------------------------------------
from os import path
import glob
import json
import bimpy
from tools.test import *
from enum import Enum
import pandas as pd
from pathlib import Path
import io

if __name__ != '__main__':
    exit()

parser = argparse.ArgumentParser(description='PyTorch Tracking Annotation')
parser.add_argument('--resume', default='', type=str, required=True,
                    metavar='PATH',help='path to latest checkpoint (default: none)')
parser.add_argument('--config', dest='config', default='config_davis.json',
                    help='hyper-parameter of SiamMask in json format')
parser.add_argument('--base_path_video', default='', help='datasets - video path')
parser.add_argument('--save_path', default=None, help='output directory')
args = parser.parse_args()

if args.save_path is None:
    args.save_path = str(Path(args.base_path_video).with_suffix('.csv'))

all_labels = ['HANDGUN', 'RIFLE']

# Init UI
ctx = bimpy.Context()
ctx.init(1200, 1200, "Hello")
img_view = None
img_data = None

# Load video.
video_file = cv2.VideoCapture(args.base_path_video)
video_file.set(cv2.CAP_PROP_POS_AVI_RATIO, 1)
video_len = int(video_file.get(cv2.CAP_PROP_POS_FRAMES))
video_file.set(cv2.CAP_PROP_POS_FRAMES, 0)
video_w = float(video_file.get(cv2.CAP_PROP_FRAME_WIDTH))
video_h = float(video_file.get(cv2.CAP_PROP_FRAME_HEIGHT))
video_name = str(Path(args.base_path_video).name)

print(video_w, video_h)

display_frame = -1
def set_video_frame(frame_idx=None):
    global display_frame
    global img_view
    global img_data

    if frame_idx is None:
        frame_idx = display_frame + 1
    if frame_idx < 0 or frame_idx >= video_len:
        return None
    if frame_idx != display_frame:
        video_file.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

    display_frame = frame_idx
    _, frame = video_file.read()
    img_data = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_view = bimpy.Image(img_data)

set_video_frame()


# Initialize torch and load model.
# Setup device
# device = torch.device('cpu')
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


# Setup Data
class Rect:
    def __init__(self, x1=0, y1=0, x2=0, y2=0, idx=-1, label='HANDGUN'):
        self.x = (x1 + x2) / 2
        self.y = (y1 + y2) / 2
        self.w = x2 - x1
        self.h = y2 - y1
        self.idx = idx
        self.label = label

    @property
    def pos(self):
        return np.array([self.x, self.y])

    @pos.setter
    def pos(self, value):
        self.x = value[0]
        self.y = value[1]

    @property
    def size(self):
        return np.array([self.w, self.h])

    @size.setter
    def size(self, value):
        self.w = value[0]
        self.h = value[1]

    @property
    def x1(self):
        return self.x - self.w / 2

    @property
    def y1(self):
        return self.y - self.h / 2

    @property
    def x2(self):
        return self.x + self.w / 2

    @property
    def y2(self):
        return self.y + self.h / 2


anot_idx = 0
rect_db = [{} for x in range(video_len)]
rect_table_of_contents = set()
rect_table_of_contents_sorted = []
annotation_frame = 0
active_annotations = {}
current_label_idx = bimpy.Int(0)

def is_rect_active(rect):
    return rect.idx in active_annotations

def update_rect_toc(frame_idx):
    global rect_table_of_contents_sorted
    if len(rect_db[frame_idx]) == 0:
        rect_table_of_contents.discard(frame_idx)
        if len(rect_db) > frame_idx + 1 and len(rect_db[frame_idx + 1]) != 0:
            rect_table_of_contents.add(frame_idx + 1)
            rect_table_of_contents_sorted = sorted(rect_table_of_contents)

    elif len(rect_db[frame_idx]) == 1 and frame_idx == 0:
        rect_table_of_contents.add(frame_idx)
        rect_table_of_contents_sorted = sorted(rect_table_of_contents)

    elif len(rect_db[frame_idx]) == 1 and len(rect_db[frame_idx - 1]) == 0:
        rect_table_of_contents.add(frame_idx)
        rect_table_of_contents_sorted = sorted(rect_table_of_contents)

class Annotation:
    def __init__(self, image, rect, frame_idx):
        global anot_idx
        self.is_done = False
        self.idx = anot_idx
        self.frame_idx = frame_idx
        anot_idx += 1
        rect.idx = self.idx
        rect.label = all_labels[current_label_idx.value]
        self.label = rect.label
        self.reinit(image, rect, frame_idx)

    def reinit(self, image, rect, frame_idx):
        self.siam_state = siamese_init(image, rect.pos, rect.size, siammask, cfg['hp'])
        self.frame_idx = frame_idx
        rect_db[frame_idx][self.idx] = rect
        update_rect_toc(frame_idx)

    def next_frame(self, image, frame_idx):
        self.siam_state = siamese_track(self.siam_state, image, mask_enable=False)
        r = Rect()
        r.idx = self.idx
        r.label = self.label
        r.pos = self.siam_state["target_pos"]
        r.size = self.siam_state["target_sz"]
        rect_db[frame_idx][self.idx] = r
        update_rect_toc(frame_idx)


def simulate_to_frame(frame_idx):
    global active_annotations

    old_frame = display_frame
    set_video_frame(frame_idx)

    if frame_idx != old_frame + 1:
        active_annotations = {}
        return

    to_delete = []
    for idx, a in active_annotations.items():
        if a.idx in rect_db[old_frame]:
            a.next_frame(img_data, frame_idx)
        else:
            to_delete.append(idx)

    for idx in to_delete:
        del active_annotations[idx]


# UI Tabs
tab_video_view = bimpy.Bool(True)
tab_rects = bimpy.Bool(True)
tab_keyframes = bimpy.Bool(True)

# Global state
is_autoplay = bimpy.Bool(False)
cursor_x1 = -10
cursor_y1 = -10
cursor_x2 = -10
cursor_y2 = -10
img_display_w = 0
img_display_h = 0
img_display_zero = bimpy.Vec2(0, 0)
is_placing_rect = True
end_rect_action = None

def ui_get_screen_rect(x1, y1, x2, y2):
    x1 = img_display_zero.x + x1 / video_w * img_display_w
    y1 = img_display_zero.y + y1 / video_h * img_display_h
    x2 = img_display_zero.x + x2 / video_w * img_display_w
    y2 = img_display_zero.y + y2 / video_h * img_display_h
    return x1, y1, x2, y2

def ui_draw_rect(x1, y1, x2, y2, color=0xFF00FFFF, label=''):
    x1, y1, x2, y2 = ui_get_screen_rect(x1, y1, x2, y2)
    bimpy.add_rect(bimpy.Vec2(x1, y1), bimpy.Vec2(x2, y2), color, thickness=4)
    if label:
        bimpy.add_text

def action_add_annotation():
    a = Annotation(img_data, Rect(cursor_x1, cursor_y1, cursor_x2, cursor_y2), display_frame)
    active_annotations[a.idx] = a

def save_data(path):
    lines = []
    for frame_idx, rects in enumerate(rect_db):
        for r in rects.values():
            lines.append('%s,%d,%d,%d,%d,%d,%d,%s\n' % (video_name, frame_idx, r.idx,
                                                        r.x1, r.y1, r.x2, r.y2, r.label))

    with io.open(path, 'w') as f:
        f.write("video_file,frame_idx,rect_idx,left,top,right,bottom,label\n")
        f.writelines(lines)

    print("saved data to %s" % path)

def load_data(path):
    df = None
    try:
        df = pd.read_csv(path)
    except Exception:
        print("failed to load data from %s" % path)
        return

    global rect_db
    global rect_table_of_contents
    global rect_table_of_contents_sorted
    rect_db = [{} for x in range(video_len)]
    rect_table_of_contents = set()
    rect_table_of_contents_sorted = []

    for index, row in df.iterrows():
        rect_db[row['frame_idx']][row['rect_idx']] = Rect(row['left'], row['top'], row['right'], row['bottom'], row['rect_idx'], row['label'])
        update_rect_toc(row['frame_idx'])

    print("loaded data from %s" % path)

end_rect_action = action_add_annotation

while(not ctx.should_close()):
    ctx.new_frame()
    bimpy.show_demo_window()  # Widget reference

    if bimpy.begin_main_menu_bar():
        if bimpy.menu_item('Save', ''):
            save_data(args.save_path)
        if bimpy.menu_item('Load', ''):
            load_data(args.save_path)
        bimpy.end_main_menu_bar()  # According to bimpy docs, this is a special case where end is called inside the if.

    if bimpy.begin("Video", opened=tab_video_view):
        is_placing_rect = True
        s = bimpy.text(args.base_path_video)
        b_i = bimpy.Int(display_frame)
        bimpy.slider_int("Frame", b_i, 0, video_len, "%d")

        if bimpy.button(" < Prev (z) ") or bimpy.is_key_released(ord('Z')):
            b_i.value -= 1
        bimpy.same_line()
        bimpy.checkbox("Autoplay (c to stop)", is_autoplay)
        if bimpy.is_key_down(ord('C')):
            is_autoplay.value = False
        bimpy.same_line()
        if bimpy.button(" Next > (x) ") or bimpy.is_key_released(ord('X')) or is_autoplay.value:
            b_i.value += 1

        if display_frame != b_i.value:
            simulate_to_frame(b_i.value)

        bimpy.combo('Label used for annotation', current_label_idx, all_labels)

        img_display_w = bimpy.get_window_content_region_width()
        img_display_h = img_display_w * video_h / video_w

        # Get the zero coordinate of the actual image rendering
        img_display_zero = bimpy.get_cursor_pos() + bimpy.get_window_pos()
        bimpy.image(img_view, bimpy.Vec2(img_display_w, img_display_h))

        # Logic for deleting rects. Click an active rect to make inactive, click to delete.
        delete_rect = None
        for key, r in rect_db[display_frame].items():
            color = 0xFF00FF00 if is_rect_active(r) else 0xFFFFFF00
            x1, y1, x2, y2 = ui_get_screen_rect(r.x1, r.y1, r.x2, r.y2)
            if bimpy.is_mouse_hovering_rect(bimpy.Vec2(x1, y1), bimpy.Vec2(x2, y2), 0):
                color = 0xFFFFFF00 if is_rect_active(r) else 0xFF0000FF
                is_placing_rect = False
                bimpy.begin_tooltip()
                bimpy.text(r.label)
                bimpy.text('Idx: %d' % r.idx)
                bimpy.end_tooltip()

                if bimpy.is_mouse_released(0):
                    delete_rect = key
            ui_draw_rect(r.x1, r.y1, r.x2, r.y2, color)

        if delete_rect is not None:
            if is_rect_active(rect_db[display_frame][delete_rect]):
                del active_annotations[delete_rect]
            else:
                del rect_db[display_frame][delete_rect]
                update_rect_toc(display_frame)

        if bimpy.is_item_hovered(0):
            # Code here mostly computes coordinates within the image and draws active selection
            mouse = bimpy.get_mouse_pos() - img_display_zero
            mouse_img_x = mouse.x / img_display_w * video_w
            mouse_img_y = mouse.y / img_display_h * video_h
            mouse_img_x = max(min(video_w, mouse_img_x), 0)
            mouse_img_y = max(min(video_h, mouse_img_y), 0)

            if is_placing_rect:
                if cursor_x1 >= 0:
                    ui_draw_rect(cursor_x1, cursor_y1, mouse_img_x, mouse_img_y)

                if bimpy.is_mouse_released(0):
                    if cursor_x1 >= 0:
                        cursor_x1, cursor_x2 = min(cursor_x1, mouse_img_x), max(cursor_x1, mouse_img_x)
                        cursor_y1, cursor_y2 = min(cursor_y1, mouse_img_y), max(cursor_y1, mouse_img_y)
                        if end_rect_action is not None:
                            end_rect_action()
                        cursor_x1 = cursor_x2 = cursor_y1 = cursor_y2 = -10
                    else:
                        cursor_x1 = mouse_img_x
                        cursor_y1 = mouse_img_y

                if bimpy.is_mouse_released(1):
                    cursor_x1 = cursor_x2 = cursor_y1 = cursor_y2 = -10

            a = bimpy.Vec2(img_display_zero.x + mouse.x, img_display_zero.y)
            b = bimpy.Vec2(img_display_zero.x + mouse.x, img_display_zero.y + img_display_h)
            bimpy.add_line(a, b, 0x99FF00FF, 2)
            a = bimpy.Vec2(img_display_zero.x, img_display_zero.y + mouse.y)
            b = bimpy.Vec2(img_display_zero.x + img_display_w, img_display_zero.y + mouse.y)
            bimpy.add_line(a, b, 0x99FF00FF, 2)

            bimpy.begin_tooltip()
            bimpy.text('%d, %d' % (mouse_img_x, mouse_img_y))
            bimpy.end_tooltip()
    bimpy.end()


    if bimpy.begin("Rects in frame", opened=tab_rects):
        for r in rect_db[display_frame].values():
            bimpy.text("Rect %d" % r.idx)
            bimpy.text("Label: %s" % r.label)
            bimpy.text("Pos  : %d, %d" % (r.x, r.y))
            bimpy.text("Size : %d, %d" % (r.w, r.h))
            bimpy.separator()
    bimpy.end()


    if bimpy.begin("Frames where rects begin", opened=tab_keyframes):
        for frame_idx in rect_table_of_contents_sorted:
            if bimpy.button("Frame %d##goto" % frame_idx):
                simulate_to_frame(frame_idx)
    bimpy.end()


    ctx.render()
