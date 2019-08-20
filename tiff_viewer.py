import tkinter as tk
from tkinter import filedialog
from viewer.patch_loader import TIFFPatchLoader
from viewer.annotation import BBoxList
from viewer.predictor import RetinaNetPredictor
import numpy as np
import pandas as pd
from osgeo import gdal
from PIL import Image, ImageTk


class TIFFViewer(tk.Tk):
    def __init__(self):
        super().__init__()

        self.wm_title('TIFF Viewer')

        self.settings_menu = None

        self.INITIAL_PATH = "/home/kruglov/projects/cit/"

        self.predictor = RetinaNetPredictor()

        self.filepath = None
        self.tiff = None
        self.img = None

        self.filter_file_path = None

        self.scalefactor = None

        self.bbox_list = None

        self.x1 = 0
        self.y1 = 0
        self.x0 = None
        self.y0 = None
        self.x2 = None
        self.y2 = None
        self.width = None
        self.height = None

        self.x_delta = 0.
        self.y_delta = 0.

        self.click_x = None
        self.click_y = None

        self.bbox_state = None
        self.start_bbox = None
        self.frame_coords = None
        self.frame = None

        self.show_frames = False

        self.zoom_state = None
        self.zoom_scalefactor = None

        self.area_state = None
        self.start_area = None

        # maximizing main window
        self.attributes('-zoomed', True)
        w, h = self.winfo_screenwidth(), self.winfo_screenheight()
        self.geometry("%dx%d+0+0" % (w, h))

        self.canvas = ResizingCanvas(self)  # width= width + 10, height= height + 10
        self.canvas.bind('<Button-1>', self.click_handler)

        self.canvas.pack(fill=tk.BOTH, anchor=tk.NW)

        # creating a top level menu
        self.menu_bar = tk.Menu(self)
        self.config(menu=self.menu_bar)

        self.menu_names = []

        # file submenu
        self.file_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label='File', menu=self.file_menu)
        self.menu_names.append('File')
        self.file_menu.add_command(label="Open Image...", underline=1, command=self.file_open, accelerator="Ctrl+O")
        self.file_menu.add_command(label="Open markup CSV...", command=self.csv_open)
        self.file_menu.add_command(label="Save markup CSV...", underline=1, command=self.csv_save)
        self.file_menu.add_command(label="Load object detection model...", underline=1, command=self.model_open)
        self.file_menu.add_command(label="Load distance filter...", command=self.filter_open)
        self.file_menu.add_command(label="Settings...", command=self.show_settings_menu)
        self.file_menu.add_command(label="Exit", command=self.quit)

        # view submenu
        self.view_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label='View', menu=self.view_menu)
        self.menu_names.append('View')
        self.view_menu.add_command(label="Overview", command=self.load_overview, accelerator="Ctrl+0")
        self.view_menu.add_command(label="Zoom 1/3", command=lambda: self.zoom_switch_on(0.3), accelerator="Ctrl+3")
        self.view_menu.add_command(label="Zoom 1/2", command=lambda: self.zoom_switch_on(0.5), accelerator="Ctrl+2")
        self.view_menu.add_command(label="Zoom 1/1", command=self.zoom_switch_on, accelerator="Ctrl+1")

        self.show_hide_frames = tk.IntVar()
        self.view_menu.add_radiobutton(label="Show frames", variable=self.show_hide_frames, value=1)
        self.view_menu.add_radiobutton(label="Hide frames", variable=self.show_hide_frames, value=0)

        self.show_hide_bboxes = tk.IntVar()
        self.view_menu.add_radiobutton(label="Show bounding boxes", variable=self.show_hide_bboxes, value=1)
        self.view_menu.add_radiobutton(label="Hide bounding boxes", variable=self.show_hide_bboxes, value=0)

        # markup submenu
        self.annotation_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label='Annotation', menu=self.annotation_menu)
        self.menu_names.append('Annotation')
        self.annotation_menu.add_command(label="Add label box...", command=self.add_bbox_switch_on,
                                         accelerator="B")

        # object detection submenu
        self.od_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label='Object detection', menu=self.od_menu)
        self.menu_names.append('Object detection')
        self.od_menu.add_command(label='Detect objects in current frame...', command=self.detect_objects,
                                 accelerator="Ctrl-D")
        self.od_menu.add_command(label='Detect objects in area...', command=self.select_area_switch_on)
        self.od_menu.add_command(label='Apply distance filter...', command=self.apply_distance_filter)

        # self.status = StatusBar(self)
        # self.status.pack(side= tk.BOTTOM, fill= tk.X)

        self.main()

    def show_patch(self, x0=None, y0=None, x1=0, y1=0, x2=None, y2=None):
        self.log_msg("Patch size: {}x{}.\nCanvas size: {}x{}.".format(self.width, self.height, self.canvas.width,
                                                                     self.canvas.height), level=0)

        self.canvas.delete('frame')
        self.canvas.delete('bbox')
        self.canvas.delete('patch')

        self.canvas.create_image(self.canvas.width / 2, self.canvas.height / 2, anchor=tk.CENTER, image=self.img, tag='patch')

        self.set_patch_xy(x0=x0, y0=y0, x1=x1, y1=y1, x2=x2, y2=y2)
        if self.show_hide_frames.get() == 1:
            self.show_frames_for_patch()

        if self.show_hide_bboxes.get() == 1:
            self.show_bboxes_for_patch()
        # else:
        #    print(self.show_hide_frames.get())

    def show_frames_for_patch(self):
        frames = self.bbox_list.get_frames_in_patch(self.x1, self.y1, self.x2, self.y2)
        for f in frames:
            f = f * self.scalefactor
            f = f + np.array([self.x_delta, self.y_delta] * 2)
            self.canvas.create_rectangle(*f, tag='frame', width=2, outline='blue')

    def show_bboxes_for_patch(self):
        bboxes = self.bbox_list.get_bboxes_in_patch(self.x1, self.y1, self.x2, self.y2)
        for b in bboxes:
            b = b * self.scalefactor
            b = b + np.array([self.x_delta, self.y_delta] * 2)
            self.canvas.create_rectangle(*b, tag='bbox', width=2, outline='green')

    def set_patch_xy(self, x0=None, y0=None, x1=0, y1=0, x2=None, y2=None):
        self.x_delta = (self.canvas.width - self.width) / 2
        self.y_delta = (self.canvas.height - self.height) / 2

        self.log_msg(['x_delta:', self.x_delta, 'y_delta:', self.y_delta], level=0)

        self.x0 = x0 if x0 is not None else x1 + int((self.width / 2) / self.scalefactor)
        self.y0 = y0 if y0 is not None else y1 + int((self.height / 2) / self.scalefactor)

        self.x1 = x1
        self.y1 = y1

        self.x2 = x2 if x2 is not None else x1 + int(self.width / self.scalefactor)
        self.y2 = y2 if y2 is not None else y1 + int(self.height / self.scalefactor)

        self.log_msg(['self.x0, self.y0, self.x1, self.y1, self.x2, self.y2',
                      self.x0, self.y0, self.x1, self.y1, self.x2, self.y2], level=0)

    def load_tiff(self, filepath):
        self.tiff = gdal.Open(filepath, gdal.GA_ReadOnly)
        # print(self.canvas.width, self.canvas.height)
        self.load_overview()
        self.bbox_list = BBoxList(filepath)

    def load_overview(self, event=None):
        self.img, self.width, self.height, self.scalefactor = TIFFPatchLoader(self.tiff).get_overview(self.canvas.width,
                                                                                                      self.canvas.height)
        self.img = Image.fromarray(self.img)
        self.img = ImageTk.PhotoImage(self.img)
        self.show_patch()

    def load_zoom(self, event=None):
        x0 = self.x0 + int((self.click_x - self.canvas.width / 2) / self.scalefactor)
        y0 = self.y0 + int((self.click_y - self.canvas.height / 2) / self.scalefactor)

        self.log_msg(['scalefactor', self.scalefactor], level=0)

        x1 = x0 - int(self.canvas.width / 2 / self.zoom_scalefactor)
        y1 = y0 - int(self.canvas.height / 2 / self.zoom_scalefactor)
        x2 = x1 + int(self.canvas.width / self.zoom_scalefactor)
        y2 = y1 + int(self.canvas.height / self.zoom_scalefactor)
        buffer_x = self.canvas.width
        buffer_y = self.canvas.height

        self.log_msg(['get_patch', x1, y1, x2, y2, buffer_x, buffer_y])

        self.img = TIFFPatchLoader(self.tiff).get_patch(x1, y1, x2, y2, buffer_x, buffer_y)
        self.img = Image.fromarray(self.img)
        self.img = ImageTk.PhotoImage(self.img)

        self.width = self.canvas.width
        self.height = self.canvas.height

        self.scalefactor = self.zoom_scalefactor
        self.show_patch(x0=x0, y0=y0, x1=x1, y1=y1, x2=x2, y2=y2)

    def zoom_switch_on(self, zoom_scalefactor=1., event=None):
        self.zoom_state = True
        self.zoom_scalefactor = zoom_scalefactor
        self.config(cursor='target')

    def zoom_switch_off(self, event=None):
        self.zoom_state = False
        self.config(cursor='arrow')

    def add_bbox_switch_on(self, event=None):
        self.bbox_state = True
        self.config(cursor='pencil')
        self.frame_coords = {
            'xoff': self.x0 - 400,
            'yoff': self.y0 - 400,
            'width': 800,
            'height': 800
        }
        self.bbox_list.set_frame(**self.frame_coords)

        self.frame = self.canvas.create_rectangle(*(
            int(self.canvas.width / 2) - 400,  # x1
            int(self.canvas.height / 2) - 400,  # y1
            int(self.canvas.width / 2) + 400,  # x2
            int(self.canvas.height / 2) + 400  # y2
        ), tag='frame', width=2, outline='blue')

    def add_bbox_switch_off(self, event=None):
        bbox = self.click_x, self.click_y, event.x, event.y
        self.canvas.create_rectangle(*bbox, tag='bbox1', width=2, outline='green',
                                     activeoutline='yellow')
        self.bbox_list.add_bbox(
            x1=int(self.click_x - self.width / 2 + 400),
            y1=int(self.click_y - self.height / 2 + 400),
            x2=int(event.x - self.width / 2 + 400),
            y2=int(event.y - self.height / 2 + 400),
            class_name='car'
        )
        self.bbox_state = False
        self.start_bbox = False
        self.config(cursor='arrow')
        self.frame_coords = None
        self.frame = None

    def click_handler(self, event=None):

        if self.bbox_state:
            if not self.start_bbox:  # start bbox
                self.save_click_xy(event=event)
                self.start_bbox = True
            else:  # end bbox
                self.add_bbox_switch_off(event=event)

        if self.zoom_state:
            self.save_click_xy(event=event)
            self.load_zoom()
            self.zoom_switch_off()

        if self.area_state:
            if not self.start_area:
                self.save_click_xy(event=event)
                self.start_area = True
            else:
                self.select_area_switch_off(event=event)
                self.start_area = None

    def esc_handler(self, event=None):

        if self.bbox_state:
            self.add_bbox_switch_off()

    def save_click_xy(self, event=None):
        self.log_msg(['Click coordinates: ', event.x, event.y], level=0)

        self.click_x = event.x
        self.click_y = event.y

    def file_open(self, event=None, filepath=None):
        if filepath is None:
            filepath = filedialog.askopenfilename(initialdir=self.INITIAL_PATH,
                                                  title="Select file to open...",
                                                  filetypes=[('GeoTIFF files', '*.tif')])

            if bool(filepath):
                self.filepath = filepath
                self.load_tiff(filepath)

                self.enable_menu('View')
                self.enable_menu('Annotation')

                if self.predictor.model is not None:
                    self.enable_menu('Object detection')

    def csv_open(self, event=None, filepath=None):
        if filepath is None:
            filepath = filedialog.askopenfilename(initialdir=self.INITIAL_PATH,
                                                  title="Select annotations CSV file to open...",
                                                  filetypes=[("Comma-separated CSV", '*.csv')])
        if bool(filepath):
            self.bbox_list.read_csv(filepath)

    def csv_save(self, event=None, filepath=None):
        if filepath is None:
            filepath = filedialog.asksaveasfilename(initialdir=self.INITIAL_PATH,
                                                    title="Select path and file name to save CSV...",
                                                    filetypes=[("Comma-separated CSV", '*.csv')])
        if bool(filepath):
            self.bbox_list.save_csv(filepath)

    def model_open(self, event=None):
        filepath = filedialog.askopenfilename(initialdir=self.INITIAL_PATH,
                                              title="Select pre-trained RetinaNet model file...",
                                              filetypes=[("PyTorch model", '*.pt')])
        if bool(filepath):
            if self.predictor.model_load(filepath):
                if self.tiff is not None:
                    self.enable_menu('Object detection')
            else:
                self.log_msg('Error occurred while loading the model.', level=1)

    def filter_open(self, event=None):
        filepath = filedialog.askopenfilename(initialdir=self.INITIAL_PATH,
                                              title="Select pre-trained distance filter...",
                                              filetypes=[("PyTorch model", '*.pt')])

        if bool(filepath):
            self.filter_file_path = filepath

    def show_settings_menu(self, event=None):
        self.settings_menu = tk.Toplevel(self)
        self.settings_menu.wm_title("Settings")

        self.settings_menu.overlap_ratio_var = tk.DoubleVar()
        self.settings_menu.overlap_ratio_var.set(
            self.predictor.overlap_threshold
        )
        self.settings_menu.overlap_ratio_lbl = tk.Label(
            self.settings_menu,
            text='Overlap ratio threshold: '
        )
        self.settings_menu.overlap_ratio_scale = tk.Scale(
            self.settings_menu,
            from_=0, to=1, resolution=0.1, orient=tk.HORIZONTAL,
            variable=self.settings_menu.overlap_ratio_var
        )

        self.settings_menu.score_var = tk.DoubleVar()
        self.settings_menu.score_var.set(
            self.predictor.score_threshold
        )
        self.settings_menu.score_lbl = tk.Label(
            self.settings_menu,
            text='Score threshold: '
        )
        self.settings_menu.score_scale = tk.Scale(
            self.settings_menu,
            from_=0, to=1, resolution=0.1, orient=tk.HORIZONTAL,
            variable=self.settings_menu.score_var
        )

        self.settings_menu.distance_var = tk.DoubleVar()
        self.settings_menu.distance_var.set(
            self.predictor.distance_threshold
        )
        self.settings_menu.distance_lbl = tk.Label(
            self.settings_menu,
            text='Distance threshold: '
        )
        self.settings_menu.distance_scale = tk.Scale(
            self.settings_menu,
            from_=2, to=4, resolution=0.1, orient=tk.HORIZONTAL,
            variable=self.settings_menu.distance_var
        )

        self.settings_menu.btn_save = tk.Button(
            self.settings_menu,
            text='Save settings',
            command=lambda: self.close_settings_menu(save=True))

        self.settings_menu.overlap_ratio_lbl.pack()
        self.settings_menu.overlap_ratio_scale.pack()
        self.settings_menu.score_lbl.pack()
        self.settings_menu.score_scale.pack()
        self.settings_menu.distance_lbl.pack()
        self.settings_menu.distance_scale.pack()
        self.settings_menu.btn_save.pack(side="bottom")

    def close_settings_menu(self, save=False):
        if save:
            self.predictor.score_threshold = self.settings_menu.score_var.get()
            self.predictor.overlap_threshold = self.settings_menu.overlap_ratio_var.get()
            self.predictor.distance_threshold = self.settings_menu.distance_var.get()
        self.settings_menu.destroy()

    def detect_objects(self, event=None):
        f_x1 = int(self.canvas.width / 2) - 400
        f_y1 = int(self.canvas.height / 2) - 400
        f_x2 = int(self.canvas.width / 2) + 400
        f_y2 = int(self.canvas.height / 2) + 400

        self.frame = self.canvas.create_rectangle(*(f_x1, f_y1, f_x2, f_y2), tag='frame', width=2, outline='red')

        self.log_msg('Object detection started...', level=1)
        self.busy_mode_start()
        x1 = self.x0 - 400
        y1 = self.y0 - 400
        x2 = self.x0 + 400
        y2 = self.y0 + 400

        img = TIFFPatchLoader(self.tiff).get_patch(x1=x1, y1=y1, x2=x2, y2=y2)
        bbox_list = self.predictor.get_bboxes(img=img)
        self.log_msg('Detected {} objects.'.format(len(bbox_list)), level=1)
        for b in bbox_list:
            b_x1 = f_x1 + int(b[0])
            b_y1 = f_y1 + int(b[1])
            b_x2 = f_x1 + int(b[2])
            b_y2 = f_y1 + int(b[3])
            self.canvas.create_rectangle(*(b_x1, b_y1, b_x2, b_y2), width=2, outline='green')
        self.busy_mode_stop()

    def apply_distance_filter(self, event=None):
        self.busy_mode_start()
        bbox_list = self.predictor.apply_distance_filter(
            path=self.filepath,
            xoff=self.x0 - 400,
            yoff=self.y0 - 400,
            width=100,
            height=100,
            model_path=self.filter_file_path
        )

        f_x1 = int(self.canvas.width / 2) - 400
        f_y1 = int(self.canvas.height / 2) - 400

        for b in bbox_list:
            b_x1 = f_x1 + int(b[0])
            b_y1 = f_y1 + int(b[1])
            b_x2 = f_x1 + int(b[2])
            b_y2 = f_y1 + int(b[3])
            self.canvas.create_rectangle(*(b_x1, b_y1, b_x2, b_y2), width=2, outline='yellow')
        self.busy_mode_stop()

    def select_area_switch_on(self, event=None):
        self.area_state = True
        self.config(cursor="pencil")

    def select_area_switch_off(self, event=None):
        self.area_state = None

        x1 = min(self.click_x, event.x) - self.x_delta
        y1 = min(self.click_y, event.y) - self.y_delta
        x2 = max(self.click_x, event.x) - self.x_delta
        y2 = max(self.click_y, event.y) - self.y_delta

        width = (x2 - x1) / self.scalefactor
        height = (y2 - y1) / self.scalefactor

        self.canvas.create_rectangle(*(self.click_x, self.click_y, event.x, event.y), width=2, outline='red')

        a_x1 = x1 / self.scalefactor
        a_y1 = y1 / self.scalefactor
        a_x2 = x2 / self.scalefactor
        a_y2 = y2 / self.scalefactor

        self.busy_mode_start()
        bbox_list = self.predictor.get_bboxes_for_area(tiff=self.tiff, a_x1=a_x1, a_y1=a_y1, a_x2=a_x2, a_y2=a_y2)
        columns = ['xoff', 'yoff', 'width', 'height', 'x1', 'y1', 'x2', 'y2']
        bb_df = pd.DataFrame(columns=columns, data=bbox_list)
        bb_df['img_path'] = ''
        bb_df['class_name'] = ''

        self.bbox_list.load_df(df=bb_df)

        self.busy_mode_stop()
        self.config(cursor="arrow")

    def enable_menu(self, menuname, event=None):
        self.menu_bar.entryconfig(menuname, state='normal')

    def disable_menu(self, menuname, event=None):
        self.menu_bar.entryconfig(menuname, state='disabled')

    def busy_mode_start(self):
        self.config(cursor='watch')
        for m in self.menu_names:
            self.disable_menu(m)

    def busy_mode_stop(self):
        self.config(cursor='arrow')
        for m in self.menu_names:
            self.enable_menu(m)

    def log_msg(self, msg, level=0):
        if level > 0:
            if type(msg) is list:
                print(' '.join([str(i) for i in msg]))
            else:
                print(msg)

    def main(self, event=None):
        self.disable_menu('View')
        self.disable_menu('Annotation')
        self.disable_menu('Object detection')

        self.bind("<Control-o>", self.file_open)
        self.bind("<Control-O>", self.file_open)
        self.bind("<Key-0>", self.load_overview)
        self.bind("<Key-1>", lambda x: self.zoom_switch_on(zoom_scalefactor=1., event=x))
        self.bind("<Key-2>", lambda x: self.zoom_switch_on(zoom_scalefactor=0.5, event=x))
        self.bind("<Key-3>", lambda x: self.zoom_switch_on(zoom_scalefactor=0.3, event=x))
        self.bind("<b>", self.add_bbox_switch_on)
        self.bind("<B>", self.add_bbox_switch_on)
        self.bind("<Control-d>", self.detect_objects)
        self.bind("<Control-D>", self.detect_objects)

        self.bind("<Escape>", self.esc_handler)


class ResizingCanvas(tk.Canvas):
    def __init__(self, parent, **kwargs):
        tk.Canvas.__init__(self, parent, **kwargs)
        self.bind('<Configure>', self.on_resize)
        self.height = self.winfo_reqheight()
        self.width = self.winfo_reqwidth()
        self.config(background='black')

    def on_resize(self, event):
        wscale = float(event.width) / self.width
        hscale = float(event.height) / self.height

        self.width = event.width
        self.height = event.height

        self.config(width=self.width, height=self.height)
        self.scale("all", 0, 0, wscale, hscale)
        # print(self.width, self.height)


class StatusBar(tk.Frame):
    def __init__(self, master):
        tk.Frame.__init__(self, master)
        self.label = tk.Label(self, bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.label.pack(fill=tk.X)

    def set(self, format, *args):
        self.label.config(text=format % args)
        self.label.update_idletasks()

    def clear(self):
        self.label.config(text="")
        self.label.update_idletasks()


if __name__ == "__main__":
    app = TIFFViewer()
    app.mainloop()
