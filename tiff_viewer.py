import tkinter as tk
from tkinter import filedialog
from viewer.patch_loader import TIFFPatchLoader
from viewer.annotation import BBoxList
import numpy as np
from osgeo import gdal
from PIL import Image, ImageTk


class TIFFViewer(tk.Tk):
    def __init__(self):
        super().__init__()

        self.INITIAL_PATH = "/home/kruglov/projects/cit/"

        self.filepath = None
        self.tiff = None
        self.img = None

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

        self.click_x = None
        self.click_y = None

        self.bbox_state = None
        self.start_bbox = None
        self.frame_coords = None
        self.frame = None

        self.show_frames = False

        self.zoom_state = None
        self.zoom_scalefactor = None

        # maximizing main window
        self.attributes('-zoomed', True)
        w, h = self.winfo_screenwidth(), self.winfo_screenheight()
        self.geometry("%dx%d+0+0" % (w, h))

        self.canvas = ResizingCanvas(self)  # width= width + 10, height= height + 10
        self.canvas.bind('<Button-1>', self.click_handler)

        self.canvas.pack(fill=tk.BOTH, anchor= tk.NW)

        # creating a top level menu
        self.menu_bar = tk.Menu(self)
        self.config(menu=self.menu_bar)

        # file submenu
        self.file_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label='File', menu=self.file_menu)
        self.file_menu.add_command(label="Open Image...", underline=1, command=self.file_open, accelerator="Ctrl+O")
        self.file_menu.add_command(label="Open CSV...", command=self.csv_open)
        self.file_menu.add_command(label= "Save CSV...", underline= 1, command= self.csv_save)
        self.file_menu.add_command(label="Exit", command=self.quit)

        # view submenu
        self.view_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label='View', menu=self.view_menu)
        self.view_menu.add_command(label="Overview", command=self.load_overview, accelerator="Ctrl+0")
        self.view_menu.add_command(label="Zoom 1/3", command=lambda: self.zoom_switch_on(0.3), accelerator="Ctrl+3")
        self.view_menu.add_command(label="Zoom 1/2", command=lambda: self.zoom_switch_on(0.5), accelerator="Ctrl+2")
        self.view_menu.add_command(label="Zoom 1/1", command=self.zoom_switch_on, accelerator="Ctrl+1")

        self.show_hide_frames = tk.IntVar()
        self.view_menu.add_radiobutton(label="Show frames", variable= self.show_hide_frames, value= 1)
        self.view_menu.add_radiobutton(label="Hide frames", variable= self.show_hide_frames, value= 0)

        # markup submenu
        self.annotation_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label='Annotation', menu=self.annotation_menu)
        self.annotation_menu.add_command(label="Add label box...", command=self.add_bbox_switch_on,
                                         accelerator="B")

        # self.status = StatusBar(self)
        # self.status.pack(side= tk.BOTTOM, fill= tk.X)

        self.main()

    def show_patch(self, x0=None, y0=None, x1=0, y1=0, x2= None, y2= None):
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.img, tag='imagesprite')

        self.set_patch_xy(x0=x0, y0=y0, x1=x1, y1=y1, x2=x2, y2=y2)
        if self.show_hide_frames.get() == 1:
            self.show_frames_for_patch()
            self.show_bboxes_for_patch()
        else:
            print(self.show_hide_frames.get())

    def show_frames_for_patch(self):
        frames = self.bbox_list.get_frames_in_patch(self.x1, self.y1, self.x2, self.y2)
        for f in frames:
            f = f * self.scalefactor
            self.canvas.create_rectangle(*f, tag='frame', width=2, outline='blue')

    def show_bboxes_for_patch(self):
        bboxes = self.bbox_list.get_bboxes_in_patch(self.x1, self.y1, self.x2, self.y2)
        for b in bboxes:
            b = b * self.scalefactor
            self.canvas.create_rectangle(*b, tag='bbox', width=2, outline='green')

    def set_patch_xy(self, x0=None, y0=None, x1=0, y1=0, x2= None, y2= None):
        self.x0 = x0 if x0 is not None else x1 + int((self.width / 2) / self.scalefactor)
        self.y0 = y0 if y0 is not None else y1 + int((self.height / 2) / self.scalefactor)

        self.x1 = x1
        self.y1 = y1

        self.x2 = x2 if x2 is not None else x1 + int(self.width / self.scalefactor)
        self.y2 = y2 if y2 is not None else y1 + int(self.height / self.scalefactor)

        self.log_msg(['self.x0, self.y0, self.x1, self.y1, self.x2, self.y2',
                      self.x0, self.y0, self.x1, self.y1, self.x2, self.y2], level= 1)

    def load_tiff(self, filepath):
        self.tiff = gdal.Open(filepath, gdal.GA_ReadOnly)
        # print(self.canvas.width, self.canvas.height)
        self.img, self.width, self.height, self.scalefactor = TIFFPatchLoader(self.tiff).get_overview(self.canvas.width,
                                                                                            self.canvas.height)
        self.show_patch()
        self.bbox_list = BBoxList(filepath)

    def load_overview(self, event=None):
        self.img, self.width, self.height, self.scalefactor = TIFFPatchLoader(self.tiff).get_overview(self.canvas.width,
                                                                                            self.canvas.height)
        self.show_patch()

    def load_zoom(self, event=None):
        x0 = self.x0 + int((self.click_x - self.width / 2) / self.scalefactor)
        y0 = self.y0 + int((self.click_y - self.height / 2) / self.scalefactor)

        self.log_msg(['scalefactor', self.scalefactor], level= 1)

        x1 = x0 - int(self.width / 2 / self.zoom_scalefactor)
        y1 = y0 - int(self.height / 2 / self.zoom_scalefactor)
        x2 = x1 + int(self.width / self.zoom_scalefactor)
        y2 = y1 + int(self.height / self.zoom_scalefactor)
        buffer_x = self.width
        buffer_y = self.height

        self.log_msg(['get_patch', x1, y1, x2, y2, buffer_x, buffer_y])

        self.img = TIFFPatchLoader(self.tiff).get_patch(x1, y1, x2, y2, buffer_x, buffer_y)

        #

        self.scalefactor = self.zoom_scalefactor
        self.show_patch(x0=x0, y0=y0, x1=x1, y1=y1, x2= x2, y2= y2)

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
            class_name= 'car'
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

    def esc_handler(self, event=None):

        if self.bbox_state:
            self.add_bbox_switch_off()

    def save_click_xy(self, event=None):
        self.log_msg(['Click coordinates: ', event.x, event.y], level= 1)

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

    def csv_open(self, event= None, filepath= None):
        if filepath is None:
            filepath = filedialog.askopenfilename(initialdir= self.INITIAL_PATH,
                                                  title="Select annotations CSV file to open...",
                                                  filetypes=[("Comma-separated CSV", '*.csv')])
        if bool(filepath):
            self.bbox_list.read_csv(filepath)

    def csv_save(self, event= None, filepath= None):
        if filepath is None:
            filepath = filedialog.asksaveasfilename(initialdir= self.INITIAL_PATH,
                                                    title="Select path and file name to save CSV...",
                                                    filetypes=[("Comma-separated CSV", '*.csv')])
        if bool(filepath):
            self.bbox_list.save_csv(filepath)


    def enable_menu(self, menuname, event=None):
        self.menu_bar.entryconfig(menuname, state='normal')

    def disable_menu(self, menuname, event=None):
        self.menu_bar.entryconfig(menuname, state='disabled')

    def log_msg(self, msg, level= 0):
        if level > 0:
            if type(msg) is list:
                print(' '.join([str(i) for i in msg]))
            else:
                print(msg)

    def main(self, event=None):
        self.disable_menu('View')
        self.disable_menu('Annotation')

        self.bind("<Control-o>", self.file_open)
        self.bind("<Control-O>", self.file_open)
        self.bind("<Key-0>", self.load_overview)
        self.bind("<Key-1>", lambda x: self.zoom_switch_on(zoom_scalefactor=1., event=x))
        self.bind("<Key-2>", lambda x: self.zoom_switch_on(zoom_scalefactor=0.5, event=x))
        self.bind("<Key-3>", lambda x: self.zoom_switch_on(zoom_scalefactor=0.3, event=x))
        self.bind("<b>", self.add_bbox_switch_on)
        self.bind("<B>", self.add_bbox_switch_on)

        self.bind("<Escape>", self.esc_handler)


class ResizingCanvas(tk.Canvas):
    def __init__(self, parent, **kwargs):
        tk.Canvas.__init__(self, parent, **kwargs)
        self.bind('<Configure>', self.on_resize)
        self.height = self.winfo_reqheight()
        self.width = self.winfo_reqwidth()

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
