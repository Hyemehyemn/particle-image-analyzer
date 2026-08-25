"""A beginner-friendly particle-shape analyzer."""


import csv
import math
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

from analyzer_core import (
    detect_horizontal_scale_bar,
    draw_particle_labels,
    evaluate_custom_formula,
    find_particles,
    length_to_micrometers,
    particle_formula_variables,
    segment_particles,
)

IMAGE_FILETYPES = [
    ("Image files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp"),
    ("All files", "*.*"),
]

class ImageAnalyzer:
    """The main application window."""

    MAX_DISPLAY_WIDTH = 900
    MAX_DISPLAY_HEIGHT = 550

    def __init__(self, root):
        self.root = root
        self.root.title("Particle Image Analyzer")
        self.original_image = None
        self.display_image = None
        self.photo = None
        self.particles = []
        self.selected_particle_id = None
        self.updating_table = False
        self.scale = 1.0
        self.calibration_image = None
        self.calibration_pixel_length = None
        self.micrometers_per_pixel = None
        self.calibration_source = None
        self.calibration_roi_window = None
        self.manual_calibration_window = None
        self.custom_formulas = []
        self.next_custom_formula_id = 1
        self.batch_results = []
        self.batch_formula_definitions = []
        self.batch_results_window = None
        self.display_decimal_places = {
            "area": None,
            "perimeter": None,
            "major_axis": None,
            "minor_axis": None,
            "circularity": None,
            "aspect_ratio": None,
        }

        scroll_container = tk.Frame(root)
        scroll_container.pack(fill="both", expand=True)
        self.scroll_canvas = tk.Canvas(
            scroll_container, highlightthickness=0, borderwidth=0
        )
        main_scrollbar = ttk.Scrollbar(
            scroll_container, orient="vertical", command=self.scroll_canvas.yview
        )
        self.scroll_canvas.configure(yscrollcommand=main_scrollbar.set)
        main_scrollbar.pack(side="right", fill="y")
        self.scroll_canvas.pack(side="left", fill="both", expand=True)

        self.content = tk.Frame(self.scroll_canvas)
        self.content_window = self.scroll_canvas.create_window(
            0, 0, anchor="nw", window=self.content
        )
        self.content.bind("<Configure>", self.content_size_changed)
        self.scroll_canvas.bind("<Configure>", self.scroll_canvas_size_changed)
        self.root.bind_all("<MouseWheel>", self.main_mouse_wheel, add="+")
        self.root.bind_all("<Button-4>", self.main_mouse_wheel, add="+")
        self.root.bind_all("<Button-5>", self.main_mouse_wheel, add="+")

        controls = tk.Frame(self.content, padx=10, pady=10)
        controls.pack(fill="x")
        tk.Button(controls, text="Open image", command=self.open_image).pack(side="left")

        particle_controls = tk.LabelFrame(
            self.content, text="Particle Shape Analysis", padx=10, pady=6
        )
        particle_controls.pack(fill="x", padx=10, pady=(0, 4))
        tk.Label(particle_controls, text="Minimum particle area:").pack(side="left")
        self.minimum_area = tk.IntVar(value=20)
        area_entry = tk.Entry(particle_controls, textvariable=self.minimum_area, width=7)
        area_entry.pack(side="left")
        area_entry.bind("<Return>", self.minimum_area_changed)
        area_entry.bind("<FocusOut>", self.minimum_area_changed)
        tk.Label(particle_controls, text="pixels").pack(side="left", padx=(3, 0))
        tk.Button(
            particle_controls, text="Analyze all particles",
            command=self.start_particle_analysis,
        ).pack(side="left", padx=(20, 0))
        tk.Button(
            particle_controls,
            text="Decimal Places…",
            command=self.open_decimal_places_dialog,
        ).pack(side="left", padx=(6, 0))
        tk.Button(
            particle_controls,
            text="Batch Analysis…",
            command=self.start_batch_analysis,
        ).pack(side="left", padx=(6, 0))

        formula_controls = tk.LabelFrame(
            self.content, text="Custom Formula", padx=10, pady=6
        )
        formula_controls.pack(fill="x", padx=10, pady=(0, 4))
        tk.Label(
            formula_controls,
            text=("A=Area, P=Perimeter, L=Major axis, S=Minor axis, "
                  "C=Circularity, AR=Aspect ratio  (dimensional variables use table units)"),
        ).pack(anchor="w")
        formula_input = tk.Frame(formula_controls)
        formula_input.pack(fill="x", pady=(4, 0))
        tk.Label(formula_input, text="Name:").pack(side="left")
        self.custom_formula_name = tk.StringVar()
        tk.Entry(
            formula_input, textvariable=self.custom_formula_name, width=24
        ).pack(side="left", padx=(4, 10))
        tk.Label(formula_input, text="Formula:").pack(side="left")
        self.custom_formula = tk.StringVar()
        formula_entry = tk.Entry(
            formula_input, textvariable=self.custom_formula, width=35
        )
        formula_entry.pack(side="left", padx=(4, 6))
        formula_entry.bind("<Return>", self.add_custom_formula)
        tk.Button(
            formula_input, text="Add Formula", command=self.add_custom_formula
        ).pack(side="left")
        self.formula_status = tk.Label(
            formula_input, text="Example: sqrt(4*A/pi)", anchor="w"
        )
        self.formula_status.pack(side="left", padx=(10, 0))
        self.formula_status_default_fg = self.formula_status.cget("foreground")
        formula_list_frame = tk.Frame(formula_controls)
        formula_list_frame.pack(fill="x", pady=(5, 0))
        tk.Label(formula_list_frame, text="Saved formulas:").pack(side="left")
        self.formula_list = tk.Listbox(
            formula_list_frame, height=3, exportselection=False
        )
        self.formula_list.pack(side="left", fill="x", expand=True, padx=(6, 6))
        tk.Button(
            formula_list_frame,
            text="Remove Selected Formula",
            command=self.remove_selected_formula,
        ).pack(side="left")

        calibration_controls = tk.LabelFrame(
            self.content, text="Scale Calibration", padx=10, pady=6
        )
        calibration_controls.pack(fill="x", padx=10, pady=(0, 4))
        tk.Button(
            calibration_controls,
            text="Load calibration image",
            command=self.load_calibration_image,
        ).pack(side="left")
        tk.Label(calibration_controls, text="Scale bar:").pack(
            side="left", padx=(15, 4)
        )
        self.scale_bar_value = tk.DoubleVar(value=3.0)
        tk.Entry(
            calibration_controls, textvariable=self.scale_bar_value, width=7
        ).pack(side="left")
        self.scale_bar_unit = tk.StringVar(value="µm")
        ttk.Combobox(
            calibration_controls,
            textvariable=self.scale_bar_unit,
            values=("nm", "µm", "mm"),
            width=4,
            state="readonly",
        ).pack(side="left", padx=(3, 0))
        tk.Button(
            calibration_controls,
            text="Select scale-bar region",
            command=self.select_scale_bar_region,
        ).pack(side="left", padx=(15, 0))
        tk.Button(
            calibration_controls,
            text="Manual 2-point fallback",
            command=self.start_manual_calibration,
        ).pack(side="left", padx=(6, 0))

        self.calibration_status = tk.Label(
            self.content,
            text="Calibration: load a corresponding SEM image.",
            anchor="w",
            padx=10,
        )
        self.calibration_status.pack(fill="x")

        summary = tk.Frame(self.content, padx=10)
        summary.pack(fill="x")
        self.result_label = tk.Label(summary, text="Open a recognition image.")
        self.result_label.pack(side="left")
        self.average_label = tk.Label(summary)
        self.average_label.pack(side="right")

        selection_summary = tk.Frame(self.content, padx=10)
        selection_summary.pack(fill="x")
        self.selected_particle_label = tk.Label(
            selection_summary,
            text="Selected particle: —",
            font=("TkDefaultFont", 12, "bold"),
        )
        self.selected_particle_label.pack(side="left")

        self.canvas = tk.Canvas(
            self.content, background="#333333", cursor="crosshair"
        )
        self.canvas.pack(padx=10, pady=10)
        self.canvas.bind("<Button-1>", self.image_clicked)

        table_frame = tk.Frame(self.content, padx=10)
        table_frame.pack(fill="both", expand=True, pady=(0, 10))
        self.base_columns = (
            "id", "area", "perimeter", "major_axis", "minor_axis",
            "circularity", "aspect_ratio",
        )
        self.table = ttk.Treeview(
            table_frame, columns=self.base_columns, show="headings", height=8
        )
        headings = {
            "id": "ID", "area": "Area (px)", "perimeter": "Perimeter (px)",
            "major_axis": "Major axis (px)", "minor_axis": "Minor axis (px)",
            "circularity": "Circularity", "aspect_ratio": "Aspect ratio",
        }
        for column in self.base_columns:
            self.table.heading(column, text=headings[column])
            self.table.column(column, width=115, anchor="center", stretch=False)
        self.table.column("id", width=55, stretch=False)
        vertical_scrollbar = ttk.Scrollbar(
            table_frame, orient="vertical", command=self.table.yview
        )
        horizontal_scrollbar = ttk.Scrollbar(
            table_frame, orient="horizontal", command=self.table.xview
        )
        self.table.configure(
            yscrollcommand=vertical_scrollbar.set,
            xscrollcommand=horizontal_scrollbar.set,
        )
        table_frame.grid_rowconfigure(0, weight=1)
        table_frame.grid_columnconfigure(0, weight=1)
        self.table.grid(row=0, column=0, sticky="nsew")
        vertical_scrollbar.grid(row=0, column=1, sticky="ns")
        horizontal_scrollbar.grid(row=1, column=0, sticky="ew")
        self.table.bind("<<TreeviewSelect>>", self.table_selection_changed)

        # Keep the initial window within a laptop display; additional content
        # remains available through the main scrollbar.
        self.root.update_idletasks()
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        window_width = min(max(self.content.winfo_reqwidth() + 20, 700), screen_width - 40)
        window_height = min(self.content.winfo_reqheight(), screen_height - 100)
        self.root.geometry(f"{window_width}x{max(window_height, 500)}")

    def content_size_changed(self, _event=None):
        """Update the main scrollable region when its contents change size."""
        self.scroll_canvas.configure(scrollregion=self.scroll_canvas.bbox("all"))

    def scroll_canvas_size_changed(self, event):
        """Make the main content follow the available window width."""
        self.scroll_canvas.itemconfigure(self.content_window, width=event.width)

    def main_mouse_wheel(self, event):
        """Scroll the complete interface with a mouse wheel or trackpad."""
        try:
            if event.widget.winfo_toplevel() != self.root:
                return
        except tk.TclError:
            return

        if getattr(event, "num", None) == 4:
            direction = -1
        elif getattr(event, "num", None) == 5:
            direction = 1
        else:
            delta = getattr(event, "delta", 0)
            if not delta:
                return
            # Windows commonly reports multiples of 120; macOS trackpads
            # commonly report small deltas.
            direction = -int(delta / 120) if abs(delta) >= 120 else (-1 if delta > 0 else 1)
        self.scroll_canvas.yview_scroll(direction, "units")
        return "break"

    def open_image(self):
        """Ask the user for an image file and display it."""
        file_path = filedialog.askopenfilename(
            title="Choose an image",
            filetypes=IMAGE_FILETYPES,
        )
        if not file_path:
            return
        try:
            self.original_image = Image.open(file_path).convert("RGB")
        except (OSError, ValueError) as error:
            messagebox.showerror("Could not open image", str(error))
            return

        width, height = self.original_image.size
        self.scale = min(1.0, self.MAX_DISPLAY_WIDTH / width, self.MAX_DISPLAY_HEIGHT / height)
        display_size = (max(1, int(width * self.scale)), max(1, int(height * self.scale)))
        self.display_image = self.original_image.resize(display_size, Image.Resampling.LANCZOS)
        self.clear_particle_selection()
        self.show_image(self.display_image)
        self.clear_table()
        self.average_label.config(text="")
        if self.calibration_image is not None:
            self.apply_calibration(self.calibration_pixel_length)
        self.analyze_particles()

    def load_calibration_image(self):
        """Load the SEM image used for ROI-assisted scale calibration."""
        file_path = filedialog.askopenfilename(
            title="Choose the corresponding SEM calibration image",
            filetypes=IMAGE_FILETYPES,
        )
        if not file_path:
            return
        try:
            calibration_image = Image.open(file_path).convert("RGB")
        except (OSError, ValueError) as error:
            messagebox.showerror("Could not open calibration image", str(error))
            return

        self.calibration_image = calibration_image
        self.calibration_pixel_length = None
        self.micrometers_per_pixel = None
        self.calibration_source = None
        self.recalculate_custom_formulas()
        self.update_table(self.particles)
        self.calibration_status.config(
            text="Calibration image loaded. Click Select scale-bar region."
        )

    def images_have_matching_dimensions(self, show_warning=True):
        """Check that calibration can transfer to the recognition image."""
        if self.original_image is None or self.calibration_image is None:
            return False
        if self.original_image.size == self.calibration_image.size:
            return True
        if show_warning:
            messagebox.showwarning(
                "Calibration image size mismatch",
                "Calibration was not applied.\n\n"
                f"Recognition image: {self.original_image.width} × "
                f"{self.original_image.height} px\n"
                f"Calibration image: {self.calibration_image.width} × "
                f"{self.calibration_image.height} px\n\n"
                "Both images must have the same pixel dimensions and field of view.",
            )
        return False

    def select_scale_bar_region(self):
        """Let the user drag an ROI, then detect and show its bracket scale bar."""
        if self.calibration_image is None:
            messagebox.showinfo(
                "Calibration image needed", "Load the SEM calibration image first."
            )
            return

        if self.calibration_roi_window is not None:
            try:
                self.calibration_roi_window.destroy()
            except tk.TclError:
                pass
        window = tk.Toplevel(self.root)
        self.calibration_roi_window = window
        window.title("Select scale-bar region")
        instruction = tk.Label(
            window, text="Drag a rectangle tightly around the horizontal scale bar."
        )
        instruction.pack(padx=10, pady=(10, 4))
        pending_detection = {"pixel_length": None}

        def accept_detection():
            pixel_length = pending_detection["pixel_length"]
            if pixel_length is None:
                return
            if self.calibration_source == "manual":
                replace_manual = messagebox.askyesno(
                    "Replace manual calibration?",
                    "A manual calibration is currently accepted. Replace it with "
                    f"this automatically detected {pixel_length:.2f} px scale bar?",
                    parent=window,
                )
                if not replace_manual:
                    retained_scale = (
                        f"{self.micrometers_per_pixel:.6g} µm/pixel"
                        if self.micrometers_per_pixel is not None else "pending image match"
                    )
                    self.calibration_status.config(
                        text=f"Automatic detection: {pixel_length:.2f} px (not applied; "
                        f"manual calibration retained at {retained_scale})."
                    )
                    return
            if self.apply_calibration(
                pixel_length, source="automatic", confirmed_manual_replacement=True
            ):
                instruction.config(
                    text=f"Applied detected bracket scale bar: {pixel_length:.2f} px"
                )

        accept_button = tk.Button(
            window,
            text="Apply detected calibration",
            command=accept_detection,
            state="disabled",
        )
        accept_button.pack(pady=(0, 4))
        image = self.calibration_image
        roi_scale = min(
            1.0, self.MAX_DISPLAY_WIDTH / image.width,
            self.MAX_DISPLAY_HEIGHT / image.height,
        )
        shown = image.resize(
            (max(1, int(image.width * roi_scale)),
             max(1, int(image.height * roi_scale))),
            Image.Resampling.LANCZOS,
        )
        photo = ImageTk.PhotoImage(shown)
        canvas = tk.Canvas(window, width=shown.width, height=shown.height, cursor="crosshair")
        canvas.pack(padx=10, pady=(0, 10))
        canvas.create_image(0, 0, anchor="nw", image=photo)
        canvas.image = photo
        drag = {"start": None, "rectangle": None}

        def drag_started(event):
            drag["start"] = (event.x, event.y)
            pending_detection["pixel_length"] = None
            accept_button.config(state="disabled")
            if drag["rectangle"] is not None:
                canvas.delete(drag["rectangle"])
            canvas.delete("detected-scale-bar")
            drag["rectangle"] = canvas.create_rectangle(
                event.x, event.y, event.x, event.y, outline="#00ffff", width=2
            )

        def drag_moved(event):
            if drag["start"] is not None:
                canvas.coords(drag["rectangle"], *drag["start"], event.x, event.y)

        def drag_finished(event):
            if drag["start"] is None:
                return
            start_x, start_y = drag["start"]
            drag["start"] = None
            left = max(0, int(min(start_x, event.x) / roi_scale))
            top = max(0, int(min(start_y, event.y) / roi_scale))
            right = min(image.width, int(math.ceil(max(start_x, event.x) / roi_scale)))
            bottom = min(image.height, int(math.ceil(max(start_y, event.y) / roi_scale)))
            if right - left < 10 or bottom - top < 5:
                instruction.config(text="ROI is too small. Drag around the complete scale bar.")
                return
            detection = detect_horizontal_scale_bar(image.crop((left, top, right, bottom)))
            if detection is None:
                instruction.config(
                    text="No bracket scale bar found (horizontal line plus vertical lines at "
                    "both ends). Try another ROI or use manual fallback."
                )
                return

            pixel_length, (roi_left, roi_right) = detection
            endpoint_1 = (left + roi_left[0], top + roi_left[1])
            endpoint_2 = (left + roi_right[0], top + roi_right[1])
            displayed = tuple(
                coordinate * roi_scale
                for point in (endpoint_1, endpoint_2) for coordinate in point
            )
            canvas.delete("detected-scale-bar")
            canvas.create_line(
                *displayed, fill="#ff00ff", width=3, tags="detected-scale-bar"
            )
            radius = 5
            for x, y in (displayed[:2], displayed[2:]):
                canvas.create_oval(
                    x - radius, y - radius, x + radius, y + radius,
                    fill="#ffff00", outline="#ff00ff", width=2,
                    tags="detected-scale-bar",
                )
            instruction.config(
                text=f"Detected scale bar: {pixel_length:.2f} px. Review the magenta "
                "line and yellow endpoints, then click Apply detected calibration."
            )
            pending_detection["pixel_length"] = pixel_length
            accept_button.config(state="normal")
            canvas.update_idletasks()

        canvas.bind("<ButtonPress-1>", drag_started)
        canvas.bind("<B1-Motion>", drag_moved)
        canvas.bind("<ButtonRelease-1>", drag_finished)

    def apply_calibration(
        self, pixel_length, source=None, confirmed_manual_replacement=False
    ):
        """Validate and store a detected or manually measured calibration."""
        if not pixel_length or pixel_length <= 0:
            return False
        if (
            source == "automatic"
            and self.calibration_source == "manual"
            and not confirmed_manual_replacement
        ):
            if not messagebox.askyesno(
                "Replace manual calibration?",
                "A manual calibration is currently accepted. Replace it with "
                "the automatic result?",
            ):
                return False
        try:
            physical_micrometers = length_to_micrometers(
                self.scale_bar_value.get(), self.scale_bar_unit.get()
            )
            if physical_micrometers <= 0:
                raise ValueError("Scale-bar value must be greater than zero")
        except (tk.TclError, TypeError, ValueError) as error:
            messagebox.showwarning("Invalid scale-bar value", str(error))
            return False

        self.calibration_pixel_length = float(pixel_length)
        if source is not None:
            self.calibration_source = source
        if self.original_image is None:
            self.micrometers_per_pixel = None
            self.recalculate_custom_formulas()
            self.calibration_status.config(
                text=f"Detected scale bar: {pixel_length:.2f} px. "
                "Load the corresponding recognition image to apply it."
            )
            return True
        if not self.images_have_matching_dimensions():
            self.micrometers_per_pixel = None
            self.recalculate_custom_formulas()
            self.calibration_status.config(
                text=f"Detected scale bar: {pixel_length:.2f} px. "
                "Not applied: image dimensions differ."
            )
            self.update_table(self.particles)
            return True

        self.micrometers_per_pixel = physical_micrometers / pixel_length
        self.recalculate_custom_formulas()
        entered_value_per_pixel = self.scale_bar_value.get() / pixel_length
        entered_unit = self.scale_bar_unit.get()
        self.calibration_status.config(
            text=f"Detected scale bar: {pixel_length:.2f} px    "
            f"Calibration: {entered_value_per_pixel:.6g} {entered_unit}/pixel "
            f"({self.micrometers_per_pixel:.6g} µm/pixel)"
        )
        self.update_table(self.particles)
        return True

    def start_manual_calibration(self):
        """Open the calibration image for a two-endpoint manual measurement."""
        if self.calibration_image is None:
            messagebox.showinfo(
                "Calibration image needed", "Load the SEM calibration image first."
            )
            return
        if self.manual_calibration_window is not None:
            try:
                self.manual_calibration_window.destroy()
            except tk.TclError:
                pass

        window = tk.Toplevel(self.root)
        self.manual_calibration_window = window
        window.title("Manual scale-bar calibration")
        tk.Label(
            window, text="Click the left and right endpoints of the scale bar."
        ).pack(padx=10, pady=(10, 4))
        image = self.calibration_image
        manual_scale = min(
            1.0, self.MAX_DISPLAY_WIDTH / image.width,
            self.MAX_DISPLAY_HEIGHT / image.height,
        )
        shown = image.resize(
            (max(1, int(image.width * manual_scale)),
             max(1, int(image.height * manual_scale))),
            Image.Resampling.LANCZOS,
        )
        photo = ImageTk.PhotoImage(shown)
        canvas = tk.Canvas(window, width=shown.width, height=shown.height, cursor="crosshair")
        canvas.pack(padx=10, pady=(0, 10))
        canvas.create_image(0, 0, anchor="nw", image=photo)
        canvas.image = photo
        points = []

        def endpoint_clicked(event):
            points.append((event.x / manual_scale, event.y / manual_scale))
            radius = 4
            canvas.create_oval(
                event.x - radius, event.y - radius, event.x + radius, event.y + radius,
                outline="#ff00ff", width=2,
            )
            if len(points) == 2:
                pixel_length = math.dist(points[0], points[1])
                if self.apply_calibration(pixel_length, source="manual"):
                    window.destroy()
                    self.manual_calibration_window = None

        canvas.bind("<Button-1>", endpoint_clicked)

    def image_clicked(self, event):
        """Select the particle under the clicked image position."""
        if self.original_image is None:
            return
        particle_id = self.particle_at_display_point(event.x, event.y)
        if particle_id is not None:
            self.select_particle(particle_id, select_table_row=True)

    def minimum_area_changed(self, _event=None):
        """Re-run particle analysis when the minimum area changes."""
        if self.original_image is not None:
            self.analyze_particles()

    def start_particle_analysis(self):
        """Analyze every particle in the recognition image."""
        if self.original_image is not None:
            self.analyze_particles()

    def get_minimum_area(self):
        """Read and validate the minimum-area entry."""
        try:
            value = self.minimum_area.get()
        except tk.TclError:
            value = 1
        value = max(1, value)
        self.minimum_area.set(value)
        return value

    def open_decimal_places_dialog(self):
        """Configure display-only precision independently for table columns."""
        window = tk.Toplevel(self.root)
        window.title("Table Decimal Places")
        window.transient(self.root)
        tk.Label(
            window,
            text="Enter 0–12 decimal places. Leave blank to keep the current default format.",
            padx=10,
            pady=8,
        ).pack(anchor="w")

        body = tk.Frame(window)
        body.pack(fill="both", expand=True, padx=10)
        canvas = tk.Canvas(body, highlightthickness=0, width=380)
        scrollbar = ttk.Scrollbar(body, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        fields = tk.Frame(canvas)
        fields_window = canvas.create_window(0, 0, anchor="nw", window=fields)
        fields.bind(
            "<Configure>",
            lambda _event: canvas.configure(scrollregion=canvas.bbox("all")),
        )
        canvas.bind(
            "<Configure>",
            lambda event: canvas.itemconfigure(fields_window, width=event.width),
        )

        column_labels = [
            ("area", "Area"),
            ("perimeter", "Perimeter"),
            ("major_axis", "Major axis"),
            ("minor_axis", "Minor axis"),
            ("circularity", "Circularity"),
            ("aspect_ratio", "Aspect ratio"),
        ] + [
            (formula["column_id"], formula["name"])
            for formula in self.custom_formulas
        ]
        precision_variables = {}
        for row, (column_id, label) in enumerate(column_labels):
            tk.Label(fields, text=label, anchor="w").grid(
                row=row, column=0, sticky="ew", padx=(4, 15), pady=3
            )
            current = self.display_decimal_places.get(column_id)
            variable = tk.StringVar(value="" if current is None else str(current))
            precision_variables[column_id] = variable
            tk.Entry(fields, textvariable=variable, width=6).grid(
                row=row, column=1, sticky="e", padx=(0, 4), pady=3
            )
        fields.grid_columnconfigure(0, weight=1)

        def apply_precision():
            updated = {}
            for column_id, variable in precision_variables.items():
                text = variable.get().strip()
                if not text:
                    updated[column_id] = None
                    continue
                try:
                    places = int(text)
                except ValueError:
                    places = -1
                if not 0 <= places <= 12:
                    messagebox.showerror(
                        "Invalid decimal places",
                        "Each value must be a whole number from 0 to 12, or blank.",
                        parent=window,
                    )
                    return
                updated[column_id] = places
            self.display_decimal_places.update(updated)
            self.refresh_table_preserving_selection()
            if self.batch_results and self.batch_results_window is not None:
                self.show_batch_results()
            window.destroy()

        buttons = tk.Frame(window, padx=10, pady=10)
        buttons.pack(fill="x")
        tk.Button(buttons, text="Apply", command=apply_precision).pack(side="right")
        tk.Button(buttons, text="Cancel", command=window.destroy).pack(
            side="right", padx=(0, 6)
        )
        window.update_idletasks()
        window.geometry(f"420x{min(600, max(260, window.winfo_reqheight()))}")

    def format_table_number(self, column_id, value, default_format):
        """Format a number without changing its stored full-precision value."""
        decimal_places = self.display_decimal_places.get(column_id)
        if decimal_places is None:
            return default_format(value)
        return f"{float(value):.{decimal_places}f}"

    def start_batch_analysis(self):
        """Analyze multiple compatible recognition images with one calibration."""
        if self.micrometers_per_pixel is None or self.calibration_image is None:
            messagebox.showwarning(
                "Calibration required",
                "Apply a scale calibration before starting batch analysis.",
            )
            return
        file_paths = filedialog.askopenfilenames(
            title="Choose recognition images for batch analysis",
            filetypes=IMAGE_FILETYPES,
        )
        if not file_paths:
            return

        progress_window = tk.Toplevel(self.root)
        progress_window.title("Batch Analysis Progress")
        progress_window.transient(self.root)
        progress_label = tk.Label(
            progress_window, text="Preparing batch…", width=55, anchor="w"
        )
        progress_label.pack(padx=15, pady=(15, 8))
        progress = ttk.Progressbar(
            progress_window, maximum=len(file_paths), length=430
        )
        progress.pack(padx=15, pady=(0, 15))
        progress_window.update_idletasks()

        expected_size = self.calibration_image.size
        length_scale = self.micrometers_per_pixel
        minimum_area = self.get_minimum_area()
        batch_formulas = [
            {
                "name": formula["name"],
                "expression": formula["expression"],
                "column_id": formula["column_id"],
            }
            for formula in self.custom_formulas
        ]
        completed_results = []
        for index, file_path in enumerate(file_paths, start=1):
            image_name = Path(file_path).name
            progress_label.config(
                text=f"Analyzing {index} of {len(file_paths)}: {image_name}"
            )
            progress["value"] = index - 1
            progress_window.update()
            try:
                with Image.open(file_path) as opened_image:
                    image = opened_image.convert("RGB")
            except (OSError, ValueError) as error:
                messagebox.showwarning(
                    "Could not open batch image",
                    f"{image_name} was skipped:\n{error}",
                    parent=progress_window,
                )
                progress["value"] = index
                continue

            if image.size != expected_size:
                skip_image = messagebox.askyesno(
                    "Incompatible batch image",
                    f"{image_name} is {image.width} × {image.height} px.\n"
                    f"The calibration requires {expected_size[0]} × {expected_size[1]} px.\n\n"
                    "Skip this image and continue? Click No to stop the batch.",
                    parent=progress_window,
                    icon="warning",
                )
                if skip_image:
                    progress["value"] = index
                    continue
                progress_window.destroy()
                return

            particles = find_particles(segment_particles(image), minimum_area)
            rows = []
            formula_errors = set()
            for particle_id, particle in enumerate(particles, start=1):
                row = {
                    "particle_id": particle_id,
                    "area": particle["area"] * length_scale**2,
                    "perimeter": particle["perimeter"] * length_scale,
                    "major_axis": particle["major_axis"] * length_scale,
                    "minor_axis": particle["minor_axis"] * length_scale,
                    "circularity": particle["circularity"],
                    "aspect_ratio": particle["aspect_ratio"],
                    "custom": {},
                }
                variables = self.particle_formula_variables(
                    particle, length_scale=length_scale
                )
                for formula in batch_formulas:
                    try:
                        value = evaluate_custom_formula(
                            formula["expression"], variables
                        )
                    except ValueError as error:
                        value = None
                        formula_errors.add(f'{formula["name"]}: {error}')
                    row["custom"][formula["column_id"]] = value
                rows.append(row)
            if formula_errors:
                messagebox.showwarning(
                    "Batch formula warning",
                    f"Some custom values for {image_name} are blank:\n"
                    + "\n".join(sorted(formula_errors)),
                    parent=progress_window,
                )
            completed_results.append({
                "image_name": image_name,
                "source_path": str(file_path),
                "particle_count": len(particles),
                "rows": rows,
            })
            progress["value"] = index
            progress_window.update_idletasks()

        progress_window.destroy()
        if not completed_results:
            messagebox.showinfo("Batch analysis", "No compatible images were analyzed.")
            return
        self.batch_results = completed_results
        self.batch_formula_definitions = batch_formulas
        self.show_batch_results()

    def batch_display_value(self, column_id, value):
        """Format a batch value with the same display settings as the main table."""
        if value is None:
            return ""
        defaults = {
            "area": lambda number: f"{number:.6g}",
            "perimeter": lambda number: f"{number:.6g}",
            "major_axis": lambda number: f"{number:.6g}",
            "minor_axis": lambda number: f"{number:.6g}",
            "circularity": lambda number: f"{number:.3f}",
            "aspect_ratio": lambda number: f"{number:.3f}",
        }
        default = defaults.get(column_id, lambda number: f"{number:.8g}")
        return self.format_table_number(column_id, value, default)

    def show_batch_results(self):
        """Show batch particles grouped under their source image names."""
        if self.batch_results_window is not None:
            try:
                self.batch_results_window.destroy()
            except tk.TclError:
                pass
        window = tk.Toplevel(self.root)
        self.batch_results_window = window
        window.title("Batch Particle Results")
        window.geometry("1000x520")
        tk.Label(
            window,
            text=f"Analyzed {len(self.batch_results)} images with the current calibration.",
            anchor="w",
            padx=10,
            pady=8,
        ).pack(fill="x")

        table_frame = tk.Frame(window, padx=10)
        table_frame.pack(fill="both", expand=True)
        custom_columns = tuple(
            formula["column_id"] for formula in self.batch_formula_definitions
        )
        columns = ("particle_id",) + self.base_columns[1:] + custom_columns
        table = ttk.Treeview(
            table_frame, columns=columns, show="tree headings"
        )
        table.heading("#0", text="Image Name")
        table.column("#0", width=220, stretch=False)
        headings = {
            "particle_id": "Particle ID",
            "area": "Area (µm²)",
            "perimeter": "Perimeter (µm)",
            "major_axis": "Major axis (µm)",
            "minor_axis": "Minor axis (µm)",
            "circularity": "Circularity",
            "aspect_ratio": "Aspect ratio",
        }
        for column_id, heading in headings.items():
            table.heading(column_id, text=heading)
            table.column(column_id, width=120, anchor="center", stretch=False)
        table.column("particle_id", width=90, anchor="center", stretch=False)
        for formula in self.batch_formula_definitions:
            table.heading(formula["column_id"], text=formula["name"])
            table.column(
                formula["column_id"], width=140, anchor="center", stretch=False
            )
        vertical = ttk.Scrollbar(table_frame, orient="vertical", command=table.yview)
        horizontal = ttk.Scrollbar(table_frame, orient="horizontal", command=table.xview)
        table.configure(yscrollcommand=vertical.set, xscrollcommand=horizontal.set)
        table_frame.grid_rowconfigure(0, weight=1)
        table_frame.grid_columnconfigure(0, weight=1)
        table.grid(row=0, column=0, sticky="nsew")
        vertical.grid(row=0, column=1, sticky="ns")
        horizontal.grid(row=1, column=0, sticky="ew")

        for image_index, result in enumerate(self.batch_results):
            parent_id = f"batch-image-{image_index}"
            table.insert(
                "", "end", iid=parent_id,
                text=result["image_name"],
                values=(f'{result["particle_count"]} particles',),
                open=True,
            )
            for row in result["rows"]:
                values = [
                    row["particle_id"],
                    self.batch_display_value("area", row["area"]),
                    self.batch_display_value("perimeter", row["perimeter"]),
                    self.batch_display_value("major_axis", row["major_axis"]),
                    self.batch_display_value("minor_axis", row["minor_axis"]),
                    self.batch_display_value("circularity", row["circularity"]),
                    self.batch_display_value("aspect_ratio", row["aspect_ratio"]),
                ]
                values.extend(
                    self.batch_display_value(
                        formula["column_id"], row["custom"].get(formula["column_id"])
                    )
                    for formula in self.batch_formula_definitions
                )
                table.insert(parent_id, "end", values=values)

        buttons = tk.Frame(window, padx=10, pady=10)
        buttons.pack(fill="x")
        tk.Button(
            buttons, text="Export One CSV per Image",
            command=self.export_batch_separate_csvs,
        ).pack(side="left")
        tk.Button(
            buttons, text="Export Combined CSV",
            command=self.export_batch_combined_csv,
        ).pack(side="left", padx=(8, 0))

    def batch_csv_headers(self, include_image_name=False):
        """Return stable export headers for current batch results."""
        headers = [
            "Particle ID", "Area (µm²)", "Perimeter (µm)",
            "Major axis (µm)", "Minor axis (µm)", "Circularity", "Aspect ratio",
        ] + [formula["name"] for formula in self.batch_formula_definitions]
        return (["Image Name"] + headers) if include_image_name else headers

    def batch_csv_row(self, result, row, include_image_name=False):
        """Return one full-precision batch particle export row."""
        values = [
            row["particle_id"], row["area"], row["perimeter"],
            row["major_axis"], row["minor_axis"], row["circularity"],
            row["aspect_ratio"],
        ] + [
            row["custom"].get(formula["column_id"], "")
            for formula in self.batch_formula_definitions
        ]
        return ([result["image_name"]] + values) if include_image_name else values

    def export_batch_separate_csvs(self):
        """Export one full-precision particle CSV for each batch image."""
        directory = filedialog.askdirectory(title="Choose folder for batch CSV files")
        if not directory:
            return
        used_names = set()
        for result in self.batch_results:
            stem = Path(result["image_name"]).stem or "image"
            candidate = stem
            suffix = 2
            while candidate.casefold() in used_names or (Path(directory) / f"{candidate}.csv").exists():
                candidate = f"{stem}_{suffix}"
                suffix += 1
            used_names.add(candidate.casefold())
            output_path = Path(directory) / f"{candidate}.csv"
            with output_path.open("w", newline="", encoding="utf-8-sig") as output:
                writer = csv.writer(output)
                writer.writerow(self.batch_csv_headers())
                for row in result["rows"]:
                    writer.writerow(self.batch_csv_row(result, row))
        messagebox.showinfo(
            "Batch export complete",
            f"Exported {len(self.batch_results)} CSV files to {directory}.",
        )

    def export_batch_combined_csv(self):
        """Export all batch particles with an Image Name column."""
        output_path = filedialog.asksaveasfilename(
            title="Save combined batch CSV",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if not output_path:
            return
        with Path(output_path).open("w", newline="", encoding="utf-8-sig") as output:
            writer = csv.writer(output)
            writer.writerow(self.batch_csv_headers(include_image_name=True))
            for result in self.batch_results:
                for row in result["rows"]:
                    writer.writerow(
                        self.batch_csv_row(result, row, include_image_name=True)
                    )
        messagebox.showinfo("Batch export complete", f"Saved {output_path}.")

    def particle_formula_variables(self, particle, length_scale=None):
        """Return the fixed formula variables in the table's current units."""
        if length_scale is None:
            length_scale = self.micrometers_per_pixel or 1.0
        return particle_formula_variables(particle, length_scale)

    def calculate_formula_for_particles(self, expression):
        """Calculate one expression for every current particle."""
        calculated = {}
        for particle_id, particle in enumerate(self.particles, start=1):
            try:
                calculated[particle_id] = evaluate_custom_formula(
                    expression, self.particle_formula_variables(particle)
                )
            except ValueError as error:
                raise ValueError(f"Particle {particle_id}: {error}") from error
        return calculated

    def add_custom_formula(self, _event=None):
        """Validate, calculate, and retain a new named custom formula."""
        if not self.particles:
            messagebox.showinfo(
                "No particles", "Open and analyze a recognition image first."
            )
            return
        name = self.custom_formula_name.get().strip()
        expression = self.custom_formula.get()
        if not name:
            messagebox.showerror("Formula name required", "Enter a name for the formula.")
            return
        if any(formula["name"].casefold() == name.casefold() for formula in self.custom_formulas):
            messagebox.showerror(
                "Duplicate formula name",
                f'A custom formula named "{name}" already exists. Choose a unique name.',
            )
            return
        try:
            calculated = self.calculate_formula_for_particles(expression)
        except ValueError as error:
            self.formula_status.config(
                text=f"Formula error: {error}", foreground="red"
            )
            messagebox.showerror(
                "Invalid custom formula", f"Could not add {name}:\n{error}"
            )
            return

        column_id = f"custom_formula_{self.next_custom_formula_id}"
        self.next_custom_formula_id += 1
        self.custom_formulas.append({
            "name": name,
            "expression": expression,
            "column_id": column_id,
            "values": calculated,
        })
        self.display_decimal_places[column_id] = None
        self.refresh_formula_list()
        self.configure_custom_formula_columns()
        self.formula_status.config(
            text=f'Added "{name}" for {len(calculated)} particles',
            foreground=self.formula_status_default_fg,
        )
        self.custom_formula_name.set("")
        self.custom_formula.set("")
        self.refresh_table_preserving_selection()

    def remove_selected_formula(self):
        """Remove the selected saved formula and its result column."""
        selection = self.formula_list.curselection()
        if not selection:
            messagebox.showinfo(
                "Select a formula", "Select a saved formula to remove."
            )
            return
        removed = self.custom_formulas.pop(selection[0])
        self.display_decimal_places.pop(removed["column_id"], None)
        self.refresh_formula_list()
        self.configure_custom_formula_columns()
        self.formula_status.config(
            text=f'Removed "{removed["name"]}"',
            foreground=self.formula_status_default_fg,
        )
        self.refresh_table_preserving_selection()

    def refresh_formula_list(self):
        """Synchronize the saved-formula list with formula state."""
        self.formula_list.delete(0, "end")
        for formula in self.custom_formulas:
            self.formula_list.insert(
                "end", f'{formula["name"]} = {formula["expression"]}'
            )

    def configure_custom_formula_columns(self):
        """Make the table columns match all currently saved formulas."""
        columns = self.base_columns + tuple(
            formula["column_id"] for formula in self.custom_formulas
        )
        self.table.configure(columns=columns)
        for column in self.base_columns:
            self.table.column(column, stretch=False)
        self.table.heading("circularity", text="Circularity")
        self.table.heading("aspect_ratio", text="Aspect ratio")
        for formula in self.custom_formulas:
            column_id = formula["column_id"]
            self.table.heading(column_id, text=formula["name"])
            self.table.column(
                column_id, width=130, anchor="center", stretch=False
            )

    def refresh_table_preserving_selection(self):
        """Refresh formula columns without breaking particle/table selection."""
        selected_particle_id = self.selected_particle_id
        self.update_table(self.particles)
        if selected_particle_id is not None:
            row_id = f"particle-{selected_particle_id}"
            if self.table.exists(row_id):
                self.table.selection_set(row_id)
                self.table.focus(row_id)
                self.table.see(row_id)

    def recalculate_custom_formulas(self):
        """Update every saved formula after particles or table units change."""
        failed = []
        for formula in self.custom_formulas:
            try:
                formula["values"] = self.calculate_formula_for_particles(
                    formula["expression"]
                )
            except ValueError as error:
                formula["values"] = {}
                failed.append(f'{formula["name"]}: {error}')
        if failed:
            self.formula_status.config(
                text=f"Formula recalculation error: {failed[0]}", foreground="red"
            )
        return not failed

    def analyze_particles(self):
        """Detect and measure every particle in the recognition image."""
        color_masks = segment_particles(self.original_image)
        particles = find_particles(color_masks, self.get_minimum_area())
        self.particles = particles
        recalculated = self.recalculate_custom_formulas()
        if self.custom_formulas and recalculated:
            self.formula_status.config(
                text=f"Recalculated {len(self.custom_formulas)} saved formulas.",
                foreground=self.formula_status_default_fg,
            )
        self.clear_particle_selection()
        self.result_label.config(text=f"Total detected particles: {len(particles)}")
        self.update_table(particles)
        self.render_particle_image()

    def draw_particle_labels(self, image, particles):
        """Draw a green outline and yellow ID on every retained particle."""
        draw_particle_labels(image, particles)

    def render_particle_image(self):
        """Render all particle labels and a strong highlight for the selection."""
        outlined = np.asarray(self.original_image, dtype=np.uint8).copy()
        self.draw_particle_labels(outlined, self.particles)

        if self.selected_particle_id is not None:
            particle = self.particles[self.selected_particle_id - 1]
            contour = particle["contour"]
            highlight_width = max(4, int(round(4 / self.scale)))
            cv2.drawContours(
                outlined, [contour], -1, (255, 0, 255), highlight_width
            )
            center = tuple(int(value) for value in particle["center"])
            marker_radius = max(8, int(round(8 / self.scale)))
            cv2.circle(outlined, center, marker_radius, (255, 255, 0), 3)

        result = Image.fromarray(outlined, mode="RGB").resize(
            self.display_image.size, Image.Resampling.LANCZOS
        )
        self.show_image(result)

    def particle_at_display_point(self, display_x, display_y):
        """Return the one-based ID of the particle under a canvas point."""
        image_point = (float(display_x) / self.scale, float(display_y) / self.scale)
        matching_particles = (
            (particle["area"], particle_id)
            for particle_id, particle in enumerate(self.particles, start=1)
            if cv2.pointPolygonTest(particle["contour"], image_point, False) >= 0
        )
        smallest_match = min(matching_particles, default=None)
        return smallest_match[1] if smallest_match is not None else None

    def select_particle(self, particle_id, select_table_row=False):
        """Select one particle and synchronize the image and measurement table."""
        if not 1 <= particle_id <= len(self.particles):
            return
        self.selected_particle_id = particle_id
        self.selected_particle_label.config(text=f"Selected particle: ID {particle_id}")

        if select_table_row:
            row_id = f"particle-{particle_id}"
            if self.table.exists(row_id):
                self.table.selection_set(row_id)
                self.table.focus(row_id)
                self.table.see(row_id)

        self.render_particle_image()

    def table_selection_changed(self, _event=None):
        """Highlight the particle represented by the newly selected table row."""
        if self.updating_table:
            return
        selected_rows = self.table.selection()
        if not selected_rows:
            return
        row_id = selected_rows[0]
        if row_id.startswith("particle-"):
            self.select_particle(int(row_id.removeprefix("particle-")))

    def clear_particle_selection(self):
        """Clear the shared image/table particle selection state."""
        self.selected_particle_id = None
        if hasattr(self, "selected_particle_label"):
            self.selected_particle_label.config(text="Selected particle: —")
        if hasattr(self, "table"):
            self.table.selection_remove(*self.table.selection())

    def update_table(self, particles):
        """Fill the table and show average shape measurements."""
        calibrated = self.micrometers_per_pixel is not None
        dimensional_headings = {
            "area": "Area (µm²)" if calibrated else "Area (px)",
            "perimeter": "Perimeter (µm)" if calibrated else "Perimeter (px)",
            "major_axis": "Major axis (µm)" if calibrated else "Major axis (px)",
            "minor_axis": "Minor axis (µm)" if calibrated else "Minor axis (px)",
        }
        for column, heading in dimensional_headings.items():
            self.table.heading(column, text=heading)

        self.updating_table = True
        try:
            self.clear_table()
            for particle_id, particle in enumerate(particles, start=1):
                if calibrated:
                    length_scale = self.micrometers_per_pixel
                    area_value = particle["area"] * length_scale**2
                    perimeter_value = particle["perimeter"] * length_scale
                    major_axis_value = particle["major_axis"] * length_scale
                    minor_axis_value = particle["minor_axis"] * length_scale
                    dimensional_default = lambda value: f"{value:.6g}"
                else:
                    area_value = particle["area"]
                    perimeter_value = particle["perimeter"]
                    major_axis_value = particle["major_axis"]
                    minor_axis_value = particle["minor_axis"]
                    dimensional_default = None
                area = self.format_table_number(
                    "area",
                    area_value,
                    dimensional_default or (lambda value: str(int(value))),
                )
                perimeter = self.format_table_number(
                    "perimeter",
                    perimeter_value,
                    dimensional_default or (lambda value: f"{value:.2f}"),
                )
                major_axis = self.format_table_number(
                    "major_axis",
                    major_axis_value,
                    dimensional_default or (lambda value: f"{value:.2f}"),
                )
                minor_axis = self.format_table_number(
                    "minor_axis",
                    minor_axis_value,
                    dimensional_default or (lambda value: f"{value:.2f}"),
                )
                circularity = self.format_table_number(
                    "circularity", particle["circularity"],
                    lambda value: f"{value:.3f}",
                )
                aspect_ratio = self.format_table_number(
                    "aspect_ratio", particle["aspect_ratio"],
                    lambda value: f"{value:.3f}",
                )
                custom_displays = []
                for formula in self.custom_formulas:
                    custom_value = formula["values"].get(particle_id)
                    custom_displays.append(
                        self.format_table_number(
                            formula["column_id"], custom_value,
                            lambda value: f"{value:.8g}",
                        ) if custom_value is not None else ""
                    )
                self.table.insert("", "end", iid=f"particle-{particle_id}", values=(
                    particle_id, area, perimeter, major_axis, minor_axis,
                    circularity, aspect_ratio,
                    *custom_displays,
                ))
        finally:
            self.updating_table = False

        if particles:
            average_circularity = sum(p["circularity"] for p in particles) / len(particles)
            average_aspect_ratio = sum(p["aspect_ratio"] for p in particles) / len(particles)
            self.average_label.config(
                text=f"Average Circularity: {average_circularity:.3f}    Average Aspect Ratio: {average_aspect_ratio:.3f}"
            )
        else:
            self.average_label.config(
                text="Average Circularity: —    Average Aspect Ratio: —"
            )

    def clear_table(self):
        """Remove all old measurement rows."""
        for row in self.table.get_children():
            self.table.delete(row)

    def show_image(self, image):
        """Put a Pillow image on the Tkinter canvas."""
        self.photo = ImageTk.PhotoImage(image)
        self.canvas.config(width=image.width, height=image.height)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self.photo)


if __name__ == "__main__":
    window = tk.Tk()
    ImageAnalyzer(window)
    window.mainloop()
