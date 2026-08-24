# Particle Image Analyzer

This beginner-friendly desktop program groups similar RGB colors in recognition
images, then detects connected particles within each color group. It outlines
and numbers each particle and reports its shape measurements.

## Files

- `image_analyzer.py` — the complete application.
- `requirements.txt` — the Python packages the application needs.

Tkinter provides the window and file picker. It is included with most Python
installations. Pillow loads and displays images, NumPy compares their pixels,
and OpenCV detects and measures particles.

## Run the program

Open a terminal in this folder and run:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 image_analyzer.py
```

On Windows, activate the environment with `.venv\Scripts\activate` instead.

### Run the Streamlit web application

The internal-laboratory web interface is implemented separately in
`streamlit_app.py`; the desktop application remains available unchanged.

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run streamlit_app.py
```

Then open `http://localhost:8501` in a browser. The web app supports single and
batch recognition-image analysis, ROI-assisted bracket scale calibration,
custom formulas, configurable display/export precision, CSV downloads, and
annotated-image downloads.

## Use it

1. Click **Open image** and choose an image.
2. Particle analysis starts automatically. You can also click **Analyze all
   particles** to rerun it.
3. Set **Minimum particle area** (in pixels) and press Enter to ignore noise.
4. Read the total and average measurements above the image. Click a particle to
   highlight it and jump to its measurement row, or click a table row to
   highlight the matching particle. The selected particle ID appears above the
   image.
5. Click **Decimal Places…** to set display precision independently for Area,
   Perimeter, Major axis, Minor axis, Circularity, Aspect ratio, and every saved
   custom formula column. Leave a field blank to retain its default formatting.
   These settings affect table display only; calculations continue to use the
   original full-precision measurements.

### Custom formulas

Enter a unique name and formula, then click **Add Formula** to evaluate it for
every detected particle and add a table column with that name. Multiple saved
formulas and result columns can coexist; select one in the saved-formula list
and click **Remove Selected Formula** to delete it. The available variables are
`A` (area), `P` (perimeter), `L` (major axis), `S` (minor axis), `C`
(circularity), and `AR` (aspect ratio). For example, an **Equivalent Diameter**
column can use `sqrt(4*A/pi)`. Supported syntax is limited to safe arithmetic
and approved functions/constants including `sqrt`, `log`, `exp`, `abs`, `sin`,
`cos`, `tan`, `pi`, and `e`.

### Batch analysis

Apply scale calibration, then click **Batch Analysis…** to select multiple
recognition images. Every compatible image is analyzed independently with the
current minimum-area setting, calibration, measurements, and saved custom
formulas. Images whose pixel dimensions differ from the calibration image can
be skipped after a warning. A progress window reports the current image, and
the results window groups particles by image name. From there, export either
one full-precision CSV per image or one combined CSV with an **Image Name**
column. Configurable decimal places affect the batch results display but not the
stored or exported numerical precision.

### Calibrate particle dimensions from a corresponding SEM image

1. Load the recognition image normally with **Open image**.
2. Enter the scale-bar value and choose `nm`, `µm`, or `mm`.
3. Click **Load calibration image** and choose the SEM image with the same pixel
   dimensions and field of view.
4. Click **Select scale-bar region**, then drag a tight rectangle around the
   scale bar. Detection requires its bracket geometry: one horizontal line and
   a vertical line at each endpoint. Unrelated horizontal lines are rejected.
   A magenta line and yellow endpoints are drawn over the calibration image for
   visual verification.
5. Review the displayed pixel length, then click **Apply detected calibration**.
   Check the resulting unit-per-pixel value. When the
   two images have matching dimensions, particle area, perimeter, and axis
   columns are shown in calibrated units. Circularity and aspect ratio are
   unchanged.
6. If ROI detection cannot find the bar, click **Manual 2-point fallback** and
   click its two endpoints. An accepted manual calibration is retained unless
   you explicitly confirm replacing it with a later automatic detection.

Particle segmentation works directly with RGB pixels and groups nearby colors
using an RGB-distance tolerance. The color group most common on the image border
is considered background. Connected components are found separately for every
remaining color group, so differently colored particles stay separate even when
they touch. Regions smaller than the configured minimum area are discarded.
This mode is intended for recognition images in which individual particles are
represented by different colors and the image border mainly contains background.

The original image file is never changed.
