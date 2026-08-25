"""Streamlit web interface for the particle image analyzer.

Run with: streamlit run streamlit_app.py
"""

from __future__ import annotations

import csv
import io
from pathlib import Path
import zipfile

import pandas as pd
from PIL import Image, ImageDraw
import streamlit as st
from streamlit_cropper import st_cropper
from streamlit_image_coordinates import streamlit_image_coordinates

from analyzer_core import (
    annotate_particles,
    detect_horizontal_scale_bar_safe,
    evaluate_custom_formula,
    find_particles,
    length_to_micrometers,
    particle_at_point,
    particle_formula_variables,
    particle_measurements,
    segment_particles,
    selected_particle_image,
)


STANDARD_COLUMNS = (
    "Area",
    "Perimeter",
    "Major axis",
    "Minor axis",
    "Circularity",
    "Aspect ratio",
)
DEFAULT_DECIMALS = {
    "Area": None,
    "Perimeter": None,
    "Major axis": None,
    "Minor axis": None,
    "Circularity": None,
    "Aspect ratio": None,
}
UNIT_FACTORS = {"nm": 0.001, "µm": 1.0, "mm": 1000.0}


def initialize_state() -> None:
    defaults = {
        "recognition_key": None,
        "recognition_image": None,
        "recognition_name": None,
        "single_result": None,
        "single_selected_particle_id": None,
        "single_last_image_click": None,
        "single_last_table_selection": (),
        "calibration_key": None,
        "calibration_image": None,
        "pending_detection": None,
        "detected_pixel_length_input": None,
        "micrometers_per_pixel": None,
        "calibration_pixel_length": None,
        "calibration_source": None,
        "calibration_physical_value": None,
        "calibration_unit": None,
        "custom_formulas": [],
        "decimal_places": DEFAULT_DECIMALS.copy(),
        "batch_results": [],
        "batch_formulas": [],
        "batch_images": [],
        "batch_selected_index": 0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def uploaded_image(uploaded_file) -> tuple[Image.Image, bytes]:
    data = uploaded_file.getvalue()
    image = Image.open(io.BytesIO(data)).convert("RGB")
    return image, data



def image_png_bytes(image: Image.Image) -> bytes:
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def active_scale_for_size(image_size: tuple[int, int]) -> float | None:
    calibration_image = st.session_state.calibration_image
    scale = st.session_state.micrometers_per_pixel
    if calibration_image is None or scale is None:
        return None
    return scale if calibration_image.size == image_size else None


def measurement_row(
    particle_id: int,
    particle: dict,
    length_scale: float | None,
    formulas: list[dict],
    formula_errors: set[str] | None = None,
) -> dict:
    measurements = particle_measurements(particle, length_scale)
    row = {
        "Particle ID": particle_id,
        "Area": measurements["area"],
        "Perimeter": measurements["perimeter"],
        "Major axis": measurements["major_axis"],
        "Minor axis": measurements["minor_axis"],
        "Circularity": measurements["circularity"],
        "Aspect ratio": measurements["aspect_ratio"],
    }
    variables = particle_formula_variables(particle, length_scale)
    for formula in formulas:
        try:
            row[formula["name"]] = evaluate_custom_formula(
                formula["expression"], variables
            )
        except ValueError as error:
            row[formula["name"]] = None
            if formula_errors is not None:
                formula_errors.add(f'{formula["name"]}: {error}')
    return row


def analyze_image(
    image: Image.Image,
    minimum_area: int,
    length_scale: float | None,
    formulas: list[dict],
) -> dict:
    particles = find_particles(segment_particles(image), minimum_area)
    formula_errors: set[str] = set()
    rows = [
        measurement_row(
            particle_id, particle, length_scale, formulas, formula_errors
        )
        for particle_id, particle in enumerate(particles, start=1)
    ]
    return {
        "particles": particles,
        "rows": rows,
        "particle_count": len(particles),
        "annotated_image": None,
        "calibrated": length_scale is not None,
        "formula_errors": sorted(formula_errors),
    }


def display_headers(calibrated: bool, formulas: list[dict]) -> dict[str, str]:
    headers = {
        "Particle ID": "Particle ID",
        "Area": "Area (µm²)" if calibrated else "Area (px)",
        "Perimeter": "Perimeter (µm)" if calibrated else "Perimeter (px)",
        "Major axis": "Major axis (µm)" if calibrated else "Major axis (px)",
        "Minor axis": "Minor axis (µm)" if calibrated else "Minor axis (px)",
        "Circularity": "Circularity",
        "Aspect ratio": "Aspect ratio",
    }
    headers.update({formula["name"]: formula["name"] for formula in formulas})
    return headers


def rounded_rows(
    rows: list[dict], calibrated: bool, formulas: list[dict],
    include_image_name: bool = False,
) -> tuple[list[str], list[list]]:
    keys = ["Particle ID", *STANDARD_COLUMNS, *[f["name"] for f in formulas]]
    headers_map = display_headers(calibrated, formulas)
    headers = [headers_map[key] for key in keys]
    if include_image_name:
        headers.insert(0, "Image Name")
    output_rows = []
    decimals = st.session_state.decimal_places
    for source in rows:
        output = []
        if include_image_name:
            output.append(source["Image Name"])
        output.append(source["Particle ID"])
        for key in keys[1:]:
            value = source.get(key)
            places = decimals.get(key)
            if value is None:
                display = ""
            elif places is not None:
                display = f"{float(value):.{int(places)}f}"
            elif key == "Area" and not calibrated:
                display = str(int(value))
            elif key in ("Area", "Perimeter", "Major axis", "Minor axis") and calibrated:
                display = f"{float(value):.6g}"
            elif key in ("Perimeter", "Major axis", "Minor axis"):
                display = f"{float(value):.2f}"
            elif key in ("Circularity", "Aspect ratio"):
                display = f"{float(value):.3f}"
            else:
                display = f"{float(value):.8g}"
            output.append(display)
        output_rows.append(output)
    return headers, output_rows


def dataframe_for_display(
    rows: list[dict], calibrated: bool, formulas: list[dict],
    include_image_name: bool = False,
) -> pd.DataFrame:
    headers, values = rounded_rows(rows, calibrated, formulas, include_image_name)
    return pd.DataFrame(values, columns=headers)


def csv_bytes(
    rows: list[dict], calibrated: bool, formulas: list[dict],
    include_image_name: bool = False,
) -> bytes:
    keys = ["Particle ID", *STANDARD_COLUMNS, *[f["name"] for f in formulas]]
    headers_map = display_headers(calibrated, formulas)
    headers = [headers_map[key] for key in keys]
    if include_image_name:
        headers.insert(0, "Image Name")
    output = io.StringIO(newline="")
    writer = csv.writer(output)
    writer.writerow(headers)
    for row in rows:
        values = [row.get(key, "") for key in keys]
        if include_image_name:
            values.insert(0, row["Image Name"])
        writer.writerow(values)
    return output.getvalue().encode("utf-8-sig")


def calibration_overlay(
    image: Image.Image, roi: dict, endpoints: tuple[tuple[float, float], tuple[float, float]]
) -> Image.Image:
    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)
    left, top = int(roi["left"]), int(roi["top"])
    right = left + int(roi["width"])
    bottom = top + int(roi["height"])
    draw.rectangle((left, top, right, bottom), outline="cyan", width=3)
    first, second = endpoints
    draw.line((*first, *second), fill="magenta", width=4)
    radius = max(4, round(min(image.size) * 0.006))
    for x, y in endpoints:
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            fill="yellow",
            outline="magenta",
            width=2,
        )
    return overlay


def render_calibration_panel() -> None:
    st.subheader("Scale calibration")
    calibration_upload = st.file_uploader(
        "Calibration SEM image", type=["png", "jpg", "jpeg", "bmp", "tif", "tiff", "webp"],
        key="calibration_upload",
    )
    if calibration_upload is None:
        st.info("Upload the corresponding SEM image to calibrate particle dimensions.")
        return
    try:
        image, data = uploaded_image(calibration_upload)
    except (OSError, ValueError) as error:
        st.error(f"Could not read calibration image: {error}")
        return
    calibration_key = (calibration_upload.name, len(data), hash(data))
    if calibration_key != st.session_state.calibration_key:
        st.session_state.calibration_key = calibration_key
        st.session_state.calibration_image = image
        st.session_state.pending_detection = None
        st.session_state.detected_pixel_length_input = None
        st.session_state.micrometers_per_pixel = None
        st.session_state.calibration_pixel_length = None
        st.session_state.calibration_source = None
        st.session_state.calibration_physical_value = None
        st.session_state.calibration_unit = None
        st.session_state.single_result = None

    value_column, unit_column = st.columns([2, 1])
    scale_value = value_column.number_input(
        "Scale-bar physical value", min_value=0.000001, value=3.0, format="%.6f"
    )
    scale_unit = unit_column.selectbox("Unit", ["nm", "µm", "mm"], index=1)
    st.caption("Drag a tight ROI around the bracket-style scale bar.")
    roi = st_cropper(
        image,
        realtime_update=True,
        box_color="#00ffff",
        aspect_ratio=None,
        return_type="box",
    )
    if st.button("Detect bracket scale bar", type="primary"):
        left = max(0, int(roi["left"]))
        top = max(0, int(roi["top"]))
        right = min(image.width, left + max(0, int(roi["width"])))
        bottom = min(image.height, top + max(0, int(roi["height"])))
        roi_width = right - left
        roi_height = bottom - top
        detection = None
        detection_error_reported = False
        if roi_width < 10 or roi_height < 10:
            st.session_state.pending_detection = None
            st.session_state.detected_pixel_length_input = None
            st.error("Scale-bar ROI must be at least 10 × 10 pixels.")
            detection_error_reported = True
        else:
            try:
                roi_image = image.crop((left, top, right, bottom))
                detection = detect_horizontal_scale_bar_safe(roi_image)
            except (MemoryError, OSError, RuntimeError, TimeoutError, ValueError) as error:
                st.session_state.pending_detection = None
                st.session_state.detected_pixel_length_input = None
                st.error(f"Scale-bar detection could not be completed: {error}")
                detection_error_reported = True
        if detection is None and not detection_error_reported:
            st.session_state.pending_detection = None
            st.session_state.detected_pixel_length_input = None
            st.error(
                "No bracket scale bar found. The ROI must contain one horizontal line "
                "with a vertical line at both endpoints."
            )
        elif detection is not None:
            pixel_length, (roi_first, roi_second) = detection
            endpoints = (
                (left + roi_first[0], top + roi_first[1]),
                (left + roi_second[0], top + roi_second[1]),
            )
            st.session_state.pending_detection = {
                "pixel_length": pixel_length,
                "endpoints": endpoints,
                "roi": {"left": left, "top": top, "width": right-left, "height": bottom-top},
            }
            st.session_state.detected_pixel_length_input = float(pixel_length)

    pending = st.session_state.pending_detection
    if pending:
        st.image(
            calibration_overlay(image, pending["roi"], pending["endpoints"]),
            caption="Detected bracket: magenta bar with yellow endpoints",
            width="stretch",
        )
        st.caption(
            f'Automatically detected length: {pending["pixel_length"]:.15g} px'
        )
        edited_pixel_length = st.number_input(
            "Scale-bar pixel length (editable)",
            min_value=0.000000000001,
            step=1.0,
            format="%.15g",
            key="detected_pixel_length_input",
            help="Review the detected length and edit it before applying calibration.",
        )
        if st.button("Apply detected calibration"):
            if edited_pixel_length is None or edited_pixel_length <= 0:
                st.error("Scale-bar pixel length must be greater than zero.")
                return
            recognition = st.session_state.recognition_image
            if recognition is not None and recognition.size != image.size:
                st.error(
                    f"Image dimensions differ: recognition {recognition.width}×{recognition.height} px; "
                    f"calibration {image.width}×{image.height} px. Calibration was not applied."
                )
            else:
                applied_pixel_length = float(edited_pixel_length)
                micrometers = length_to_micrometers(scale_value, scale_unit)
                st.session_state.micrometers_per_pixel = (
                    micrometers / applied_pixel_length
                )
                st.session_state.calibration_pixel_length = applied_pixel_length
                st.session_state.calibration_source = "automatic"
                st.session_state.calibration_physical_value = scale_value
                st.session_state.calibration_unit = scale_unit
                st.session_state.single_result = None
                st.success("Calibration applied using the reviewed pixel length.")

    if st.session_state.micrometers_per_pixel is not None:
        applied_value = st.session_state.calibration_physical_value
        applied_unit = st.session_state.calibration_unit
        unit_value = applied_value / st.session_state.calibration_pixel_length
        st.success(
            f"Active calibration: {st.session_state.calibration_pixel_length:.6g} px; "
            f"{unit_value:.8g} {applied_unit}/pixel "
            f"({st.session_state.micrometers_per_pixel:.8g} µm/pixel)"
        )


def render_formula_panel() -> None:
    st.subheader("Custom formulas")
    st.caption(
        "A=Area, P=Perimeter, L=Major axis, S=Minor axis, "
        "C=Circularity, AR=Aspect ratio"
    )
    with st.form("add_formula", clear_on_submit=True):
        name_col, formula_col = st.columns([1, 2])
        name = name_col.text_input("Name", placeholder="Equivalent Diameter")
        expression = formula_col.text_input("Formula", placeholder="sqrt(4*A/pi)")
        submitted = st.form_submit_button("Add Formula")
    if submitted:
        normalized = name.strip().casefold()
        reserved = {column.casefold() for column in ("Particle ID", *STANDARD_COLUMNS, "Image Name")}
        existing = {formula["name"].casefold() for formula in st.session_state.custom_formulas}
        if not name.strip():
            st.error("Enter a formula name.")
        elif normalized in existing or normalized in reserved:
            st.error("Formula names must be unique and must not duplicate a standard column.")
        elif not st.session_state.single_result:
            st.error("Upload and analyze a recognition image before adding a formula.")
        else:
            result = st.session_state.single_result
            try:
                calculated = {}
                for row in result["rows"]:
                    variables = {
                        "A": row["Area"], "P": row["Perimeter"],
                        "L": row["Major axis"], "S": row["Minor axis"],
                        "C": row["Circularity"], "AR": row["Aspect ratio"],
                    }
                    try:
                        calculated[row["Particle ID"]] = evaluate_custom_formula(
                            expression, variables
                        )
                    except ValueError as error:
                        raise ValueError(
                            f'Particle {row["Particle ID"]}: {error}'
                        ) from error
            except ValueError as error:
                st.error(f"Invalid formula: {error}")
            else:
                formula = {"name": name.strip(), "expression": expression.strip()}
                st.session_state.custom_formulas.append(formula)
                st.session_state.decimal_places[name.strip()] = None
                for row in result["rows"]:
                    row[name.strip()] = calculated[row["Particle ID"]]
                result["formulas"].append(formula.copy())
                result["analysis_signature"] = None
                st.success(
                    f'Added "{name.strip()}" for {len(calculated)} particles.'
                )

    formulas = st.session_state.custom_formulas
    if formulas:
        options = [f'{formula["name"]} = {formula["expression"]}' for formula in formulas]
        selected = st.selectbox("Saved formulas", options)
        if st.button("Remove selected formula"):
            index = options.index(selected)
            removed = formulas.pop(index)
            st.session_state.decimal_places.pop(removed["name"], None)
            result = st.session_state.single_result
            if result:
                for row in result["rows"]:
                    row.pop(removed["name"], None)
                result["formulas"] = [
                    formula for formula in result["formulas"]
                    if formula["name"] != removed["name"]
                ]
                result["analysis_signature"] = None
            st.success(f'Removed "{removed["name"]}".')
            st.rerun()
    else:
        st.info("No custom formulas saved.")


def render_decimal_panel() -> None:
    st.subheader("Table decimal places")
    st.caption("Rounding affects table display only; CSV exports retain full precision.")
    columns = [*STANDARD_COLUMNS, *[f["name"] for f in st.session_state.custom_formulas]]
    grid = st.columns(3)
    for index, column in enumerate(columns):
        current = st.session_state.decimal_places.get(column)
        st.session_state.decimal_places[column] = grid[index % 3].number_input(
            column,
            min_value=0,
            max_value=12,
            value=current,
            step=1,
            placeholder="Default",
            key=f"decimal_{column}",
        )


def render_single_analysis(minimum_area: int) -> None:
    st.subheader("Single-image particle analysis")
    uploaded = st.file_uploader(
        "Recognition image", type=["png", "jpg", "jpeg", "bmp", "tif", "tiff", "webp"],
        key="recognition_upload",
    )
    if uploaded is None:
        st.info("Upload a recognition image to begin.")
        return
    try:
        image, data = uploaded_image(uploaded)
    except (OSError, ValueError) as error:
        st.error(f"Could not read recognition image: {error}")
        return
    image_key = (uploaded.name, len(data), hash(data))
    if image_key != st.session_state.recognition_key:
        st.session_state.recognition_key = image_key
        st.session_state.recognition_image = image
        st.session_state.recognition_name = uploaded.name
        st.session_state.single_result = None
        st.session_state.single_selected_particle_id = None
        st.session_state.single_last_image_click = None
        st.session_state.single_last_table_selection = ()
        st.session_state.pop("single_results_table", None)

    calibration_image = st.session_state.calibration_image
    if calibration_image is not None and calibration_image.size != image.size:
        st.warning(
            f"Calibration image is {calibration_image.width}×{calibration_image.height} px, "
            f"but this image is {image.width}×{image.height} px. Results will remain in pixels."
        )
    scale = active_scale_for_size(image.size)
    formula_signature = tuple(
        (formula["name"], formula["expression"])
        for formula in st.session_state.custom_formulas
    )
    analysis_signature = (image_key, minimum_area, scale, formula_signature)
    analyze_requested = st.button("Analyze all particles", type="primary")
    current_result = st.session_state.single_result
    analysis_needed = (
        current_result is None
        or current_result.get("analysis_signature") != analysis_signature
    )
    if analyze_requested or analysis_needed:
        try:
            with st.spinner("Segmenting and measuring particles…"):
                result = analyze_image(
                    image, minimum_area, scale, st.session_state.custom_formulas
                )
        except ValueError as error:
            st.error(f"Analysis failed: {error}")
        else:
            result["image_name"] = uploaded.name
            result["formulas"] = [formula.copy() for formula in st.session_state.custom_formulas]
            result["minimum_area"] = minimum_area
            result["analysis_signature"] = analysis_signature
            st.session_state.single_result = result
            st.session_state.single_selected_particle_id = None
            st.session_state.single_last_image_click = None
            st.session_state.single_last_table_selection = ()
            st.session_state.pop("single_results_table", None)

    result = st.session_state.single_result
    if not result:
        return

    table_state = st.session_state.get("single_results_table", {})
    selected_rows = tuple(table_state.get("selection", {}).get("rows", []))
    if selected_rows != st.session_state.single_last_table_selection:
        st.session_state.single_last_table_selection = selected_rows
        selected_particle_id = selected_rows[0] + 1 if selected_rows else None
        if selected_particle_id and selected_particle_id <= result["particle_count"]:
            st.session_state.single_selected_particle_id = selected_particle_id
        elif not selected_rows:
            st.session_state.single_selected_particle_id = None

    count_col, circularity_col, aspect_col = st.columns(3)
    count_col.metric("Detected particles", result["particle_count"])
    if result["particles"]:
        average_circularity = sum(
            particle["circularity"] for particle in result["particles"]
        ) / result["particle_count"]
        average_aspect_ratio = sum(
            particle["aspect_ratio"] for particle in result["particles"]
        ) / result["particle_count"]
        circularity_col.metric("Average circularity", f"{average_circularity:.3f}")
        aspect_col.metric("Average aspect ratio", f"{average_aspect_ratio:.3f}")
    else:
        circularity_col.metric("Average circularity", "—")
        aspect_col.metric("Average aspect ratio", "—")

    selected_particle_id = st.session_state.single_selected_particle_id
    st.caption(
        f"Selected particle: ID {selected_particle_id}"
        if selected_particle_id is not None
        else "Selected particle: —"
    )
    st.caption("Click a detected particle in the image or select its table row.")
    click = streamlit_image_coordinates(
        selected_particle_image(image, result["particles"], selected_particle_id),
        width="content",
        key=f"single_particle_image_{abs(hash(image_key))}",
        cursor="crosshair",
    )
    if click:
        click_token = click.get("unix_time", (click.get("x"), click.get("y")))
        if click_token != st.session_state.single_last_image_click:
            st.session_state.single_last_image_click = click_token
            displayed_width = max(1, int(click.get("width", image.width)))
            displayed_height = max(1, int(click.get("height", image.height)))
            image_x = float(click["x"]) * image.width / displayed_width
            image_y = float(click["y"]) * image.height / displayed_height
            clicked_particle_id = particle_at_point(
                result["particles"], image_x, image_y
            )
            if clicked_particle_id is not None:
                st.session_state.single_selected_particle_id = clicked_particle_id
                st.rerun()
    if result["formula_errors"]:
        st.warning(
            "Some custom formula values are blank:\n\n"
            + "\n\n".join(result["formula_errors"])
        )
    results_frame = dataframe_for_display(
        result["rows"], result["calibrated"], result["formulas"]
    )
    selected_particle_id = st.session_state.single_selected_particle_id
    if selected_particle_id is not None:
        selected_row = selected_particle_id - 1
        results_display = results_frame.style.apply(
            lambda row: [
                "background-color: #ff00ff; color: #ffffff; font-weight: bold"
                if row.name == selected_row else ""
                for _value in row
            ],
            axis=1,
        )
    else:
        results_display = results_frame
    st.dataframe(
        results_display,
        width="stretch",
        height=420,
        hide_index=True,
        key="single_results_table",
        on_select="rerun",
        selection_mode="single-row",
    )
    csv_data = csv_bytes(result["rows"], result["calibrated"], result["formulas"])
    first, second = st.columns(2)
    first.download_button(
        "Download results CSV",
        csv_data,
        file_name=f"{Path(result['image_name']).stem}_particles.csv",
        mime="text/csv",
    )



def render_batch_analysis(minimum_area: int) -> None:
    st.subheader("Batch analysis")
    scale = st.session_state.micrometers_per_pixel
    calibration_image = st.session_state.calibration_image
    uploads = st.file_uploader(
        "Recognition images",
        type=["png", "jpg", "jpeg", "bmp", "tif", "tiff", "webp"],
        accept_multiple_files=True,
        key="batch_uploads",
    )

    # The uploader reflects only its current widget value. Copy each new image into
    # session state so removing it from the uploader, or uploading more files later,
    # cannot discard images or results already collected by the browser.
    known_keys = {item["upload_key"] for item in st.session_state.batch_images}
    for uploaded in uploads or []:
        try:
            image, data = uploaded_image(uploaded)
        except (OSError, ValueError) as error:
            st.warning(f"{uploaded.name}: unreadable ({error})")
            continue
        upload_key = (uploaded.name, len(data), hash(data))
        if upload_key not in known_keys:
            st.session_state.batch_images.append(
                {
                    "upload_key": upload_key,
                    "image_name": uploaded.name,
                    "image": image,
                    "result": None,
                }
            )
            known_keys.add(upload_key)

    items = st.session_state.batch_images
    if not items:
        st.info("Upload recognition images to build the multi-image browser.")
        return

    labels = [
        f"Image {index} — {item['image_name']}"
        for index, item in enumerate(items, start=1)
    ]
    selected_index = min(st.session_state.batch_selected_index, len(items) - 1)

    def select_batch_image() -> None:
        st.session_state.batch_selected_index = labels.index(
            st.session_state.batch_image_selector
        )

    def move_batch_image(offset: int) -> None:
        new_index = max(
            0,
            min(len(items) - 1, st.session_state.batch_selected_index + offset),
        )
        st.session_state.batch_selected_index = new_index
        st.session_state.batch_image_selector = labels[new_index]

    if st.session_state.get("batch_image_selector") not in labels:
        st.session_state.batch_image_selector = labels[selected_index]
    selected_label = st.selectbox(
        "Image selector",
        labels,
        key="batch_image_selector",
        on_change=select_batch_image,
    )
    selected_index = labels.index(selected_label)
    st.session_state.batch_selected_index = selected_index

    previous_col, position_col, next_col = st.columns([1, 2, 1])
    previous_col.button(
        "Previous",
        icon=":material/arrow_back:",
        disabled=selected_index == 0,
        on_click=move_batch_image,
        args=(-1,),
    )
    position_col.caption(f"Image {selected_index + 1} of {len(items)}")
    next_col.button(
        "Next",
        icon=":material/arrow_forward:",
        disabled=selected_index == len(items) - 1,
        on_click=move_batch_image,
        args=(1,),
    )

    if scale is None or calibration_image is None:
        st.warning("Apply scale calibration before analyzing pending images.")
        st.image(
            items[selected_index]["image"],
            caption=f"Image {selected_index + 1}: {items[selected_index]['image_name']}",
            width="stretch",
        )
        return

    skip_incompatible = st.checkbox("Skip incompatible image dimensions", value=True)
    pending_items = [item for item in items if item["result"] is None]
    if st.button("Analyze pending images", disabled=not pending_items, type="primary"):
        compatible = []
        incompatible = []
        for item in pending_items:
            image = item["image"]
            if image.size != calibration_image.size:
                incompatible.append(
                    f"{item['image_name']}: {image.width}×{image.height} px "
                    f"(expected {calibration_image.width}×{calibration_image.height})"
                )
            else:
                compatible.append(item)
        if incompatible:
            st.warning("Incompatible images:\n\n" + "\n\n".join(incompatible))
        if incompatible and not skip_incompatible:
            st.error("Batch stopped. Enable skipping or provide compatible images.")
        elif not compatible:
            st.error("No compatible images to analyze.")
        else:
            progress = st.progress(0, text="Starting batch…")
            formulas = [formula.copy() for formula in st.session_state.custom_formulas]
            calibration = {
                "micrometers_per_pixel": scale,
                "pixel_length": st.session_state.calibration_pixel_length,
                "source": st.session_state.calibration_source,
                "physical_value": st.session_state.calibration_physical_value,
                "unit": st.session_state.calibration_unit,
            }
            for index, item in enumerate(compatible, start=1):
                name = item["image_name"]
                progress.progress(
                    (index - 1) / len(compatible),
                    text=f"Analyzing {index} of {len(compatible)}: {name}",
                )
                try:
                    result = analyze_image(item["image"], minimum_area, scale, formulas)
                except ValueError as error:
                    st.error(f"{name} failed: {error}")
                    continue
                result["image_name"] = name
                result["formulas"] = [formula.copy() for formula in formulas]
                result["minimum_area"] = minimum_area
                result["calibration"] = calibration.copy()
                item["result"] = result
                progress.progress(index / len(compatible), text=f"Completed {name}")
            progress.empty()

    results = [item["result"] for item in items if item["result"] is not None]
    # Keep the legacy state fields synchronized for compatibility with existing
    # sessions and any downstream code, while the browser uses per-image state.
    st.session_state.batch_results = results
    if not results:
        st.image(
            items[selected_index]["image"],
            caption=f"Image {selected_index + 1}: {items[selected_index]['image_name']}",
            width="stretch",
        )
        st.info(f"{len(pending_items)} image(s) are waiting to be analyzed.")
        return
    st.success(
        f"Stored results: {len(results)} of {len(items)} images, "
        f"{sum(result['particle_count'] for result in results)} particles."
    )
    selected_item = items[selected_index]
    selected_result = selected_item["result"]
    if selected_result is None:
        st.image(
            selected_item["image"],
            caption=f"Image {selected_index + 1}: {selected_item['image_name']}",
            width="stretch",
        )
        st.info("This image is waiting to be analyzed.")
        return
    st.metric("Particle count", selected_result["particle_count"])
    if selected_result["particles"]:
        average_circularity = sum(
            particle["circularity"] for particle in selected_result["particles"]
        ) / selected_result["particle_count"]
        average_aspect_ratio = sum(
            particle["aspect_ratio"] for particle in selected_result["particles"]
        ) / selected_result["particle_count"]
        st.caption(
            f"Average circularity: {average_circularity:.3f} · "
            f"Average aspect ratio: {average_aspect_ratio:.3f}"
        )
    
    if selected_result["formula_errors"]:
        st.warning(
            "Some custom formula values are blank:\n\n"
            + "\n\n".join(selected_result["formula_errors"])
        )
    st.dataframe(
        dataframe_for_display(
            selected_result["rows"], selected_result["calibrated"],
            selected_result["formulas"],
        ),
        width="stretch",
        height=420,
    )

    combined_rows = []
    for result in results:
        for row in result["rows"]:
            combined_rows.append({"Image Name": result["image_name"], **row})
    combined_formulas = []
    formula_names = set()
    for result in results:
        for formula in result["formulas"]:
            if formula["name"] not in formula_names:
                combined_formulas.append(formula)
                formula_names.add(formula["name"])
    combined_csv = csv_bytes(
        combined_rows, True, combined_formulas,
        include_image_name=True,
    )
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        used_names = set()
        for result in results:
            stem = Path(result["image_name"]).stem or "image"
            candidate = stem
            suffix = 2
            while candidate.casefold() in used_names:
                candidate = f"{stem}_{suffix}"
                suffix += 1
            used_names.add(candidate.casefold())
            bundle.writestr(
                f"{candidate}.csv",
                csv_bytes(
                    result["rows"], result["calibrated"], result["formulas"]
                ),
            )
    first, second = st.columns(2)
    first.download_button(
        "Download combined CSV",
        combined_csv,
        file_name="combined_particle_results.csv",
        mime="text/csv",
    )
    second.download_button(
        "Download one CSV per image (ZIP)",
        archive.getvalue(),
        file_name="particle_results_by_image.zip",
        mime="application/zip",
    )


def main() -> None:
    st.set_page_config(
        page_title="Laboratory Particle Image Analyzer",
        page_icon="🔬",
        layout="wide",
    )
    initialize_state()
    st.title("Laboratory Particle Image Analyzer")
    st.caption(
        "Particle segmentation, calibrated shape measurements, custom formulas, and batch export"
    )
    minimum_area = st.sidebar.number_input(
        "Minimum particle area (pixels)", min_value=1, value=100, step=1
    )
    st.sidebar.info(
        "All calculations and CSV exports retain full precision. Decimal settings affect table display only."
    )

    calibration_tab, formulas_tab, decimals_tab = st.tabs(
        ["Scale Calibration", "Custom Formulas", "Decimal Places"]
    )
    with calibration_tab:
        render_calibration_panel()
    with formulas_tab:
        render_formula_panel()
    with decimals_tab:
        render_decimal_panel()

    single_tab, batch_tab = st.tabs(["Single Image", "Batch Analysis"])
    with single_tab:
        render_single_analysis(int(minimum_area))
    with batch_tab:
        render_batch_analysis(int(minimum_area))


if __name__ == "__main__":
    main()
