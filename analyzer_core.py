"""Tkinter-free particle analysis and image-processing core."""

import ast
import math

import cv2
import numpy as np
from PIL import Image


__all__ = [
    "PARTICLE_COLOR_TOLERANCE",
    "annotate_particles",
    "detect_horizontal_scale_bar",
    "detect_horizontal_scale_bar_safe",
    "draw_particle_labels",
    "evaluate_custom_formula",
    "find_particles",
    "length_to_micrometers",
    "particle_at_point",
    "particle_formula_variables",
    "particle_measurements",
    "segment_particles",
    "selected_particle_image",
]


PARTICLE_COLOR_TOLERANCE = 30
FORMULA_VARIABLES = {"A", "P", "L", "S", "C", "AR"}
FORMULA_FUNCTIONS = {
    "sqrt": math.sqrt,
    "log": math.log,
    "exp": math.exp,
    "abs": abs,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
}
FORMULA_CONSTANTS = {"pi": math.pi, "e": math.e}
FORMULA_BINARY_OPERATORS = {
    ast.Add: lambda left, right: left + right,
    ast.Sub: lambda left, right: left - right,
    ast.Mult: lambda left, right: left * right,
    ast.Div: lambda left, right: left / right,
    ast.Pow: lambda left, right: left**right,
}
FORMULA_UNARY_OPERATORS = {
    ast.UAdd: lambda value: value,
    ast.USub: lambda value: -value,
}


def evaluate_custom_formula(expression, variables):
    """Safely evaluate one custom formula using approved math syntax only."""
    if not expression.strip():
        raise ValueError("Enter a formula before calculating")
    if len(expression) > 250:
        raise ValueError("Formula is too long")
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as error:
        raise ValueError("Invalid formula syntax") from error

    def evaluate(node):
        if isinstance(node, ast.Expression):
            return evaluate(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
                raise ValueError("Only numeric constants are allowed")
            return float(node.value)
        if isinstance(node, ast.Name):
            if node.id in FORMULA_VARIABLES:
                if node.id not in variables:
                    raise ValueError(f"Missing measurement variable: {node.id}")
                return float(variables[node.id])
            if node.id in FORMULA_CONSTANTS:
                return FORMULA_CONSTANTS[node.id]
            raise ValueError(f"Name is not allowed: {node.id}")
        if isinstance(node, ast.BinOp) and type(node.op) in FORMULA_BINARY_OPERATORS:
            left = evaluate(node.left)
            right = evaluate(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 100:
                raise ValueError("Exponent magnitude must not exceed 100")
            try:
                result = FORMULA_BINARY_OPERATORS[type(node.op)](left, right)
            except (ArithmeticError, OverflowError, ValueError) as error:
                raise ValueError(str(error) or "Invalid arithmetic operation") from error
            if isinstance(result, complex) or not math.isfinite(result):
                raise ValueError("Formula result must be a finite real number")
            return result
        if isinstance(node, ast.UnaryOp) and type(node.op) in FORMULA_UNARY_OPERATORS:
            return FORMULA_UNARY_OPERATORS[type(node.op)](evaluate(node.operand))
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in FORMULA_FUNCTIONS:
                raise ValueError("Function is not allowed")
            if node.keywords:
                raise ValueError("Keyword arguments are not allowed")
            arguments = [evaluate(argument) for argument in node.args]
            try:
                result = FORMULA_FUNCTIONS[node.func.id](*arguments)
            except (TypeError, ArithmeticError, OverflowError, ValueError) as error:
                raise ValueError(
                    f"Invalid {node.func.id}() call: {str(error) or 'bad arguments'}"
                ) from error
            if isinstance(result, complex) or not math.isfinite(result):
                raise ValueError("Formula result must be a finite real number")
            return result
        raise ValueError("Formula contains an operator or construct that is not allowed")

    return float(evaluate(tree))


def _validated_scale_detection_rgb(image):
    """Return a bounded contiguous RGB array suitable for native OpenCV calls."""
    if image is None or not hasattr(image, "size"):
        raise ValueError("Scale-bar ROI is missing or is not an image")
    width, height = image.size
    if not isinstance(width, int) or not isinstance(height, int) or width <= 0 or height <= 0:
        raise ValueError("Scale-bar ROI has invalid dimensions")
    if width < 10 or height < 10:
        raise ValueError("Scale-bar ROI must be at least 10 × 10 pixels")
    if width * height > 4_000_000:
        raise ValueError(
            "Scale-bar ROI is too large for safe detection. "
            "Crop more tightly around the scale bar (maximum 4,000,000 pixels)."
        )

    try:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    except (MemoryError, OSError, ValueError) as error:
        raise ValueError(f"Could not prepare the scale-bar ROI: {error}") from error
    if rgb.size == 0 or rgb.ndim != 3 or rgb.shape != (height, width, 3):
        raise ValueError("Scale-bar ROI contains no valid RGB pixel data")
    return np.ascontiguousarray(rgb)


def detect_horizontal_scale_bar(image):
    """Detect a bracket scale bar: horizontal line with two vertical ends."""
    rgb = _validated_scale_detection_rgb(image)
    height, width = rgb.shape[:2]

    try:
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        if gray is None or gray.size == 0:
            raise ValueError("OpenCV produced an empty grayscale ROI")
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        median = float(np.median(blurred))
        if not math.isfinite(median):
            raise ValueError("Scale-bar ROI has invalid pixel values")
        lower = max(0, int(0.66 * median))
        upper = min(255, max(lower + 1, int(1.33 * median)))
        edges = cv2.Canny(blurred, lower, upper)
        if edges is None or edges.size == 0:
            return None
        minimum_horizontal = max(8, int(round(width * 0.10)))
        minimum_vertical = max(4, int(round(height * 0.04)))
        lines = cv2.HoughLinesP(
            edges,
            1,
            np.pi / 180,
            threshold=max(6, min(minimum_horizontal, minimum_vertical)),
            minLineLength=min(minimum_horizontal, minimum_vertical),
            maxLineGap=max(2, int(round(min(width, height) * 0.02))),
        )
    except cv2.error as error:
        raise RuntimeError(f"OpenCV scale-bar detection failed: {error}") from error
    except MemoryError as error:
        raise RuntimeError("Not enough memory to detect the scale bar") from error

    if lines is None:
        return None
    line_values = np.asarray(lines)
    if line_values.size == 0:
        return None
    if line_values.size % 4:
        raise RuntimeError("OpenCV returned invalid scale-bar line data")
    line_values = line_values.reshape(-1, 4)
    if len(line_values) > 20_000:
        raise RuntimeError(
            "Scale-bar ROI produced too many line candidates. Crop more tightly."
        )

    horizontal_lines = []
    vertical_lines = []
    for x1, y1, x2, y2 in line_values:
        dx = abs(int(x2) - int(x1))
        dy = abs(int(y2) - int(y1))
        if dx >= minimum_horizontal and dy <= max(2, int(round(dx * 0.05))):
            horizontal_lines.append(
                (float(min(x1, x2)), float(max(x1, x2)), (float(y1) + y2) / 2)
            )
        elif dy >= minimum_vertical and dx <= max(2, int(round(dy * 0.15))):
            vertical_lines.append(
                ((float(x1) + x2) / 2, float(min(y1, y2)), float(max(y1, y2)), float(dy))
            )

    if not horizontal_lines or len(vertical_lines) < 2:
        return None

    endpoint_tolerance = max(4.0, width * 0.025)
    intersection_tolerance = max(3.0, height * 0.025)
    best_bracket = None
    tested_combinations = 0
    for horizontal_left, horizontal_right, horizontal_y in horizontal_lines:
        endpoint_matches = []
        for endpoint_x in (horizontal_left, horizontal_right):
            matches = [
                vertical
                for vertical in vertical_lines
                if abs(vertical[0] - endpoint_x) <= endpoint_tolerance
                and vertical[1] - intersection_tolerance <= horizontal_y
                <= vertical[2] + intersection_tolerance
            ]
            endpoint_matches.append(matches)

        tested_combinations += len(endpoint_matches[0]) * len(endpoint_matches[1])
        if tested_combinations > 1_000_000:
            raise RuntimeError(
                "Scale-bar ROI is too complex to analyze safely. Crop more tightly."
            )
        for left_vertical in endpoint_matches[0]:
            for right_vertical in endpoint_matches[1]:
                left_x, right_x = sorted((left_vertical[0], right_vertical[0]))
                pixel_length = right_x - left_x
                if pixel_length < minimum_horizontal * 0.8:
                    continue
                left_error = abs(left_vertical[0] - horizontal_left)
                right_error = abs(right_vertical[0] - horizontal_right)
                alignment = 1.0 - min(
                    1.0, (left_error + right_error) / (2 * endpoint_tolerance)
                )
                vertical_support = min(left_vertical[3], right_vertical[3]) / max(height, 1)
                lower_preference = horizontal_y / max(height - 1, 1)
                score = 5.0 * alignment + 3.0 * vertical_support + lower_preference
                bracket = (score, pixel_length, left_x, right_x, horizontal_y)
                if best_bracket is None or bracket[0] > best_bracket[0]:
                    best_bracket = bracket

    if best_bracket is None:
        return None
    _score, pixel_length, left_x, right_x, horizontal_y = best_bracket
    return pixel_length, ((left_x, horizontal_y), (right_x, horizontal_y))


def _scale_detection_worker(width, height, rgb_bytes, connection):
    """Run native OpenCV detection in an isolated process."""
    try:
        cv2.setNumThreads(1)
        image = Image.frombytes("RGB", (width, height), rgb_bytes)
        connection.send(("ok", detect_horizontal_scale_bar(image)))
    except BaseException as error:
        try:
            connection.send(("error", f"{type(error).__name__}: {error}"))
        except BaseException:
            pass
    finally:
        connection.close()


def detect_horizontal_scale_bar_safe(image, timeout_seconds=20.0):
    """Detect a scale bar without exposing the web-server process to native crashes."""
    rgb = _validated_scale_detection_rgb(image)
    height, width = rgb.shape[:2]

    import multiprocessing

    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(
        target=_scale_detection_worker,
        args=(width, height, rgb.tobytes(), sender),
        daemon=True,
    )
    process_started = False
    try:
        process.start()
        process_started = True
        sender.close()
        if not receiver.poll(timeout_seconds):
            raise TimeoutError(
                "Scale-bar detection timed out. Crop more tightly around the scale bar."
            )
        try:
            status, payload = receiver.recv()
        except EOFError as error:
            raise RuntimeError(
                "The OpenCV scale-bar detector stopped unexpectedly. "
                "Try a smaller, tighter ROI."
            ) from error
        if status == "error":
            raise RuntimeError(f"Scale-bar detection failed: {payload}")
        return payload
    finally:
        receiver.close()
        if not process_started:
            sender.close()
        else:
            process.join(timeout=0.5)
            if process.is_alive():
                process.terminate()
                process.join(timeout=1.0)


def length_to_micrometers(value, unit):
    """Convert a supported scale-bar length to micrometers."""
    normalized = unit.strip().lower().replace("μ", "µ")
    factors = {
        "nm": 0.001,
        "µm": 1.0,
        "um": 1.0,
        "mm": 1000.0,
    }
    if normalized not in factors:
        raise ValueError("Unit must be nm, µm, um, or mm")
    return float(value) * factors[normalized]


def segment_particles(image, color_tolerance=PARTICLE_COLOR_TOLERANCE):
    """Group an RGB recognition image into separate, similar-color masks.

    Colors are first collected in small RGB bins, then nearby bins are joined
    using Euclidean RGB distance.  The color group most common on the image
    border is background.  Keeping every remaining color as its own mask is
    important: differently colored particles remain separate when they touch.
    """
    rgb = np.asarray(image, dtype=np.uint8)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("Particle segmentation requires an RGB image")

    tolerance = max(1, int(color_tolerance))
    # A half-tolerance histogram keeps the number of candidate colors bounded
    # for compressed images, while the distance check supplies the real limit.
    bin_size = max(1, tolerance // 2)
    quantized = (rgb.astype(np.int32) // bin_size).reshape(-1, 3)
    bins, inverse, counts = np.unique(
        quantized, axis=0, return_inverse=True, return_counts=True
    )

    flat_rgb = rgb.reshape(-1, 3).astype(np.float64)
    color_sums = np.zeros((len(bins), 3), dtype=np.float64)
    np.add.at(color_sums, inverse, flat_rgb)
    bin_colors = color_sums / counts[:, None]

    # Seed common colors first so small anti-aliased/JPEG variations attach to
    # the main fill color instead of becoming independent particle classes.
    cluster_centers = []
    cluster_weights = []
    bin_clusters = np.empty(len(bins), dtype=np.int32)
    for bin_index in np.argsort(-counts):
        color = bin_colors[bin_index]
        if cluster_centers:
            distances = np.linalg.norm(np.asarray(cluster_centers) - color, axis=1)
            cluster_index = int(np.argmin(distances))
        else:
            distances = np.empty(0)
            cluster_index = -1

        if cluster_index < 0 or distances[cluster_index] > tolerance:
            cluster_index = len(cluster_centers)
            cluster_centers.append(color.copy())
            cluster_weights.append(int(counts[bin_index]))
        else:
            old_weight = cluster_weights[cluster_index]
            added_weight = int(counts[bin_index])
            cluster_centers[cluster_index] = (
                cluster_centers[cluster_index] * old_weight + color * added_weight
            ) / (old_weight + added_weight)
            cluster_weights[cluster_index] += added_weight
        bin_clusters[bin_index] = cluster_index

    labels = bin_clusters[inverse].reshape(rgb.shape[:2])
    border_labels = np.concatenate(
        (labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1])
    )
    background = int(np.argmax(np.bincount(border_labels)))
    return [
        labels == cluster_index
        for cluster_index in range(len(cluster_centers))
        if cluster_index != background
    ]


def find_particles(color_masks, minimum_area):
    """Measure connected regions independently within each color group."""
    if isinstance(color_masks, np.ndarray):
        color_masks = [color_masks]
    particles = []

    for color_mask in color_masks:
        count, labels, stats, centers = cv2.connectedComponentsWithStats(
            color_mask.astype(np.uint8), connectivity=8
        )
        for label_number in range(1, count):  # Label 0 is not this color.
            area = int(stats[label_number, cv2.CC_STAT_AREA])
            if area < minimum_area:
                continue

            component = (labels == label_number).astype(np.uint8)
            contours, _ = cv2.findContours(
                component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )
            if not contours:
                continue
            contour = max(contours, key=cv2.contourArea)
            perimeter = float(cv2.arcLength(contour, True))

            # Ellipse diameters describe the particle's major and minor axes.
            if len(contour) >= 5:
                _center, axes, _angle = cv2.fitEllipse(contour)
                minor_axis, major_axis = sorted(float(value) for value in axes)
            else:
                _x, _y, width, height = cv2.boundingRect(contour)
                minor_axis, major_axis = sorted((float(width), float(height)))

            circularity = 4 * math.pi * area / perimeter**2 if perimeter else 0.0
            aspect_ratio = minor_axis / major_axis if major_axis else 0.0
            particles.append(
                {
                    "area": area,
                    "perimeter": perimeter,
                    "major_axis": major_axis,
                    "minor_axis": minor_axis,
                    "circularity": circularity,
                    "aspect_ratio": aspect_ratio,
                    "center": tuple(centers[label_number]),
                    "contour": contour,
                }
            )

    # Give particles predictable IDs: top to bottom, then left to right.
    particles.sort(key=lambda item: (item["center"][1], item["center"][0]))
    return particles



def particle_measurements(particle, length_scale=None):
    """Return one particle's shape measurements in pixels or calibrated units."""
    scale = length_scale or 1.0
    return {
        "area": particle["area"] * scale**2,
        "perimeter": particle["perimeter"] * scale,
        "major_axis": particle["major_axis"] * scale,
        "minor_axis": particle["minor_axis"] * scale,
        "circularity": particle["circularity"],
        "aspect_ratio": particle["aspect_ratio"],
    }


def particle_formula_variables(particle, length_scale=None):
    """Return the fixed custom-formula variables in the current measurement units."""
    measurements = particle_measurements(particle, length_scale)
    return {
        "A": measurements["area"],
        "P": measurements["perimeter"],
        "L": measurements["major_axis"],
        "S": measurements["minor_axis"],
        "C": measurements["circularity"],
        "AR": measurements["aspect_ratio"],
    }


def draw_particle_labels(image_array, particles):
    """Draw the established green contours and yellow one-based particle IDs."""
    for particle_id, particle in enumerate(particles, start=1):
        cv2.drawContours(image_array, [particle["contour"]], -1, (0, 255, 0), 2)
        center_x, center_y = (int(value) for value in particle["center"])
        cv2.putText(
            image_array, str(particle_id), (center_x, center_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA,
        )
    return image_array


def annotate_particles(image, particles):
    """Return an RGB Pillow image annotated with contours and particle IDs."""
    annotated = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    draw_particle_labels(annotated, particles)
    return Image.fromarray(annotated, mode="RGB")


def selected_particle_image(image, particles, particle_id):
    return image


def particle_at_point(particles, x, y):
    """Return the one-based ID of the smallest particle containing ``(x, y)``.

    This mirrors the desktop selection rule: contours are tested in original-image
    coordinates, and overlap is resolved by choosing the matching particle with the
    smallest measured area. ``None`` is returned when the point is outside every
    detected contour.
    """
    image_point = (float(x), float(y))
    matching_particles = []
    for particle_id, particle in enumerate(particles, start=1):
        if cv2.pointPolygonTest(particle["contour"], image_point, False) >= 0:
            matching_particles.append((particle["area"], particle_id))

    smallest_match = min(matching_particles, default=None)
    return smallest_match[1] if smallest_match is not None else None
