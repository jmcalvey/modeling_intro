"""
barycentric_triangle.py

Visualizes a point within a barycentric "Analytical / Data-Driven /
Physical" (A/D/P) triangle, where each vertex is colored red/green/blue and
interior pixels are shaded by distance to each vertex.

Library usage:
    from barycentric_triangle import generate_triangle
    generate_triangle(point=[(0.5, 0.3, 0.2)])

CLI usage:
    python barycentric_triangle.py
    python barycentric_triangle.py output.png
    python barycentric_triangle.py output.png 0.5 0.3 0.2
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

WIDTH = 800
HEIGHT = 700
DEFAULT_OUTPUT = "./images/triangle_plot.png"


def _point_in_triangle(p, a, b, c):
    """Return True if point p lies inside triangle abc (barycentric test)."""
    v0 = c - a
    v1 = b - a
    v2 = p - a

    dot00 = np.dot(v0, v0)
    dot01 = np.dot(v0, v1)
    dot02 = np.dot(v0, v2)
    dot11 = np.dot(v1, v1)
    dot12 = np.dot(v1, v2)

    denom = dot00 * dot11 - dot01 * dot01
    if denom == 0:
        return False

    u = (dot11 * dot02 - dot01 * dot12) / denom
    v = (dot00 * dot12 - dot01 * dot02) / denom

    return (u >= 0) and (v >= 0) and (u + v <= 1)


def _build_triangle_image(red_point, green_point, blue_point, width, height):
    """Render the RGB-gradient triangle as a (height, width, 3) float image."""
    image = np.ones((height, width, 3), dtype=float)

    max_dist = max(
        np.linalg.norm(red_point - green_point),
        np.linalg.norm(red_point - blue_point),
        np.linalg.norm(green_point - blue_point),
    )

    for y in range(height):
        for x in range(width):
            p = np.array([x, y])
            if not _point_in_triangle(p, red_point, green_point, blue_point):
                continue

            d_red = np.linalg.norm(p - red_point)
            d_green = np.linalg.norm(p - green_point)
            d_blue = np.linalg.norm(p - blue_point)

            color = np.array(
                [
                    1.0 - d_red / max_dist,
                    1.0 - d_green / max_dist,
                    1.0 - d_blue / max_dist,
                ]
            )
            color_max = color.max()
            if color_max > 0:
                color /= color_max

            image[y, x] = color

    return image


def _barycentric_to_xy(A, D, P, red_point, green_point, blue_point):
    """Convert a single (A, D, P) barycentric weight to (x, y) pixel coordinates."""
    total = A + D + P
    if total == 0:
        # All-zero case: place the point at the triangle's centroid.
        return tuple((red_point + green_point + blue_point) / 3)

    A, D, P = A / total, D / total, P / total
    point = A * red_point + D * green_point + P * blue_point
    return (point[0], point[1])


def plot_triangle(
    image,
    points_xy,
    label,
    red_point,
    green_point,
    blue_point,
    output_path=DEFAULT_OUTPUT,
    show=True,
):
    """
    Render the triangle image with a point marked on it, then save
    (and optionally show) the plot.

    image: (height, width, 3) float array, as returned by generate_triangle.
    points_xy: list of (x, y) pixel coordinates to mark.
    label: label string for the point.
    red_point, green_point, blue_point: pixel coordinates of the Analytical,
        Data-Driven, and Physical vertices.
    output_path: where to save the plot image.
    show: if True, also display the plot with plt.show().
    """
    plt.figure(figsize=(8, 7))
    plt.imshow(image)

    x, y = points_xy[0]
    plt.plot(x, y, marker="o", color="black", markersize=6)
    plt.text(x + 12, y + 12, label, fontsize=10, color="black")

    plt.text(
        green_point[0] - 20,
        green_point[1] + 20,
        "Data-Driven",
        fontsize=12,
        color="green",
    )
    plt.text(
        red_point[0] - 20, red_point[1] - 20, "Analytical", fontsize=12, color="red"
    )
    plt.text(
        blue_point[0] + 20, blue_point[1] + 20, "Physical", fontsize=12, color="blue"
    )
    plt.axis("off")

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    plt.savefig(output_path)

    if show:
        plt.show()


def generate_triangle(
    point=None,
    border=100,
    label="My Research",
    generate_plot=True,
    output_path=DEFAULT_OUTPUT,
    width=WIDTH,
    height=HEIGHT,
):
    """
    Build the A/D/P triangle and place a point on it at the given
    barycentric coordinates. Optionally renders/saves/shows the plot via
    plot_triangle.

    point: (A, D, P) tuple, A, D, P are weights toward the
        Analytical (red), Data-Driven (green), and Physical (blue) 
        vertices; each triple is normalized automatically, so only
        relative proportions matter. A single (A, D, P) tuple is also
        accepted directly. Defaults to a single centered-ish point.
    labels: optional label string. Defaults to "My Research".
    border: pixel margin from image edges to the triangle's vertices.
    generate_plot: if True, call plot_triangle to render, save, and show the
        plot; if False, just compute and return the image + point (useful
        for scripting/tests).
    output_path: where to save the plot image, passed through to
        plot_triangle.

    Returns:
        image: (height, width, 3) float array of the rendered triangle.
        points_xy: list of (x, y) pixel coordinates, one per input point.
    """

    if point is None:
        point = (0.5, 0.5, 0.0)

    red_point = np.array([width / 2, border])  # Analytical
    green_point = np.array([border, height - border])  # Data-Driven
    blue_point = np.array([width - border, height - border])  # Physical

    image = _build_triangle_image(red_point, green_point, blue_point, width, height)
    points_xy = [_barycentric_to_xy(point[0], point[1], point[2], red_point, green_point, blue_point)]
    if generate_plot:
        plot_triangle(
            image,
            points_xy,
            label,
            red_point,
            green_point,
            blue_point,
            output_path=output_path,
        )

    return image, points_xy


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Plot a point in the Analytical / Data-Driven / "
            "Physical barycentric triangle."
        )
    )
    parser.add_argument(
        "output",
        nargs="?",
        default=DEFAULT_OUTPUT,
        help=f"Path to save the plot image (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "coords",
        nargs="*",
        type=float,
        help="A D P values for the point, e.g. '0.5 0.3 0.2'",
    )
    parser.add_argument(
        "--border",
        type=int,
        default=100,
        help="Pixel margin around the triangle (default: 100)",
    )
    parser.add_argument(
        "--no-plot", action="store_true", help="Skip rendering/saving/showing the plot"
    )

    args = parser.parse_args()

    if len(args.coords) % 3 != 0:
        parser.error(
            f"Coordinates must be given as complete A D P triples (got {len(args.coords)} values)"
        )

    point = None
    if args.coords:
        point = tuple(args.coords)

    generate_triangle(
        point=point,
        border=args.border,
        generate_plot=not args.no_plot,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()