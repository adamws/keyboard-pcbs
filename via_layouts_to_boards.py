import concurrent.futures
import glob
import itertools
import json
import logging
import math
import os
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from enum import Enum
from functools import partial
from pathlib import Path
from typing import List, Union, Tuple, cast

import drawsvg as dw
import pcbnew
import svgpathtools

from colormath.color_objects import LabColor, sRGBColor
from colormath.color_conversions import convert_color
from jinja2 import Environment, FileSystemLoader, select_autoescape
from scour.scour import scourString
from scour.scour import sanitizeOptions as sanitizeScourOptions
from scour.scour import parse_args as parseScourArgs

from kbplacer.board_builder import BoardBuilder
from kbplacer.defaults import DEFAULT_DIODE_POSITION
from kbplacer.element_position import ElementInfo, PositionOption
from kbplacer.key_placer import KeyPlacer
from kbplacer.kle_serial import Key, Keyboard, ViaKeyboard, parse_via
from pyurlon import stringify

Numeric = Union[int, float]
Point = Tuple[Numeric, Numeric]
Box = Tuple[Numeric, Numeric, Numeric, Numeric]

REPOSITORY_URL = "https://github.com/the-via/keyboards.git"

FORMAT = "%(message)s"
logging.basicConfig(format=FORMAT)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

kbplacer_logger = logging.getLogger("kbplacer")
kbplacer_logger.setLevel(logging.ERROR)

ORIGIN_X = 4
ORIGIN_Y = 4

KEY_WIDTH = 52
KEY_HEIGHT = 52
INNER_GAP_LEFT = 6
INNER_GAP_TOP = 4
INNER_GAP_BOTTOM = 8

LABEL_SIZE = 12


def lighten_color(hex_color: str) -> str:
    color = sRGBColor.new_from_rgb_hex(hex_color)
    lab_color = convert_color(color, LabColor)
    lab_color.lab_l = min(100, lab_color.lab_l * 1.2)
    rgb = convert_color(lab_color, sRGBColor)
    return sRGBColor(
        rgb.clamped_rgb_r, rgb.clamped_rgb_g, rgb.clamped_rgb_b
    ).get_rgb_hex()


def rotate(origin, point, angle):
    ox, oy = origin
    px, py = point
    radians = math.radians(angle)

    qx = ox + math.cos(radians) * (px - ox) - math.sin(radians) * (py - oy)
    qy = oy + math.sin(radians) * (px - ox) + math.cos(radians) * (py - oy)
    return qx, qy


def is_encoder(key: Key) -> bool:
    if len(key.labels) >= 5 and key.labels[4] and key.labels[4].startswith("e"):
        return True
    return False


def is_iso_enter(key: Key) -> bool:
    if key.width == 1.25 and key.height == 2 and key.width2 == 1.5 and key.height2 == 1:
        return True
    return False


def build_key(key: Key):
    group = dw.Group()
    width_px = key.width * KEY_WIDTH
    height_px = key.height * KEY_HEIGHT

    not_rectangle = key.width != key.width2 or key.height != key.height2

    # some layouts fail due to: 'input #ccccccc is not in #RRGGBB format',
    # truncate to fix most of these issues
    dark_color = key.color[0:7]
    light_color = lighten_color(dark_color)

    def border(x, y, w, h) -> dw.Rectangle:  # pyright: ignore
        return dw.Rectangle(
            x * KEY_WIDTH,
            y * KEY_HEIGHT,
            w * KEY_WIDTH,
            h * KEY_HEIGHT,
            rx="5",
            fill="none",
            stroke="black",
            stroke_width=2,
        )

    def fill(x, y, w, h) -> dw.Rectangle:  # pyright: ignore
        return dw.Rectangle(
            x * KEY_WIDTH + 1,
            y * KEY_HEIGHT + 1,
            w * KEY_WIDTH - 2,
            h * KEY_HEIGHT - 2,
            rx="5",
            fill=dark_color,
        )

    def top(x, y, w, h) -> dw.Rectangle:  # pyright: ignore
        return dw.Rectangle(
            x * KEY_WIDTH + INNER_GAP_LEFT,
            y * KEY_HEIGHT + INNER_GAP_TOP,
            w * KEY_WIDTH - 2 * INNER_GAP_LEFT,
            h * KEY_HEIGHT - INNER_GAP_TOP - INNER_GAP_BOTTOM,
            rx="5",
            fill=light_color,
        )

    if not key.decal:
        layers = ["border", "fill", "top"]

        for layer in layers:
            group.append(locals()[layer](0, 0, key.width, key.height))
            if not_rectangle:
                group.append(locals()[layer](key.x2, key.y2, key.width2, key.height2))

    top_label_position = (
        INNER_GAP_LEFT + 1 + (key.x2 * KEY_WIDTH),
        INNER_GAP_TOP + LABEL_SIZE + 1 + (key.y2 * KEY_HEIGHT),
    )

    # top left-label (might be missing for some decal keys)
    if len(key.labels) >= 1 and key.labels[0]:
        group.append(
            dw.Text(
                key.labels[0],
                font_size=LABEL_SIZE,
                x=top_label_position[0],
                y=top_label_position[1],
                fill=key.textColor[0] if len(key.textColor) else key.default.textColor,
            )
        )
    # bottom right label
    if len(key.labels) >= 9 and key.labels[8]:
        group.append(
            dw.Text(
                key.labels[8],
                font_size=LABEL_SIZE,
                x=width_px - INNER_GAP_LEFT - 1,
                y=height_px - LABEL_SIZE,
                fill=key.textColor[8]
                if len(key.textColor) >= 9
                else key.default.textColor,
                text_anchor="end",
            )
        )
    # center label (denoting encoder)
    if is_encoder(key):
        group.append(
            dw.Text(
                key.labels[4],
                font_size=LABEL_SIZE,
                x=width_px / 2,
                y=height_px / 2 - 2,
                fill=key.textColor[4]
                if len(key.textColor) >= 5
                else key.default.textColor,
                text_anchor="middle",
                dominant_baseline="middle",
            )
        )

    return group


def calcualte_canvas_size(keyboard: ViaKeyboard) -> tuple[int, int]:
    max_x = 0
    max_y = 0
    for k in itertools.chain(keyboard.keys, keyboard.alternative_keys):
        angle = k.rotation_angle
        if angle != 0:
            # when rotated, check each corner
            x1 = KEY_WIDTH * k.x
            x2 = KEY_WIDTH * k.x + KEY_WIDTH * k.width
            y1 = KEY_HEIGHT * k.y
            y2 = KEY_HEIGHT * k.y + KEY_HEIGHT * k.height

            for x, y in [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]:
                rot_x = KEY_WIDTH * k.rotation_x
                rot_y = KEY_HEIGHT * k.rotation_y
                x, y = rotate((rot_x, rot_y), (x, y), angle)
                x, y = int(x), int(y)
                if x >= max_x:
                    max_x = x
                if y >= max_y:
                    max_y = y

        else:
            # when not rotated, it is safe to check only bottom right corner:
            x = KEY_WIDTH * k.x + KEY_WIDTH * k.width
            y = KEY_HEIGHT * k.y + KEY_HEIGHT * k.height
            if x >= max_x:
                max_x = x
            if y >= max_y:
                max_y = y

    return int(max_x) + 2 * ORIGIN_X, int(max_y) + 2 * ORIGIN_Y


def clone(destination: str) -> None:
    p = subprocess.Popen(["git", "clone", "--depth", "1", REPOSITORY_URL, destination])
    p.communicate()
    assert p.returncode == 0


def git_repository_sha(path):
    # Check if the path is a directory
    if not os.path.isdir(path):
        raise ValueError("Invalid path. Please provide a valid Git repository path.")
    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=path
        ).strip()
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Error executing Git command: {e}")
    return git_hash.decode("utf-8")


def get_footprints_dir():
    script_dir = Path(__file__).parent.resolve()
    return script_dir / "footprints/local.pretty"


def create_result_dir(tempdir, output: Path, layout_file: str):
    path = Path(layout_file)
    destination = str(path.parent).replace(f"{tempdir}/src", str(output))
    os.makedirs(Path(destination), exist_ok=True)
    return destination


def load_keyboard(layout_file: str) -> ViaKeyboard:
    with open(layout_file, "r") as f:
        layout = json.load(f)
        keyboard = parse_via(layout)
        return keyboard


def create_board(keyboard: ViaKeyboard, destination_pcb: Path):
    builder = BoardBuilder(
        switch_footprint=f"{get_footprints_dir()}:SW_Cherry_MX_PCB_1.00u",
        diode_footprint=f"{get_footprints_dir()}:D_SOD-323F",
    )
    # decal keys should be ignored by `builder.create_board`, workaround until
    # fixed in kbplacer:
    keyboard.keys = [k for k in keyboard.keys if not k.decal]

    board = builder.create_board(keyboard)
    placer = KeyPlacer(board)
    placer.place_switches(
        keyboard,
        "SW{}",
    )
    placer.place_switch_elements(
        "SW{}",
        [ElementInfo("D{}", PositionOption.DEFAULT, DEFAULT_DIODE_POSITION, "")],
    )
    placer.route_switches_with_diodes("SW{}", "D{}", [])
    placer.route_rows_and_columns()
    placer.remove_dangling_tracks()

    board.Save(str(destination_pcb))
    # remove unneeded project files
    os.remove(destination_pcb.with_suffix(".kicad_prl"))
    os.remove(destination_pcb.with_suffix(".kicad_pro"))


def merge_bbox(left: Box, right: Box) -> Box:
    """
    Merge bounding boxes in format (xmin, xmax, ymin, ymax)
    """
    return tuple([f(l, r) for l, r, f in zip(left, right, [min, max, min, max])])


def shrink_svg(svg: ET.ElementTree, margin: int = 0) -> Tuple[Tuple[float, float], Box]:
    """
    Shrink the SVG canvas to the size of the drawing.
    """
    root = svg.getroot()
    paths = svgpathtools.document.flattened_paths(root)

    if len(paths) == 0:
        msg = "Corrupted svg file"
        raise ValueError(msg)

    bbox = paths[0].bbox()
    for x in paths:
        bbox = merge_bbox(bbox, x.bbox())
    bbox = list(bbox)
    bbox[0] -= int(margin)
    bbox[1] += int(margin)
    bbox[2] -= int(margin)
    bbox[3] += int(margin)

    root.set(
        "viewBox",
        f"{bbox[0]} {bbox[2]} {bbox[1] - bbox[0]} {bbox[3] - bbox[2]}",
    )

    width = float(bbox[1] - bbox[0])
    height = float(bbox[3] - bbox[2])
    root.set("width", f"{width}mm")
    root.set("height", f"{height}mm")
    return (width, height), cast(Box, bbox)


def remove_empty_groups(root):
    name = "{http://www.w3.org/2000/svg}g"
    for elem in root.findall(name):
        if len(elem) == 0:
            root.remove(elem)
    for child in root:
        remove_empty_groups(child)


# pcb plotting based on https://github.com/kitspace/kitspace-v2/tree/master/processor/src/tasks/processKicadPCB
# and https://gitlab.com/kicad/code/kicad/-/blob/master/demos/python_scripts_examples/plot_board.py
def create_render(pcb_file: Path) -> Path:
    board = pcbnew.LoadBoard(pcb_file)
    project_name = pcb_file.stem
    dst_dir = pcb_file.parent.absolute()

    layers = pcbnew.LSET.PhysicalLayersMask()

    plot_control = pcbnew.PLOT_CONTROLLER(board)
    plot_options = plot_control.GetPlotOptions()
    plot_options.SetOutputDirectory(dst_dir)
    plot_options.SetColorSettings(
        pcbnew.GetSettingsManager().GetColorSettings("vampire")
    )
    plot_options.SetPlotFrameRef(False)
    plot_options.SetSketchPadLineWidth(pcbnew.FromMM(0.35))
    plot_options.SetAutoScale(False)
    plot_options.SetMirror(False)
    plot_options.SetUseGerberAttributes(False)
    plot_options.SetScale(1)
    plot_options.SetUseAuxOrigin(True)
    plot_options.SetNegative(False)
    plot_options.SetPlotReference(True)
    plot_options.SetPlotValue(True)
    plot_options.SetPlotInvisibleText(False)
    plot_options.SetDrillMarksType(pcbnew.DRILL_MARKS_NO_DRILL_SHAPE)
    plot_options.SetSvgPrecision(aPrecision=1)

    plot_control.OpenPlotfile("render", pcbnew.PLOT_FORMAT_SVG)
    plot_control.SetColorMode(True)
    plot_control.PlotLayers(layers.Seq())
    plot_control.ClosePlot()

    filepath = dst_dir / f"{project_name}-render.svg"
    tree = ET.parse(filepath)
    root = tree.getroot()

    # for some reason there is plenty empty groups in generated svg's (kicad bug?)
    # remove for clarity:
    remove_empty_groups(root)

    dimensions, bbox = shrink_svg(tree, margin=1)

    # add background
    background_group = ET.Element("{http://www.w3.org/2000/svg}g")
    background_group.set("style", "fill:#282a36; fill-opacity:1.000;")
    background = ET.Element("{http://www.w3.org/2000/svg}rect")
    background.set("x", str(bbox[0]))
    background.set("y", str(bbox[2]))
    background.set("width", str(dimensions[0]))
    background.set("height", str(dimensions[1]))

    background_group.append(background)
    # insert after title:
    root.insert(2, background_group)

    tree.write(filepath)

    optimize_svg(filepath)

    return filepath


def optimize_svg(source_svg: Path, optimization_passes: int = 2):
    def _optimize(sourcesvg):
        scouroptions = parseScourArgs(
            [
                "--enable-id-stripping",
                "--enable-comment-stripping",
                "--shorten-ids",
                "--create-groups",
            ]
        )
        scouroptions = sanitizeScourOptions(scouroptions)
        optimizedsvg = scourString(sourcesvg, scouroptions)
        return optimizedsvg

    with open(source_svg) as f:
        svg = f.read()

    for i in range(optimization_passes):
        svg = _optimize(svg)

    # overwrite
    with open(source_svg, "w") as f:
        f.write(svg)


def create_layout_image(keyboard: ViaKeyboard, png_output: Path):
    width, height = calcualte_canvas_size(keyboard)
    d = dw.Drawing(width, height)

    for k in itertools.chain(keyboard.keys, keyboard.alternative_keys):
        width = k.width
        height = k.height
        x = KEY_WIDTH * k.x
        y = KEY_WIDTH * k.y

        key = build_key(k)

        args = {}
        angle = k.rotation_angle
        if angle != 0:
            rot_x = KEY_WIDTH * k.rotation_x
            rot_y = KEY_HEIGHT * k.rotation_y
            args["transform"] = f"rotate({angle} {rot_x} {rot_y})"
        d.append(dw.Use(key, x + ORIGIN_X, y + ORIGIN_Y, **args))

    d.save_png(str(png_output))


class KeyboardTag(Enum):
    ORTHOLINEAR = 1
    STAGGERED = 2
    OTHER = 3


def tag_keyboard(keyboard: ViaKeyboard) -> List[KeyboardTag]:
    tags = []

    anchor = None
    for k in keyboard.keys:
        if not is_encoder(k):
            anchor = k
            break

    ortholinear_keys = 0
    rotated_keys = 0
    encoders = 0
    iso_enters = 0

    # anchor can be None only if 'keyboard' has only encoders,
    # then it should be tagged as 'OTHER'
    if anchor:
        for k in keyboard.keys:
            if (
                float(k.x - anchor.x).is_integer()
                and not is_encoder(k)
            ):
                ortholinear_keys += 1
            if k.rotation_angle != 0:
                rotated_keys += 1
            if is_encoder(k):
                encoders += 1
            if is_iso_enter(k):
                iso_enters += 1

        if ortholinear_keys == len(keyboard.keys) - encoders:
            tags.append(KeyboardTag.ORTHOLINEAR)
        elif rotated_keys != 0:
            tags.append(KeyboardTag.OTHER)
        else:
            tags.append(KeyboardTag.STAGGERED)

    if not tags:
        tags.append(KeyboardTag.OTHER)

    return tags


def process_layout(tempdir, output, layout_file):
    logger.info(f"Processing: {layout_file}")

    name = Path(layout_file).stem
    destination = create_result_dir(tempdir, output, layout_file)
    destination = Path(destination)
    try:
        keyboard = load_keyboard(layout_file)

        metadata = destination / f"{name}-metadata.json"
        with open(metadata, "w") as f:
            keys_without_decals = [k for k in keyboard.keys if not k.decal]
            f.write(
                json.dumps(
                    {
                        "total": len(keys_without_decals),
                        "tags": [t.name for t in tag_keyboard(keyboard)],
                    }
                )
            )

        kle_layout = destination / f"{name}-kle.json"
        with open(kle_layout, "w") as f:
            new_keyboard = Keyboard(
                meta=keyboard.meta,
                keys=keyboard.keys + keyboard.alternative_keys,
            )
            f.write(new_keyboard.to_kle())

        layout_png = destination / f"{name}-layout.png"
        create_layout_image(keyboard, layout_png)

        pcb_path = destination / f"{name}.kicad_pcb"
        create_board(keyboard, pcb_path)
        create_render(pcb_path)
    except Exception as e:
        msg = f"\t{layout_file} failed with error: '{e}'"
        logger.error(msg)
        with open(destination / "error.log", "a") as f:
            f.write(msg.strip() + "\n")


def divide_list(lst, n):
    size = len(lst) // n
    remainder = len(lst) % n
    start = 0

    result = []
    for i in range(n):
        end = start + size + (1 if i < remainder else 0)
        result.append(lst[start:end])
        start = end

    return result


def generate_images(output: Path, part: int, num_parts: int):
    with tempfile.TemporaryDirectory() as tempdir:
        logger.info(f"Created temporary directory {tempdir}")
        clone(tempdir)
        layouts = glob.glob(f"{tempdir}/src/**/*json", recursive=True)
        layouts = sorted(layouts)
        layouts = divide_list(layouts, num_parts)
        layouts = layouts[part - 1]

        shutil.rmtree(output, ignore_errors=True)
        os.makedirs(output)

        with open(output.parent / "revision.txt", "w") as f:
            f.write(git_repository_sha(tempdir))

        with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
            process_layout_partial = partial(process_layout, tempdir, output)
            executor.map(process_layout_partial, layouts)


def generate_index(output: Path, fix_links: bool = False):
    results = glob.glob(f"{output}/**/*-kle.json", recursive=True)
    results = sorted(results)

    def build_link(output: Path, item: Path, fix_links: bool):
        part1 = str(item.relative_to(output.parent))
        if fix_links:
            return "/keyboard-pcbs/" + part1
        return part1

    max_keys = 0
    keyboards = []
    for kle_layout in results:
        result = Path(kle_layout)
        name = result.stem.removesuffix("-kle")
        metadata = result.with_name(f"{name}-metadata.json")
        destination = result.parent
        header = f"{destination.relative_to(output)}/{name}"
        # some vendors put each keyboard in separate folder of the same name,
        # if name is equal folder name, shorten it in header
        header_parts = header.split("/")
        if len(header_parts) > 2:
            if header_parts[-1] == header_parts[-2]:
                header = "/".join(header_parts[:-1])

        layout_png = destination / f"{name}-layout.png"
        pcb_path = destination / f"{name}.kicad_pcb"
        render_path = destination / f"{name}-render.svg"
        with open(result, "r") as f:
            data = json.loads("[" + f.read() + "]")
            kle_url = stringify(data)
            # keyboard-layout-editor uses old version of urlon, need
            # to replace `$` with `_` to be compatible with it.
            # see https://github.com/cerebral/urlon/commit/efbdc00af4ec48cabb28372e6f3fcc0c0a30a4c7
            kle_url = kle_url.replace("$", "_")
            kle_url = "http://www.keyboard-layout-editor.com/##" + kle_url

        with open(metadata, "r") as f:
            metadata = json.load(f)
            total_keys = metadata["total"]
            if total_keys > max_keys:
                max_keys = total_keys

        keyboards.append(
            {
                "header": header,
                "links": {
                    "Layout editor": kle_url,
                    "Download PCB": build_link(output, pcb_path, fix_links),
                    "PCB render": build_link(output, render_path, fix_links),
                },
                "total_keys": total_keys,
                "tags": ", ".join(metadata["tags"]),
                "image_path": build_link(output, layout_png, fix_links),
            }
        )

    with open(output.parent / "revision.txt", "r") as f2:
        revision = f2.readline()

    env = Environment(
        loader=FileSystemLoader("templates"), autoescape=select_autoescape()
    )
    template = env.get_template("index.html")
    with open(output.parent / "index.html", "w") as f:
        f.writelines(
            template.generate(keyboards=keyboards, max_keys=max_keys, revision=revision)
        )


def get_errors(output: Path):
    errors = glob.glob(f"{output}/**/error.log", recursive=True)
    errors = sorted(errors)
    if errors:
        logger.warning("Errors summary:")
        for e in errors:
            with open(e, "r") as f:
                for line in f.readlines():
                    if line:
                        logger.warning(line.strip())


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="via PCB builder")
    subparsers = parser.add_subparsers(dest="subparser_name")
    parser_generate = subparsers.add_parser("generate", help="Generate stage")
    parser_generate.add_argument(
        "-n", required=False, default=1, type=int, help="Part to generate"
    )
    parser_generate.add_argument(
        "-parts", required=False, default=1, type=int, help="Total number of parts"
    )

    parser_collect = subparsers.add_parser("collect", help="Collect stage")
    parser_collect.add_argument(
        "-gh", required=False, action="store_true", help="Use for deploy (fixes links)"
    )

    args = parser.parse_args()

    script_dir = Path(__file__).parent.resolve()
    output = script_dir / "gh-pages" / "output"
    if args.subparser_name == "collect":
        generate_index(output, args.gh)
        get_errors(output)
    else:
        generate_images(output, args.n, args.parts)

    #layout_file = "/home/aws/git/via-keyboards/src/0_sixty/0_sixty.json"
    #layout_file = "/home/aws/git/via-keyboards/src/1upkeyboards/pi40/pi40.json"
    #keyboard = load_keyboard(layout_file)
    #result = tag_keyboard(keyboard)
    #print(result)
