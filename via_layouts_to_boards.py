import concurrent.futures
import glob
import json
import logging
import os
import shutil
import subprocess
import tempfile
from collections import defaultdict
from dataclasses import asdict
from enum import Enum
from functools import partial
from pathlib import Path
from typing import List, Union, Tuple, Dict

from jinja2 import Environment, FileSystemLoader, select_autoescape

from kbplacer.kle_serial import Key, MatrixAnnotatedKeyboard, parse_via
from pyurlon import stringify

Numeric = Union[int, float]
Box = Tuple[Numeric, Numeric, Numeric, Numeric]

REPOSITORY_URL = "https://github.com/the-via/keyboards.git"

FORMAT = "%(message)s"
logging.basicConfig(format=FORMAT)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

kle_serial_logger = logging.getLogger("kbplacer.kle_serial")
kle_serial_logger.setLevel(logging.WARNING)


def is_encoder(key: Key) -> bool:
    if len(key.labels) >= 5 and key.labels[4] and key.labels[4].startswith("e"):
        return True
    return False


def is_iso_enter(key: Key) -> bool:
    if key.width == 1.25 and key.height == 2 and key.width2 == 1.5 and key.height2 == 1:
        return True
    return False


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


def create_result_dir(tempdir, output: Path, layout_file: str):
    path = Path(layout_file)
    destination = str(path.parent).replace(f"{tempdir}/src", str(output))
    os.makedirs(Path(destination), exist_ok=True)
    return destination


class KeyboardTag(Enum):
    ORTHOLINEAR = 1
    STAGGERED = 2
    OTHER = 3


def tag_keyboard(keyboard: MatrixAnnotatedKeyboard) -> List[KeyboardTag]:
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
            if float(k.x - anchor.x).is_integer() and not is_encoder(k):
                ortholinear_keys += 1
            if k.rotation_angle != 0:
                rotated_keys += 1
            if is_encoder(k):
                encoders += 1
            if is_iso_enter(k):
                iso_enters += 1

        if ortholinear_keys == len(keyboard.keys) - encoders and rotated_keys == 0:
            tags.append(KeyboardTag.ORTHOLINEAR)
        elif rotated_keys != 0:
            tags.append(KeyboardTag.OTHER)
        else:
            tags.append(KeyboardTag.STAGGERED)

    if not tags:
        tags.append(KeyboardTag.OTHER)

    return tags


def load_keyboard(layout_file: str) -> MatrixAnnotatedKeyboard:
    with open(layout_file, "r") as f:
        layout = json.load(f)
        keyboard = parse_via(layout)
        return keyboard


def process_layout(tempdir, output, layout_files: List[str]):
    layout_file = layout_files[0]
    logger.info(f"Processing: {layout_file}")

    if len(layout_files) > 1:
        logger.info(f"Duplicates: {layout_files[1:]}")

    name = Path(layout_file).stem
    destination = create_result_dir(tempdir, output, layout_file)
    destination = Path(destination)

    shutil.copy(layout_file, destination)
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
                        "duplicates": [
                            d.removeprefix(f"{tempdir}/src/").removesuffix(".json")
                            for d in layout_files[1:]
                        ],
                    }
                )
            )

        args = ["kbplacer-generate.sh", destination, name]
        logger.info(f"Running: {args}")
        p = subprocess.Popen(args)
        p.communicate()
        assert p.returncode == 0

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


def find_duplicates(tempdir, output, layout_files: List[str]) -> List[List[str]]:
    important_properties = [
        "labels",
        "x",
        "y",
        "width",
        "height",
        "x2",
        "y2",
        "width2",
        "height2",
        "rotation_x",
        "rotation_y",
        "rotation_angle",
        "decal",
        "ghost",
        "stepped",
        "nub",
    ]

    duplicates: Dict[str, List[str]] = defaultdict(list)

    for layout_file in layout_files:

        destination = create_result_dir(tempdir, output, layout_file)
        destination = Path(destination)
        try:
            k = load_keyboard(layout_file)
            kstr = ""

            for k in k.keys_in_matrix_order():
                key_dict = asdict(k)
                filtered_props = {
                    k: v for k, v in key_dict.items() if k in important_properties
                }
                kstr += str(filtered_props)

            duplicates[kstr].append(layout_file)
        except Exception as e:
            msg = f"\t{layout_file} failed with error: '{e}'"
            logger.error(msg)
            with open(destination / "error.log", "a") as f:
                f.write(msg.strip() + "\n")

    return list(duplicates.values())


def generate_images(output: Path, part: int, num_parts: int):
    with tempfile.TemporaryDirectory() as tempdir:
        logger.info(f"Created temporary directory {tempdir}")
        clone(tempdir)
        layouts = glob.glob(f"{tempdir}/src/**/*json", recursive=True)
        layouts = sorted(layouts)

        deduplicated_layouts = find_duplicates(tempdir, output, layouts)

        layouts = divide_list(deduplicated_layouts, num_parts)
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

    with open(output.parent / "revision.txt", "r") as f2:
        revision = f2.readline()

    max_keys = 0
    keyboards = []
    for kle_layout in results:
        result = Path(kle_layout)
        name = result.stem.removesuffix("-kle")
        metadata = result.with_name(f"{name}-metadata.json")
        destination = result.parent
        header = f"{destination.relative_to(output)}/{name}"
        via_path = f"src/{destination.relative_to(output)}/{name}.json"
        # some vendors put each keyboard in separate folder of the same name,
        # if name is equal folder name, shorten it in header
        header_parts = header.split("/")
        if len(header_parts) > 2:
            if header_parts[-1] == header_parts[-2]:
                header = "/".join(header_parts[:-1])

        layout_svg = destination / f"{name}-layout.svg"
        kicad_zip_path = destination / f"{name}.zip"
        schematic_render_path = destination / f"{name}-schematic.svg"
        pcb_render_path = destination / f"{name}-render.svg"
        with open(result, "r") as f:
            data = json.load(f)
            # keyboard-layout-editor uses old version of urlon,
            # for this reason each `_` in metadata value must be replaced with `-`
            # and all `$` in resulting url with `_`.
            # see https://github.com/cerebral/urlon/commit/efbdc00af4ec48cabb28372e6f3fcc0c0a30a4c7
            if isinstance(data[0], dict):
                for k, v in data[0].items():
                    data[0][k] = v.replace("_", "-")
            kle_url = stringify(data)
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
                    "Download KiCad project": build_link(
                        output, kicad_zip_path, fix_links
                    ),
                    "Schematic": build_link(output, schematic_render_path, fix_links),
                    "PCB render": build_link(output, pcb_render_path, fix_links),
                    "VIA": f"https://github.com/the-via/keyboards/tree/{revision}/{via_path}",
                },
                "total_keys": total_keys,
                "tags": ", ".join(metadata["tags"]),
                "duplicates": metadata["duplicates"],
                "image_path": build_link(output, layout_svg, fix_links),
            }
        )

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
