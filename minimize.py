import subprocess
from pathlib import Path
from glob import glob

script_dir = Path(__file__).parent.resolve()


def images_are_equal(original_svg: Path, minimized_svg: Path) -> bool:
    import cairosvg
    import numpy as np
    from PIL import Image, ImageChops

    minimized_png = minimized_svg.with_suffix(".png")
    orignial_png = original_svg.with_suffix(".png")

    comparison_dpi = 1270
    print(f"{original_svg=} -> {orignial_png=}")
    cairosvg.svg2png(
        url=str(original_svg), write_to=str(orignial_png), dpi=comparison_dpi
    )

    print(f"{minimized_svg=} -> {minimized_png=}")
    cairosvg.svg2png(
        url=str(minimized_svg), write_to=str(minimized_png), dpi=comparison_dpi
    )

    image1 = Image.open(str(orignial_png))
    image2 = Image.open(str(minimized_png))

    if image1.size != image2.size:
        print("Images sizes are different.")
        return False

    if image1.mode != image2.mode:
        print("Images modes are different.")
        return False

    diff = ImageChops.difference(image1, image2)
    diff.save(str(minimized_png.with_suffix(".diff.png")))

    diff_array = np.asarray(diff)
    total_diff = np.sum(diff_array)

    max_diff = diff_array.shape[0] * diff_array.shape[1] * 255 * 3  # max possible sum
    percentage_diff = (total_diff / max_diff) * 100
    print(f"difference: {percentage_diff=}")

    image1.close()
    image2.close()

    return percentage_diff < np.float64(0.05)


def svgo(original_svg: Path, *, inplace: bool = True) -> Path:
    if inplace:
        minimized_out = original_svg
        args = ["npx", "svgo", "-i", str(original_svg)]
    else:
        minimized_out = original_svg.with_suffix(".minimized.svg")
        args = ["npx", "svgo", "-i", str(original_svg), "-o", str(minimized_out)]
    p = subprocess.run(args, text=True, capture_output=True)
    assert p.returncode == 0, f"error: {p}"
    lastline = p.stdout.splitlines()[-1]
    name = original_svg.relative_to(script_dir)
    print(f"svgo {name}: {lastline}")
    return minimized_out


def minimize(original_svg: Path, *, check=False) -> Path:
    minimized_svg = svgo(original_svg, inplace=check == False)
    if check:
        assert images_are_equal(original_svg, minimized_svg)
    return minimized_svg


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="via PCB builder")
    parser.add_argument(
        "--check",
        required=False,
        action="store_true",
        help="Validate if minimization works",
    )
    args = parser.parse_args()

    before = 0
    after = 0

    output = script_dir / "gh-pages" / "output"
    for filename in glob(f"{output}/**/*.svg", recursive=True):
        if filename.endswith("-layout.svg"):
            original = Path(filename)
            before += original.stat().st_size
            minimized = minimize(original, check=args.check)
            after += minimized.stat().st_size

    change = ((after - before) / before) * 100
    print(f"Initial size: {before}, optimized: {after} change: {change:.2f}%")
