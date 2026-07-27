import argparse
import os
from pathlib import Path


VALID_IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
    ".JPG"
}


def collect_image_names(indir: str, outfile: str) -> None:
    input_directory = Path(indir)

    if not input_directory.exists():
        raise FileNotFoundError(
            f"Input directory does not exist: {input_directory}"
        )

    if not input_directory.is_dir():
        raise NotADirectoryError(
            f"Input path is not a directory: {input_directory}"
        )

    image_names = sorted(
        file_path.stem
        for file_path in input_directory.iterdir()
        if file_path.is_file()
        and file_path.suffix.lower() in VALID_IMAGE_EXTENSIONS
    )

    output_file = Path(outfile)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open("w", encoding="utf-8") as file:
        for image_name in image_names:
            file.write(f"{image_name}\n")

    print(f"Found {len(image_names)} valid images")
    print(f"Image list saved to: {output_file}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect valid image names into a text file."
    )
    parser.add_argument(
        "--indir",
        type=str,
        required=True,
        help="Directory containing the images.",
    )
    parser.add_argument(
        "--outfile",
        type=str,
        default="images_to_process.txt",
        help="Path of the generated text file.",
    )
    args = parser.parse_args()

    collect_image_names(args.indir, args.outfile)


if __name__ == "__main__":
    main()