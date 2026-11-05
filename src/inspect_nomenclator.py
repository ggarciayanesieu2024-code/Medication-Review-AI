"""
Inspect the structure of the AEMPS nomenclator XML files.

Why this script exists:
- Before writing a parser, we need the real tag names and how they are nested.
- Guessing tag names can produce code that runs without errors but reads nothing.

What it does:
- Lists every XML file in data/raw/nomenclator/ and the tags under its root element.
- Lists every distinct tag found in the main file, Prescripcion.xml.
- Prints one complete medicine record, so we can see its fields.

It only reads files. It does not change any data.

Run from the project root:
    python src/inspect_nomenclator.py
"""

import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path


# Folder where download_nomenclator.py unpacked the ZIP.
NOMENCLATOR_DIR = Path("data") / "raw" / "nomenclator"

# The file that contains the medicines themselves.
MAIN_FILE = NOMENCLATOR_DIR / "Prescripcion.xml"

# How many levels of nested tags to print when showing one record.
PRINT_DEPTH = 3


def print_element(element, depth=0):
    """Print an XML element and its children, indented to show nesting.

    Recursion walks down the tree: each child is printed one level deeper.
    Text is cut to 60 characters so long descriptions do not flood the screen.
    """
    text = (element.text or "").strip()
    line = "  " * depth + f"<{element.tag}>"
    if text:
        line += f" {text[:60]}"
    if element.attrib:
        line += f" {element.attrib}"
    print(line)

    if depth < PRINT_DEPTH:
        for child in element:
            print_element(child, depth + 1)


def summarise_files() -> None:
    """Show the root tag and direct child tags of every XML file in the folder.

    Each file has a root element; its direct children are the records.
    Counting them tells us what each file contains.
    """
    for xml_file in sorted(NOMENCLATOR_DIR.glob("*.xml")):
        root = ET.parse(xml_file).getroot()
        child_tags = Counter(child.tag for child in root)
        print(f"\n{xml_file.name}")
        print(f"  root tag: <{root.tag}>, direct children: {len(root)}")
        print(f"  child tags: {dict(child_tags)}")


def list_all_tags(root) -> None:
    """List every distinct tag used anywhere in the main file, with how often it appears.

    root.iter() walks through every element in the tree, at any depth.
    """
    counts = Counter(element.tag for element in root.iter())
    print("\nAll tags in Prescripcion.xml (tag: count):")
    for tag, count in counts.most_common():
        print(f"  {tag}: {count}")


def show_first_record(root) -> None:
    """Print the first prescription record in full.

    root[0] is the file header, which is not a medicine.
    root[1] is the first prescription, which is the record we want to inspect.
    """
    print("\nFirst prescription record in full:")
    print_element(root[1])


def main() -> None:
    summarise_files()

    root = ET.parse(MAIN_FILE).getroot()
    list_all_tags(root)
    show_first_record(root)


if __name__ == "__main__":
    main()
