""" Module with script to download public data
"""
import argparse
import os
import sys

import requests
import tqdm


FILES_TO_DOWNLOAD = {
    "policy_model": {
        "filename": "uspto_model.hdf5",
        "url": "https://zenodo.org/record/7341155/files/uspto_keras_model.hdf5",
    },
    "policy_model_onnx": {
        "filename": "uspto_model.onnx",
        "url": "https://zenodo.org/record/7797465/files/uspto_model.onnx",
    },
    "template_file": {
        "filename": "uspto_templates.csv.gz",
        "url": "https://zenodo.org/record/7341155/files/uspto_unique_templates.csv.gz",
    },
    "ringbreaker_model": {
        "filename": "uspto_ringbreaker_model.hdf5",
        "url": "https://zenodo.org/record/7341155/files/uspto_ringbreaker_keras_model.hdf5",
    },
    "ringbreaker_model_onnx": {
        "filename": "uspto_ringbreaker_model.onnx",
        "url": "https://zenodo.org/record/7797465/files/uspto_ringbreaker_model.onnx",
    },
    "ringbreaker_templates": {
        "filename": "uspto_ringbreaker_templates.csv.gz",
        "url": "https://zenodo.org/record/7341155/files/uspto_ringbreaker_unique_templates.csv.gz",
    },
    "stock": {
        "filename": "zinc_stock.hdf5",
        "url": "https://ndownloader.figshare.com/files/23086469",
    },
    "filter_policy": {
        "filename": "uspto_filter_model.hdf5",
        "url": "https://ndownloader.figshare.com/files/25584743",
    },
    "filter_policy_onnx": {
        "filename": "uspto_filter_model.onnx",
        "url": "https://zenodo.org/record/7797465/files/uspto_filter_model.onnx",
    },
}

YAML_TEMPLATE = """policy:
  files:
    uspto:
      - {}
      - {}
    ringbreaker:
      - {}
      - {}     
filter:
  files:
    uspto: {}
stock:
  files:
    zinc: {}
"""

def main() -> None:
    """Entry-point for CLI"""
    path = "."

    with open(os.path.join(path, "config.yml"), "w") as fileobj:
        path = os.path.abspath(path)
        fileobj.write(
            YAML_TEMPLATE.format(
                os.path.join(path, FILES_TO_DOWNLOAD["policy_model_onnx"]["filename"]),
                os.path.join(path, FILES_TO_DOWNLOAD["template_file"]["filename"]),
                os.path.join(
                    path, FILES_TO_DOWNLOAD["ringbreaker_model_onnx"]["filename"]
                ),
                os.path.join(
                    path, FILES_TO_DOWNLOAD["ringbreaker_templates"]["filename"]
                ),
                os.path.join(path, FILES_TO_DOWNLOAD["filter_policy_onnx"]["filename"]),
                os.path.join(path, FILES_TO_DOWNLOAD["stock"]["filename"]),
            )
        )
    print("Configuration file written to config.yml")


if __name__ == "__main__":
    main()