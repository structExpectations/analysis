#!/usr/bin/env python
"""Create estimation report
This script allows to create a summary estimation report.
It is assumed that it is called in the
directory where the estimation is running.
"""
import argparse
import shutil
import glob
import os

from PyPDF2 import PdfFileMerger

from python.create_fig_model_fit import create_fig_model_fit
from configurations.analysis_soepy_config import FIGURES_DIR

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Create estimation report with essential information on results"
    )

    parser.add_argument(
        "-a",
        "--all",
        action="store_true",
        dest="is_all",
        help="create all available information",
    )

    parser.add_argument(
        "-m",
        "--model-fit",
        action="store_true",
        dest="is_model_fit",
        help="create information on model fit",
    )

    parser.add_argument(
        "-c",
        "--clear",
        action="store_true",
        dest="is_clear",
        help="clear existing results",
    )

    args = parser.parse_args()

    if args.is_clear:
        shutil.rmtree("figures", ignore_errors=True)

    if not os.path.exists(str(FIGURES_DIR)):
        os.mkdir(str(FIGURES_DIR))

    if args.is_model_fit or args.is_all:
        create_fig_model_fit()

    # We merge all output for easier accessibility in a single document
    # and store it in the estimation directory.
    merger = PdfFileMerger()
    for fname in sorted(glob.glob(str(FIGURES_DIR) + "/*.pdf")):
        merger.append(fname)
    merger.write(str(FIGURES_DIR) + "/estimation_report.pdf")
    merger.close()
