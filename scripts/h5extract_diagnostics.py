#!/usr/bin/env python3
"""
Extract specific diagnostics from an HDF5 file into a new HDF5 file,
while preserving the metadata.
"""

import argparse
import os
import sys
import warnings
import h5py


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract specific diagnostics from an HDF5 file into a new HDF5 file, "
                    "while preserving the metadata."
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to the input HDF5 file"
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Path to the output HDF5 file (will be created)"
    )
    parser.add_argument(
        "--diagnostics", "--groups", "-d", "-g",
        required=True,
        dest="diagnostics",
        metavar="DIAGNOSTICS",
        help='Comma-separated list of group names to extract (e.g. "All probe,ExB CFL")'
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Overwrite the output file if it already exists"
    )
    return parser.parse_args()


def extract_data(input_path, output_path, to_extract, force=False):
    # Validate input path
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Guard against overwrite
    if os.path.isfile(output_path):
        if force:
            os.remove(output_path)
        else:
            raise FileExistsError(
                f"Output file already exists: {output_path}  — use --force to overwrite."
            )

    # Ensure parent directory exists
    out_dir = os.path.dirname(output_path)
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir)
        print(f"Created directory: {out_dir}")

    with h5py.File(input_path, "r") as input_file, \
         h5py.File(output_path, "w") as output_file:

        simulations = list(input_file.keys())
        print(f"Found {len(simulations)} simulation(s): {', '.join(simulations)}")

        for simulation in simulations:
            sim_in = input_file[simulation]
            print(f"\n── Simulation: {simulation} ──")

            sim_out = output_file.require_group(simulation)

            print("Copying attributes")
            for k, v in sim_in.attrs.items():
                sim_out.attrs[k] = v

            # Copy requested groups
            available = list(sim_in.keys())
            for group_name in to_extract:
                if group_name in available:
                    print(f"  Copying group: '{group_name}'")
                    if group_name not in sim_out:
                        input_file.copy(
                            sim_in[group_name],
                            sim_out,
                            name=group_name
                        )
                else:
                    warnings.warn(
                        f"  Group '{group_name}' not found in simulation '{simulation}' "
                        f"— skipping. Available: {', '.join(available)}"
                    )


def main():
    args = parse_args()
    to_extract = [s.strip() for s in args.diagnostics.split(",")]
    extract_data(args.input, args.output, to_extract, force=args.force)


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, FileExistsError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)