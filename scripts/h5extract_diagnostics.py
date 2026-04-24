#!/usr/bin/env python3
"""
Extract specific diagnostics from an HDF5 file into a new HDF5 file,
while preserving the metadata.

Single file mode:
    python extract-diagnostics.py -p /data -i input.h5 -o output.h5 -d "All probe,ExB CFL"

Batch mode:
    python extract-diagnostics.py -p /data --files input1.h5 input2.h5 --suffix probes -d "All probe"

Filter by simulation index:
    python extract-diagnostics.py -p /data -i input.h5 -o output.h5 -d "All probe" -n 0:3
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

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--input", "-i",
        help="Path to a single input HDF5 file"
    )
    mode.add_argument(
        "--files", "-F",
        nargs="+",
        metavar="FILE",
        help="One or more input filenames (used together with --path and --suffix)"
    )

    parser.add_argument(
        "--path", "-p",
        default=".",
        help="Base directory for input and output files (default: current directory)"
    )
    parser.add_argument(
        "--suffix", "-s",
        default="extracted",
        help='Suffix to append to each filename for the output (default: "extracted", produces e.g. input1_extracted.h5)'
    )
    parser.add_argument(
        "--output", "-o",
        help="Path to the output HDF5 file (single file mode only)"
    )
    parser.add_argument(
        "--diagnostics", "--groups", "-d", "-g",
        required=True,
        dest="diagnostics",
        metavar="DIAGNOSTICS",
        help='Comma-separated list of group names to extract (e.g. "All probe,ExB CFL")'
    )
    parser.add_argument(
        "--index", "-n",
        default=None,
        metavar="INDEX",
        help="Select simulations by index: a single index (-n 0), a comma-separated list (-n 0,2,4), or a slice (-n 0:5 or -n ::2). Negative indices are supported (-n -1 for last). Default: all simulations."
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Overwrite output file(s) if they already exist"
    )

    args = parser.parse_args()

    if args.input and not args.output:
        parser.error("--input requires --output")

    if args.input and args.suffix != "extracted":
        warnings.warn("--suffix is ignored in single file mode, use --output to specify the output path.")

    if args.files and args.output:
        warnings.warn("--output is ignored in batch mode, output filenames are derived from --files and --suffix.")

    return args


def parse_index(index_str, total):
    """Parse an index string into a set of valid integer indices."""
    index_str = index_str.strip()

    if ":" in index_str:
        parts = [int(p) if p else None for p in index_str.split(":")]
        return set(range(*slice(*parts).indices(total)))

    indices = set()
    for part in index_str.split(","):
        i = int(part.strip())
        if i < 0:
            i += total
        if 0 <= i < total:
            indices.add(i)
        else:
            warnings.warn(f"Index {int(part.strip())} out of range (0–{total - 1}) - skipping.")
    return indices


def extract_data(input_path, output_path, to_extract, index_str=None, force=False):
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if os.path.isfile(output_path):
        if force:
            os.remove(output_path)
        else:
            raise FileExistsError(
                f"Output file already exists: {output_path}  — use --force to overwrite."
            )

    out_dir = os.path.dirname(output_path)
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir)
        print(f"Created directory: {out_dir}")

    with h5py.File(input_path, "r") as input_file, \
         h5py.File(output_path, "w") as output_file:

        all_simulations = list(input_file.keys())
        print(f"Found {len(all_simulations)} simulation(s):")
        for i, s in enumerate(all_simulations):
            print(f"  [{i}] {s}")

        if index_str is not None:
            index_set = parse_index(index_str, len(all_simulations))
            simulations = [all_simulations[i] for i in index_set]
        else:
            simulations = all_simulations

        if not simulations:
            warnings.warn("No simulations remaining after filtering - output file will be empty.")

        for simulation in simulations:
            sim_in = input_file[simulation]
            print(f"\n── Simulation: {simulation} ──")

            sim_out = output_file.require_group(simulation)

            print("  Copying attributes")
            for k, v in sim_in.attrs.items():
                sim_out.attrs[k] = v

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


def make_output_path(base_dir, filename, suffix):
    """Insert suffix before the file extension, e.g. input1.h5 -> input1_probes.h5"""
    name, ext = os.path.splitext(filename)
    return os.path.join(base_dir, f"{name}_{suffix}{ext}")


def main():
    args = parse_args()
    to_extract = [s.strip() for s in args.diagnostics.split(",")]

    if args.input:
        # Single file mode
        input_path = os.path.join(args.path, args.input)
        output_path = os.path.join(args.path, args.output)
        extract_data(input_path, output_path, to_extract,
                     index_str=args.index, force=args.force)
    else:
        # Batch mode
        errors = []
        for filename in args.files:
            input_path = os.path.join(args.path, filename)
            output_path = make_output_path(args.path, filename, args.suffix)
            print(f"\n{'='*60}")
            print(f"Processing: {input_path} → {output_path}")
            print('='*60)
            try:
                extract_data(input_path, output_path, to_extract,
                             index_str=args.index, force=args.force)
            except (FileNotFoundError, FileExistsError) as e:
                print(f"Error: {e}", file=sys.stderr)
                errors.append(filename)

        if errors:
            print(f"\nFailed for {len(errors)} file(s): {', '.join(errors)}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, FileExistsError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)