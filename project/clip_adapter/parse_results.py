import re
import numpy as np
import os.path as osp
import argparse
from collections import OrderedDict, defaultdict

from dassl.utils import check_isfile, listdir_nohidden

def parse_function(*metrics, directory="", args=None, end_signal=None):
    print(f"Parsing files in {directory}")
    subdirs = listdir_nohidden(directory, sort=True)

    outputs = []

    for subdir in subdirs:
        fpath = osp.join(directory, subdir, "log.txt")
        assert check_isfile(fpath)
        good_to_go = False
        output = OrderedDict()

        with open(fpath, "r") as f:
            lines = f.readlines()

            for line in lines:
                line = line.strip()

                if line == end_signal:
                    good_to_go = True

                for metric in metrics:
                    match = metric["regex"].search(line)
                    if match and good_to_go:
                        if "file" not in output:
                            output["file"] = fpath
                        num = float(match.group(1))
                        name = metric["name"]
                        output[name] = num

        if output:
            outputs.append(output)

    assert len(outputs) > 0, f"Nothing found in {directory}"

    metrics_results = defaultdict(list)

    for output in outputs:
        msg = ""
        for key, value in output.items():
            if isinstance(value, float):
                msg += f"{key}: {value:.2f}%. "
            else:
                msg += f"{key}: {value}. "
            if key != "file":
                metrics_results[key].append(value)
        print(msg)

    output_results = OrderedDict()

    print("===")
    print(f"Summary of directory: {directory}")
    for key, values in metrics_results.items():
        avg = np.mean(values)
        std = np.std(values)
        print(f"* {key}: {avg:.2f}% +- {std:.2f}%")
        output_results[key] = avg
    print("===")

    return output_results


def main(args, end_signal):
    metric = {
        "name": args.keyword,
        "regex": re.compile(fr"\* {args.keyword}: ([\.\deE+-]+)%"),
    }

    if args.multi_exp:
        final_results = defaultdict(list)

        for directory in listdir_nohidden(args.directory, sort=True):
            directory = osp.join(args.directory, directory)
            results = parse_function(
                metric, directory=directory, args=args, end_signal=end_signal
            )

            for key, value in results.items():
                final_results[key].append(value)

        print("Average performance")
        for key, values in final_results.items():
            avg = np.mean(values)
            print(f"* {key}: {avg:.2f}%")

    else:
        parse_function(
            metric, directory=args.directory, args=args, end_signal=end_signal
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=str, help="path to directory")
    parser.add_argument("--test-log", action="store_true", help="parse test-only logs")
    parser.add_argument(
        "--multi-exp", action="store_true", help="parse multiple experiments"
    )
    parser.add_argument(
        "--keyword", default="accuracy", type=str, help="which keyword to extract"
    )
    args = parser.parse_args()

    end_signal = "Finish training"
    if args.test_log:
        end_signal = "=> result"

    main(args, end_signal)
