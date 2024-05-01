import copy
import csv
import os
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

gkdir = os.environ.get("gkdir")
sys.path.append(gkdir)
import GraphKernel


def parse(filename, transpose):
    metadata = {}
    frames = []

    csvfile = open(filename, newline="")
    reader = csv.reader(csvfile)
    # get metadata
    for row in reader:
        if row[0].endswith("@@"):
            break  # End of metadata
        key, value = row[0].split(maxsplit=1)
        # convert dtype, substitute
        if key in (
            "ROWS",
            "COLS",
            "NOISE_THRESHOLD",
            "START_FRAME",
            "END_FRAME",
        ):
            metadata[key] = int(value)
        elif key in ("SECONDS_PER_FRAME"):
            metadata[key] = float(value)
        elif key in ("ROW_SPACING", "COL_SPACING", "SENSEL_AREA"):
            metadata[key] = float(value.split()[0])
            metadata["AREA_UNIT"] = value.split()[1]
        else:
            metadata[key] = value
    metadata["FILENAME_CSV"] = filename
    if transpose == True:
        temp = metadata["ROWS"]
        metadata["ROWS"] = metadata["COLS"]
        metadata["COLS"] = temp

        temp = metadata["ROW_SPACING"]
        metadata["ROW_SPACING"] = metadata["COL_SPACING"]
        metadata["COL_SPACING"] = temp
    if metadata["UNITS"] != "MPa":
        raise ValueError("Unit of sensor data is not in MPa")
    if metadata["AREA_UNIT"] == "cm2":  # convert cm2 to mm2
        metadata["ROW_SPACING"] = metadata["ROW_SPACING"] * 10
        metadata["COL_SPACING"] = metadata["COL_SPACING"] * 10
        metadata["SENSEL_AREA"] = metadata["SENSEL_AREA"] * 100
        metadata["AREA_UNIT"] = "mm2"

    # get frame data
    for row in reader:
        if not row:  # Skip empty lines
            continue
        if row[0].startswith("@@"):  # end of file
            break
        if row[0].startswith("Frame"):  # frame header
            frame_number = int(row[0].split()[1])
            elapsed_time = float(row[1])
            absolute_time = row[3]
            raw_sum = row[6].split("=")[1]
            sensor = []

            if raw_sum == "":
                raw_sum = 0
            else:
                raw_sum = float(raw_sum)

            for row in reader:  # sensor data
                if not row or row[0].startswith("@@"):  # end of current frame
                    frames.append(
                        {
                            "frame_number": frame_number,
                            "elapsed_time": elapsed_time,
                            "sensor": sensor,
                            "absolute_time": absolute_time,
                            "raw_sum": raw_sum,
                        }
                    )
                    break
                sensor.append([float(i) for i in row])

            # transpose if necessary
            if transpose == True:
                sensor = [list(x) for x in zip(*sensor)]

    csvfile.close()
    return metadata, frames


def contour(
    metadata,
    frame,
    tick_space=10,
    nlevels=9,
    vmin=0,
    vmax=0,
):
    fig, ax = GraphKernel.prepare(
        w="pp0.25",
        # r=1.0,
        font="arial",
        fontsize=14,
    )
    ax.set_aspect("equal", "box")  # same pixels for the same value in x and y

    # format data
    x = np.arange(
        0, metadata["COLS"] * metadata["COL_SPACING"], metadata["COL_SPACING"]
    )
    y = np.arange(
        0, metadata["ROWS"] * metadata["ROW_SPACING"], metadata["ROW_SPACING"]
    )
    y = y[::-1]
    X, Y = np.meshgrid(x, y)
    Z = frame["sensor"]

    # plotting
    if vmax == 0:
        vmax = np.ceil(np.max(Z))
    levels = np.linspace(vmin, vmax, nlevels + 1)
    CS = ax.contour(X, Y, Z, levels=levels, colors="black", linewidths=0)  # lines
    # ax.clabel(CS, inline=True) # value label
    CSf = ax.contourf(X, Y, Z, levels=levels, cmap="jet")  # fill color

    # color bar
    cbar = fig.colorbar(CSf, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("Pressure /" + metadata["UNITS"])
    if nlevels % 2 == 0:
        cbar_tick_interval = 2
    elif nlevels % 3 == 0:
        cbar_tick_interval = 3
    else:
        cbar_tick_interval = 1
    cbarlabels = levels[np.arange(0, len(levels), cbar_tick_interval, dtype=int)]
    cbarlabels = [np.round(i, 1) for i in cbarlabels]
    cbar.set_ticks(cbarlabels)
    cbar.set_ticklabels(cbarlabels)
    cbar.add_lines(CS)

    # export contour
    GraphKernel.finalize(
        fig,
        ax,
        fn_figout=metadata["FILENAME_CSV"].split(".csv")[0]
        + "_Frame"
        + str(frame["frame_number"])
        + "_contour.png",
        lims=[
            0,
            (metadata["COLS"] - 1) * metadata["COL_SPACING"],
            0,
            (metadata["ROWS"] - 1) * metadata["ROW_SPACING"],
        ],
        xspace=tick_space,
        yspace=tick_space,
    )

    plt.close(fig)


def findpeek(metadata, frames):
    raw_sum = [frames[i]["raw_sum"] for i in range(len(frames))]
    idx_max = np.argmax(raw_sum)
    return idx_max


def single_frame_stats(metadata, frame, output=False):
    min = np.min(frame["sensor"])
    max = np.max(frame["sensor"])
    dev = max - min
    ave = np.mean(frame["sensor"])
    med = np.median(frame["sensor"])
    error_p_ave = (max - ave) / ave * 100
    error_n_ave = (ave - min) / ave * 100
    error_p_med = (max - med) / med * 100
    error_n_med = (med - min) / med * 100

    if output == True:
        logfile = open(
            metadata["FILENAME_CSV"].split(".csv")[0]
            + "\\StatsResult"
            + datetime.now().strftime("_%y%m%d_%H%M")
            + ".txt",
            "w",
        )
        original_stdout = sys.stdout
        sys.stdout = logfile
    print(
        f'FSX file: {metadata["FILENAME"]}\nFrame {frame["frame_number"]}, {frame.get("trim_type", "All sensors")} (Htrim: {frame.get("htrim", "")}, Vtrim: {frame.get("vtrim_top", "")}, {frame.get("vtrim_bottom", "")})'
    )
    print(
        "Max, Min, Deviation, Mean, Median, Error+(from average), Error-(from average), Error+(from median), Error-(from median)"
    )
    print(
        f"{max:.3f}, {min:.3f}, {dev:.3f}, {ave:.3f}, {med:.3f}, {error_p_ave:.3f}, {error_n_ave:.3f}, {error_p_med:.3f}, {error_n_med:.3f}"
    )
    if output == True:
        sys.stdout = original_stdout


def trim(frames, htrim_sensel, vtrim_nsensel_top=None, vtrim_nsensel_bottom=None):
    frames_trimmed, frames_trimmed_averaged = copy.deepcopy(frames), copy.deepcopy(
        frames
    )
    for i in range(len(frames)):
        # save trimming setup
        frames_trimmed[i]["trim_type"] = "Trimmed sensor array"
        frames_trimmed[i]["htrim"] = htrim_sensel
        frames_trimmed[i]["vtrim_top"] = vtrim_nsensel_top
        frames_trimmed[i]["vtrim_bottom"] = vtrim_nsensel_bottom
        frames_trimmed_averaged[i]["trim_type"] = "Trimmed and averaged horizontally"
        frames_trimmed_averaged[i]["htrim"] = htrim_sensel
        frames_trimmed_averaged[i]["vtrim_top"] = vtrim_nsensel_top
        frames_trimmed_averaged[i]["vtrim_bottom"] = vtrim_nsensel_bottom

        htrimmed = htrim(frames_trimmed[i]["sensor"], htrim_sensel)
        frames_trimmed[i]["sensor"], frames_trimmed_averaged[i]["sensor"] = vtrim(
            htrimmed, vtrim_nsensel_top, vtrim_nsensel_bottom
        )
    return frames_trimmed, frames_trimmed_averaged


def htrim(sensor, htrim_sensel):
    htrimmed = []
    for row in sensor:
        htrimmed.append(row[htrim_sensel:-htrim_sensel])
    return htrimmed


def vtrim(htrimmed, vtrim_nsensel_top=None, vtrim_nsensel_bottom=None):
    averaged = np.zeros(len(htrimmed))
    for i, row in enumerate(htrimmed):
        averaged[i] = np.mean(row)

    if vtrim_nsensel_top == None:  # define nsensel to trim
        plt.scatter(np.arange(1, len(averaged) + 1, 1), averaged)
        plt.show(block=False)
        vtrim_nsensel_top = int(input("How many sensels to be trimmed at the TOP?\n"))
        vtrim_nsensel_bottom = int(
            input("How many sensels to be trimmed at the BOTTOM?\n")
        )
        plt.close()

    trimmed_sensor = htrimmed.copy()[vtrim_nsensel_top:-vtrim_nsensel_bottom]
    trimmed_averaged = averaged[vtrim_nsensel_top:-vtrim_nsensel_bottom]
    return trimmed_sensor, trimmed_averaged


def pressure_history(metadata, frames):
    # parameters
    ave_color = "royalblue"
    ave_barcolor = "lightsteelblue"
    med_color = "seagreen"
    med_barcolor = "lightgreen"

    min, max, dev, ave, med, error_p_ave, error_n_ave, error_p_med, error_n_med = (
        np.zeros(len(frames)) for i in range(9)
    )

    for i, frame in enumerate(frames):
        min[i] = np.min(frame["sensor"])
        max[i] = np.max(frame["sensor"])
        dev[i] = max[i] - min[i]
        ave[i] = np.mean(frame["sensor"])
        med[i] = np.median(frame["sensor"])
        error_p_ave[i] = (max[i] - ave[i]) / ave[i] * 100
        error_n_ave[i] = (ave[i] - min[i]) / ave[i] * 100
        error_p_med[i] = (max[i] - med[i]) / med[i] * 100
        error_n_med[i] = (med[i] - min[i]) / med[i] * 100

    fig, ax = GraphKernel.prepare(
        w="pp0.25", font="arial", fontsize=14, xtitle="Frame", ytitle="Pressure /MPa"
    )

    # plotting
    x = np.arange(1, len(frames) + 1, 1)
    # ax.scatter(x,ave,label="Average")
    ax.errorbar(
        x,
        ave,
        (
            error_n_ave * 0.01 * ave,
            error_p_ave * 0.01 * ave,
        ),
        label="Average",
        ecolor=ave_barcolor,
        elinewidth=2,
        c=ave_color,
    )
    # ax.scatter(x,med, label="Median")
    ax.errorbar(
        x,
        med,
        (
            error_n_med * 0.01 * med,
            error_p_med * 0.01 * med,
        ),
        label="Median",
        ecolor=med_barcolor,
        elinewidth=2,
        c=med_color,
    )

    ax.legend()

    plt.show(block=False)
    # plt.show(block=False)
    plt.close()

    # export contour
    GraphKernel.finalize(
        fig,
        ax,
        fn_figout=metadata["FILENAME_CSV"].split(".csv")[0]
        + "_PressureDistHistory.png",
        lims=[
            0,
            None,
            0,
            None,
        ],
    )
    plt.close(fig)


def debug():
    # parameters
    htrim_sensel = 10
    csvfile = os.getcwd() + r"\iscan_test.csv"
    transpose = False

    # load sensor data
    metadata, frames = parse(csvfile, transpose=transpose)

    # have a rough idea of pressure distribution
    # pressure_history()

    # locate peek, determine vtrim
    idx_max = findpeek(metadata, frames)
    # trim([frames[idx_max]], htrim_sensel=htrim_sensel)
    # contour(metadata, frames[idx_max])

    # trim edges of sensor matrix
    frames_trimmed, frames_trimmed_averaged = trim(
        frames,
        htrim_sensel=htrim_sensel,
        vtrim_nsensel_top=4,
        vtrim_nsensel_bottom=7,
    )

    # stats of a frame
    single_frame_stats(metadata, frames_trimmed_averaged[88], output=True)

    # time series of pressure error
    pressure_history(metadata, frames_trimmed_averaged)

    # # Print metadata
    # print("Metadata:")
    # for key, value in metadata.items():
    #     print(f"{key}: {value}")

    # # Print list of elapsed time
    # [frame["elapsed_time"] for frame in frames]

    # # Print first few frames
    # print("\nFrames:")
    # for frame in frames[:3]:
    #     print(f"Frame {frame['frame_number']} - Elapsed time: {frame['elapsed_time']}")
    #     for row in frame["sensor"]:
    #         print(row)
    #     print()


debug()
