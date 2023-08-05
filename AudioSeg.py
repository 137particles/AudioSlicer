from datetime import datetime, timedelta
import json
import os
import warnings

import numpy as np
from scipy.io import wavfile
from scipy.io.wavfile import WavFileWarning
from tqdm import tqdm
from colorama import Fore, Style
from prettytable import PrettyTable
import argparse

def get_time(audio_seconds):
    if audio_seconds < 0:
        return 00
    else:
        sec = timedelta(seconds=float(audio_seconds))
        d = datetime(1, 1, 1) + sec
        return f"{str(d.hour).zfill(2)}:{str(d.minute).zfill(2)}:{str(d.second).zfill(2)}.001"


def windows(signal, window_size, step_size):
    if type(window_size) is not int or type(step_size) is not int:
        raise AttributeError("Window size and step size must be integers.")
    for i_start in range(0, len(signal), step_size):
        i_end = i_start + window_size
        if i_end >= len(signal):
            break
        yield signal[i_start:i_end]


def energy(samples):
    return np.sum(np.power(samples, 2.)) / float(len(samples))


def rising_edges(binary_signal):
    previous_value = 0
    for index, x in enumerate(binary_signal):
        if x and not previous_value:
            yield index
        previous_value = x


def parse_arguments():
    parser = argparse.ArgumentParser(description="Split a WAV file into segments based on silence.")
    parser.add_argument("--input", help="Path to the input WAV file.")
    parser.add_argument("--output", help="Path to the output directory.")
    parser.add_argument("--min_silence_length", type=float, default=0.6,
                        help="The minimum length of silence at which a split may occur [seconds]. Defaults to 0.6 seconds.")
    parser.add_argument("--min_audio_length", type=int, default=3,
                        help="Minimum accepted audio length in seconds.")
    parser.add_argument("--max_audio_length", type=int, default=9,
                        help="Maximum accepted audio length in seconds.")
    parser.add_argument("--discard_outliers", type=int, default=1,
                        help=f"Do not write a file if it is outside the accepted file size range of min_audio_length to max_audio_length seconds.")
    
    args = parser.parse_args()

    if args.input is None or args.output is None:
        script_name = os.path.basename(__file__)

        print(f"Error: Missing required arguments. Run with " + Fore.BLUE + script_name + " --help" + Style.RESET_ALL + " for usage information.")
        exit(1)

    return args


def print_arguments_table(args):
    description_mapping = {
        "input": "Input WAV File",
        "output": "Output Directory",
        "min_silence_length": "Minimum Silence Length (seconds)",
        "discard_outliers": f"Discard Outliers ({args.min_audio_length}-{args.max_audio_length} seconds range)",
        "min_audio_length": "Minimum Accepted Audio Length (seconds)",
        "max_audio_length": "Maximum Accepted Audio Length (seconds)"
    }

    table = PrettyTable()
    table.field_names = ["Argument Description", "Value"]
    table.align["Argument Description"] = "l"
    table.align["Value"] = "l"

    for arg_name, arg_value in vars(args).items():
        description = description_mapping.get(arg_name, arg_name)
        table.add_row([description, arg_value])

    print(table)


def main():
    args = parse_arguments()
    print_arguments_table(args)

    # Check if input file exists
    if not os.path.isfile(args.input):
        print(f"Error: Input file {args.input} does not exist.")
        sys.exit(1)

    # Check if output directory exists, if not, create it
    if not os.path.isdir(args.output):
        try:
            os.makedirs(args.output)
        except OSError:
            print(f"Error: Could not create output directory {args.output}.")
            sys.exit(1)

    # Change the arguments and the input file here
    input_file = args.input
    output_dir = args.output
    min_silence_length = args.min_silence_length  	# The minimum length of silence at which a split may occur [seconds]. Defaults to 3 seconds.
    discard_outliers = args.discard_outliers
    min_audio_length = args.min_audio_length
    max_audio_length = args.max_audio_length

    silence_threshold = 1e-4  			# The energy level (between 0.0 and 1.0) below which the signal is regarded as silent.
    step_duration = 0.03/10   			# The amount of time to step forward in the input file after calculating energy. Smaller value = slower, but more accurate silence detection. Larger value = faster, but might miss some split opportunities. Defaults to (min-silence-length / 10.).

    input_filename = input_file
    window_duration = min_silence_length
    if step_duration is None:
        step_duration = window_duration / 10.
    else:
        step_duration = step_duration

    output_filename_prefix = os.path.splitext(os.path.basename(input_filename))[0]
    dry_run = False

    print("\nSplitting {} where energy is below {}% for longer than {}s.\n".format(
        input_filename,
        silence_threshold * 100.,
        window_duration
        )
    )

    # Read and split the file
    # Suppress the specific warning
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=WavFileWarning)
        
        # Read the WAV file
        sample_rate, samples = wavfile.read(filename=input_filename, mmap=True)

    # Print a custom message if needed
    print("WAV file read successfully. Any non-standard chunks were ignored.")

    max_amplitude = np.iinfo(samples.dtype).max
    print(f"Amplitude \t(max): {max_amplitude}")

    max_energy = energy([max_amplitude])
    print(f"Energy \t\t(max): {max_energy}\n")

    window_size = int(window_duration * sample_rate)
    step_size = int(step_duration * sample_rate)

    signal_windows = windows(
        signal=samples,
        window_size=window_size,
        step_size=step_size
    )

    progress_bar_energy = tqdm(total=int(len(samples) / float(step_size)), desc=Fore.BLUE + "Calculating window energy" + Style.RESET_ALL)

    window_energy = []
    for w in signal_windows:
        energy_value = energy(w) / max_energy
        window_energy.append(energy_value)
        progress_bar_energy.update(1)  # Update the progress bar

    progress_bar_energy.close()  # Close the progress bar when done

    window_silence = (e > silence_threshold for e in window_energy)

    cut_times = (r * step_duration for r in rising_edges(window_silence))

    # Convert cut_times to a list so we can get its length
    cut_times_list = list(cut_times)

    # Create a tqdm object for the second progress bar with a description
    progress_bar_silences = tqdm(total=len(cut_times_list), desc=Fore.BLUE + "Finding silences" + Style.RESET_ALL)

    cut_samples = [int(t * sample_rate) for t in cut_times_list]
    for _ in cut_times_list:
        progress_bar_silences.update(1)  # Update the progress bar

    # Append the total length of the samples to the cut_samples list
    cut_samples.append(len(samples))

    # Close the second progress bar when done
    progress_bar_silences.close()

    cut_ranges = [(i, cut_samples[i], cut_samples[i+1]) for i in range(len(cut_samples) - 1)]


    audio_sub = {str(i) : [str(get_time(((cut_samples[i])/sample_rate))), 
                        str(get_time(((cut_samples[i+1])/sample_rate)))] 
                for i in range(len(cut_samples) - 1)}

    # Counter for the number of files created
    files_created = 0

    # List to store file information
    file_info_list = []

    for i, start, stop in cut_ranges:
        output_file_path = "{}_{:03d}.wav".format(
            os.path.join(output_dir, output_filename_prefix),
            i
        )
        duration_seconds = (stop - start) / sample_rate
        duration_minutes, duration_seconds = divmod(duration_seconds, 60)

        is_outlier = False

        if (duration_seconds < min_audio_length or duration_seconds > max_audio_length) and discard_outliers != 0:
            duration_str = Fore.RED + f"{int(duration_minutes)}:{int(duration_seconds):02}" + Style.RESET_ALL
            is_outlier = True
        else:
            duration_str = f"{int(duration_minutes)}:{int(duration_seconds):02}"

        if not dry_run and (discard_outliers == 0 or (discard_outliers != 0 and is_outlier == False)):
            wavfile.write(
                filename=output_file_path,
                rate=sample_rate,
                data=samples[start:stop]
            )
            files_created += 1  # Increment the counter
            file_info_list.append((output_file_path, duration_str))  # Store the information
        else:
            file_info_list.append((output_file_path, "Not written"))  # Store the information

    # Print the file information
    for file_info in file_info_list:
        print(f"{Fore.WHITE}Writing file {file_info[0]} with duration {file_info[1]}{Style.RESET_ALL}")

    print(f"{Fore.GREEN}{files_created} files were created in the directory \"{output_dir}{Style.RESET_ALL}\"")

    with open (output_dir+'\\'+output_filename_prefix+'.json', 'w') as output:
        json.dump(audio_sub, output)

if __name__ == "__main__":
    main()
