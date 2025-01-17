# Credit
This repo is derived from the code https://github.com/henrymaas/AudioSlicer with credit going to them for the original codebase.

# Our Changes
The code has been modified to add command line arguments, improved output, --help options and ability to reject files for being too long or short.  

# Command Line Parameters

- `--input`
  - **Description**: Path to the input WAV file.
s
- `--output`
  - **Description**: Path to the output directory.

- `--min_silence_length`
  - **Type**: Float
  - **Default**: 0.6 seconds
  - **Description**: The minimum length of silence at which a split may occur. Defaults to 0.6 seconds.

- `--min_audio_length`
  - **Type**: Integer
  - **Default**: 3 seconds
  - **Description**: Minimum accepted audio length in seconds.

- `--max_audio_length`
  - **Type**: Integer
  - **Default**: 9 seconds
  - **Description**: Maximum accepted audio length in seconds.

- `--discard_outliers`
  - **Type**: Integer
  - **Default**: 1
  - **Description**: Do not write a file if it is outside the accepted file size range of `min_audio_length` to `max_audio_length` seconds.

# AudioSlicer

A simple Audio Slicer in Python which can split .wav audio files into multiple .wav samples, based on silence detection. Also, it dumps a .json that contains the periods of time in which the slice occours, in the following format: 

{sample nº : [cut start, cut end]}. Ex.:

{"0": ["0:0:0", "0:0:3"], "1": ["0:0:3", "0:0:10"], "2": ["0:10:0", "0:0:22"], "3": ["0:0:22", "0:0:32"]}

The code was taken from <b>/andrewphillipdoss.</b> Thanks!

The filename will also contains the parts when the video were sliced, ex.: sample01_0349_0401.wav


<h2> AI Adaptation </h2>
This project will turn into a neural network which can detect audio silence and split the files.
It will also needs to learn to detect 'breathing noises' from the dictator and remove from it.



<h2> Python 3.11.0 </h2>

numpy (1.24.1)

scypi (1.10.0)

tqdm (4.64.1)


<h2> Usage </h2>
To run this code, just change the path of the <b>input_file</b> and <b>output_dir</b> inside the code.
<br/><br/>
:exclamation:Ps: Please note that in order for your audio file to be cut into samples, it should contain periods of "silence". If you are trying to extract voice samples from a song, for example, it may not work as expected.
<br /><br />
Depending on the level of noise in your audio, the algorithm may skip the silence windows, resulting in missed cuts. Ensure that your audio is free from unwanted noise and that the silences are clearly defined. You can adjust the parameters of <b>min_silence_length</b>, <b>silence_threshold</b>, and <b>step_duration</b> to modify the length, amplitude, and duration of the silence window in order to better match your audio
