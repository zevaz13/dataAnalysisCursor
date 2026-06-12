# EEG Analysis — Working Plan

## Current milestone — COMPLETE

## Next milestone - FBCCA Stim organization — COMPLETE
- Separate stimuli in baselines stims 1, 2, 103, 104 (indexed from one)
- Get the task FBCCA stims 3:102 (should be 100)
- Transform the stimulus vector into a 10 by 10 stimulus matrix. the file sequence.txt includes the sequence of indices for each stimulations, each of these corresponds to the index in red_array, green_array respectively.
- The new matrix should be ploted as a heatmap with x axis as red_array, y axis as green_array, and the color corresponds to the ssvep amplitude for each one
- produce a plot for the baselines. A boxplot with the four of them is good.
- produce the heatmap plot for the stimulus matrix. 
- Don't forget to also create the corresponding notebook. You may add new cells to full_pipeline_fbcca.ipynb