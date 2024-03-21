#!/bin/bash

# Check if two arguments were provided
# Make the script executable by running chmod +x test_cgan.sh in the terminal.
# Execute the script with the desired range of epochs, like ./test_cgan.sh 1 60

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <start_epoch> <end_epoch>"
    exit 1
fi

START_EPOCH=$1
END_EPOCH=$2
LOG_FILE="/home/david/workingDIR/pytorch-CycleGAN-and-pix2pix/analysis_results/production_O21CVPL00001_13_01_16_v1/command_outputs.txt"
SCORES_FILE="/home/david/workingDIR/pytorch-CycleGAN-and-pix2pix/analysis_results/production_O21CVPL00001_13_01_16_v1/scores.txt"
SORTED_SCORES_FILE="/home/david/workingDIR/pytorch-CycleGAN-and-pix2pix/analysis_results/production_O21CVPL00001_13_01_16_v1/sorted_scores.txt"

# Clear previous files
> "$LOG_FILE"
> "$SCORES_FILE"

cd /home/david/workingDIR/pytorch-CycleGAN-and-pix2pix

# Loop and run the command for each epoch in the range
for (( epoch=START_EPOCH; epoch<=END_EPOCH; epoch++ ))
do
    echo "Running epoch $epoch"
   
    # COMMAND="python3 /home/david/workingDIR/pytorch-CycleGAN-and-pix2pix/test.py --dataroot /home/david/workingDIR/pytorch-CycleGAN-and-pix2pix/datasets/cGAN_input_float_20231128_v4 --name v4_FloatTest_lr10-4_batch1 --model pix2pix --direction BtoA --epoch $epoch --preprocess none --netG resnet_9blocks --netD pixel --dataset_mode aligned"
    COMMAND="python3 /home/david/workingDIR/pytorch-CycleGAN-and-pix2pix/test.py --dataroot /home/david/workingDIR/pytorch-CycleGAN-and-pix2pix/datasets/cGAN_input_O21CVPL00001_13_01_16 --name production_O21CVPL00001_13_01_16_v1 --model pix2pix --direction BtoA --epoch $epoch --preprocess none --netG resnet_9blocks --netD pixel --dataset_mode aligned"
    # Run the command and capture the output
    $COMMAND > temp_output.txt 
    
    # Append the command and its output to the log file
    echo "$COMMAND" >> "$LOG_FILE"
    cat temp_output.txt >> "$LOG_FILE"
    
    # Extract the last line (score) and save it
    echo "$COMMAND $(tail -n 1 temp_output.txt)" >> "$SCORES_FILE"
done

# Sort the scores and save to a file
sort -k2 "$SCORES_FILE" > "$SORTED_SCORES_FILE"

# Clean up
rm temp_output.txt

echo "Completed all epochs. Outputs are saved in $LOG_FILE."
echo "Scores are saved in $SCORES_FILE and sorted scores in $SORTED_SCORES_FILE."
