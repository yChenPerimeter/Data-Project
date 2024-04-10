#!/bin/bash

# Check if two arguments were provided
# Make sure change SAVE DIR and command 
# Make the script executable by running chmod +x test_cgan.sh in the terminal.
# Execute the script with the desired range of epochs, like ./test_cGAN.sh 1 60

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <start_epoch> <end_epoch>"
    exit 1
fi

START_EPOCH=$1
END_EPOCH=$2
SAVE_DIR="/home/david/workingDIR/pytorch-CycleGAN-and-pix2pix/analysis_results/production_O21CVPL00001_13_01_16"
LOG_FILE="${SAVE_DIR}/command_outputs.txt"
SCORES_FILE="${SAVE_DIR}/scores.txt"
SORTED_SCORES_FILE="${SAVE_DIR}/sorted_scores.txt"

# Clear previous files
> "$LOG_FILE"
> "$SCORES_FILE"

cd /home/david/workingDIR/pytorch-CycleGAN-and-pix2pix

# Loop and run the command for each epoch in the range
for (( epoch=START_EPOCH; epoch<=END_EPOCH; epoch++ ))
do
    echo "Running epoch $epoch"
   
    # COMMAND="python3 /home/david/workingDIR/pytorch-CycleGAN-and-pix2pix/test.py --dataroot /home/david/workingDIR/pytorch-CycleGAN-and-pix2pix/datasets/cGAN_input_float_20231128_v4 --name v4_FloatTest_lr10-4_batch1 --model pix2pix --direction BtoA --epoch $epoch --preprocess none --netG resnet_9blocks --netD pixel --dataset_mode aligned"
    COMMAND="python3 /home/david/workingDIR/pytorch-CycleGAN-and-pix2pix/test.py --dataroot /home/david/workingDIR/pytorch-CycleGAN-and-pix2pix/datasets/cGAN_input_O21CVPL00001_13_01_16 --name production_O21CVPL00001_13_01_16 --model pix2pix --direction AtoB --epoch $epoch --preprocess none --netG resnet_9blocks --netD pixel --dataset_mode aligned"
    #COMMAND="python3 /home/david/workingDIR/pytorch-CycleGAN-and-pix2pix/test.py --dataroot /home/david/workingDIR/pytorch-CycleGAN-and-pix2pix/datasets/cGAN_input_O21CVPL00001_13_01_16 --name production_O21CVPL00001_13_01_16_v1 --model pix2pix --direction AtoB --epoch $epoch --preprocess none --netG resnet_9blocks --netD pixel --dataset_mode aligned"
    #COMMAND="python3 /home/david/workingDIR/pytorch-CycleGAN-and-pix2pix/test.py --dataroot /home/david/workingDIR/pytorch-CycleGAN-and-pix2pix/datasets/cGAN_input_O21CV00001_13_01_16 --name production_O21CV00001_13_01_16 --model pix2pix --direction AtoB --epoch $epoch --preprocess none --netG resnet_9blocks --netD pixel --dataset_mode aligned"
    # Test result after rearranging the input data, removed outliers data
    #COMMAND="python3 /home/david/workingDIR/pytorch-CycleGAN-and-pix2pix/test.py --dataroot /home/david/workingDIR/pytorch-CycleGAN-and-pix2pix/datasets/cGAN_input_O21CVPL00001_13_01_16 --name production_O21CVPL_rmNans_v2 --model pix2pix --direction AtoB --epoch $epoch --preprocess none --netG resnet_9blocks --netD pixel --dataset_mode aligned"
    # Run the command and capture the output
    $COMMAND > ${SAVE_DIR}/temp_output.txt 
    
    # Append the command and its output to the log file
    echo "$COMMAND" >> "$LOG_FILE"
    cat ${SAVE_DIR}/temp_output.txt >> "$LOG_FILE"
    
    # Extract the last line (score) and save it
    echo "$COMMAND $(tail -n 1 ${SAVE_DIR}/temp_output.txt)" >> "$SCORES_FILE"
done

# Sort the scores and save to a file
sort -k2 "$SCORES_FILE" > "$SORTED_SCORES_FILE"

# Clean up
rm ${SAVE_DIR}/temp_output.txt

echo "Completed all epochs. Outputs are saved in $LOG_FILE."
echo "Scores are saved in $SCORES_FILE and sorted scores in $SORTED_SCORES_FILE."