#!/bin/bash

# Arrays for mapping graph number to app name
declare -A apps=(
  [1]="endoscopy_depth_estimation_clahe"
  [2]="endoscopy_depth_estimation"
  [4]="body_pose_estimation"
  [3]="colonoscopy_segmentation"
  [5]="endoscopy_out_of_body_detection"
  [7]="multiai_endoscopy"
)
SPEED=${1:-default}
configs=($SPEED)

for config in "${configs[@]}"; do
  echo "Processing configuration: $config"

  for graph in "${!apps[@]}"; do
    app="${apps[$graph]}"
    script="change_wcet_${graph}.py"
    json_file="application_latencies_${config}/${app}_graph.json"
    output_dir="uppaal/UPPAALXML/${config}"
    output_file="${output_dir}/UPPAALGraph${graph}.xml"
    input_xml="uppaal/UPPAALXML/UPPAALGraph${graph}.xml"

    mkdir -p "$output_dir"

    echo "Running script $script on $json_file, output to $output_file"
    python ./uppaal/change_values/$script --json "$json_file" -o "$output_file" "$input_xml"
  done
done

echo "All done."
