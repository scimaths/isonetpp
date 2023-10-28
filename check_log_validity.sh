logDir=$1
for log_file in $logDir/*
do
    diff=$(awk '/Run: [0-9]+/ {last_run_line=$0; last_run_line_number=NR} /best/ {last_best_line_number=NR} END {print last_run_line_number - last_best_line_number}' $log_file)
    [ "$diff" -ne 202 ] && echo "File: $log_file, Difference: $diff"
done