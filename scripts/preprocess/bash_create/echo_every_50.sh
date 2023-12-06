#!/bin/bash

# Check if a file is provided
if [ $# -ne 1 ]; then
    echo "Usage: $0 <filename>"
    exit 1
fi

# The word to echo every 50 lines
word="wait"

# File to read
file="$1"

# Counter for lines
counter=0

# Read the file line by line
while IFS= read -r line
do
    # Print the current line
    echo "$line"

    # Increment the counter
    ((counter++))

    # If counter reaches 50, echo the word and reset the counter
    if [ $counter -eq 50 ]; then
        echo $word
        counter=0
    fi
done < "$file"

