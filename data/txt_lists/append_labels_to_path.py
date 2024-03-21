input_file_path = "m3fd_test.txt"
output_file_path = "m3fdtest.txt"

# Dictionary to map class names to labels
class_labels = {"bus": 0, "car": 1, "lamp": 2, "motorcycle": 3, "people": 4, "truck": 5}  # Add more classes as needed

# Read the original file and process each line
with open(input_file_path, 'r') as infile, open(output_file_path, 'w') as outfile:
    for line in infile:
        # Remove trailing whitespace and split the line by '/'
        parts = line.strip().split('/')

        # Extract the class name from the path
        class_name = parts[-2]

        # Extract the file name
        file_name = parts[-1]

        # Get the label for the class name from the dictionary
        label = class_labels.get(class_name, -1)  # Default to -1 if class not found

        # If label is found, append it to the line, otherwise, ignore the line
        if label != -1:
            # Append the label to the line and write it to the output file
            modified_line = f"{line.strip()} {label}\n"
            outfile.write(modified_line)

print("File modification complete.")

