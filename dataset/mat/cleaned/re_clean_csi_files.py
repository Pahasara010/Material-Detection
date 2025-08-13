import os
import re
import shutil

def clean_csi_file(input_path, output_path=None):
    if output_path is None:
        output_path = input_path  # Overwrite the original file by default
    
    valid_lines = []
    with open(input_path, 'r') as f:
        for line in f:
            # Extract the array part within quotes
            array_match = re.search(r'"\[(.*?)\]"', line)
            if array_match:
                array_str = array_match.group(1)
                # Try to convert all values to integers, skipping invalid ones
                csi_values = []
                for x in array_str.split(','):
                    try:
                        csi_values.append(int(x))
                    except ValueError:
                        break  # Stop processing if any value is invalid
                else:
                    # Check if the length matches the expected value (e.g., 256)
                    if len(csi_values) == 256:  # Adjust this based on your expected length
                        valid_lines.append(line)
                    else:
                        print(f"Skipping line in {input_path} with length {len(csi_values)}")
            else:
                print(f"Skipping line in {input_path} with no valid array format")

    # Write cleaned data to the output file
    with open(output_path, 'w') as f:
        for line in valid_lines:
            f.write(line)
    print(f"Cleaned {input_path} - {len(valid_lines)} valid lines retained")

def main():
    data_dir = r"dataset/mat/cleaned"
    backup_dir = "dataset/mat/cleaned_backup"
    
    # Create backup directory if it doesn't exist
    if not os.path.exists(backup_dir):
        os.makedirs(backup_dir)
    
    # Process each .cleaned file
    for filename in os.listdir(data_dir):
        if filename.endswith('.cleaned'):
            input_path = os.path.join(data_dir, filename)
            # Create backup before cleaning
            backup_path = os.path.join(backup_dir, filename)
            shutil.copy2(input_path, backup_path)
            print(f"Backed up {filename} to {backup_path}")
            # Clean the file
            clean_csi_file(input_path)

if __name__ == "__main__":
    main()