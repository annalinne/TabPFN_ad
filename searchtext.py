import os

def search_string_in_file(file_path, search_string):
    """Search for the given string in file and return lines containing that string,
    along with line numbers"""
    line_numbers = []
    lines_found = []
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
        for line_number, line in enumerate(file, 1):
            if search_string in line:
                line_numbers.append(line_number)
                lines_found.append(line.strip())
    return line_numbers, lines_found

def search_all_files(directory_path, search_string):
    """Search all files in the given directory recursively for the search string"""
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                line_numbers, lines_found = search_string_in_file(file_path, search_string)
                if line_numbers:
                    print(f"Found in {file_path}:")
                    for line_number, line in zip(line_numbers, lines_found):
                        print(f"  Line {line_number}: {line}")
            except Exception as e:
                print(f"Error reading {file_path}: {e}")

# Use the current working directory as the top-level directory to search
current_directory = os.getcwd()
search_string =  "Nan "
search_all_files(current_directory, search_string)
