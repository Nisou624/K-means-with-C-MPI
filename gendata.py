import pandas as pd
import argparse

# Set up argument parser
parser = argparse.ArgumentParser(description='Create input.txt from CSV for K-means clustering.')
parser.add_argument('csv_file', type=str, help='Path to the input CSV file')
parser.add_argument('num_rows', type=int, help='Number of rows to use from the CSV file')
parser.add_argument('x_column', type=str, help='Column name for the x-coordinate')
parser.add_argument('y_column', type=str, help='Column name for the y-coordinate')

# Parse arguments
args = parser.parse_args()

# Read the CSV file
df = pd.read_csv(args.csv_file)

# Select the relevant columns for clustering
x_column = args.x_column
y_column = args.y_column

# Open the output file
with open('input.txt', 'w') as f:
    # Write the specified number of data points to the file
    for index, row in df.iterrows():
        if index >= args.num_rows:
            break
        f.write(f"{row[x_column]} {row[y_column]}\n")

print(f"input.txt file has been created successfully with {args.num_rows} rows of data.")
