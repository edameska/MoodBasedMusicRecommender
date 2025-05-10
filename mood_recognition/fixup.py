import pandas as pd

# Load the two CSV files
file_1 = 'fer2013.csv'  # Replace with the actual path of your first file
file_2 = 'fer2013_corrected.csv'  # Replace with the actual path of your second file

# Read the CSV files into DataFrames
df1 = pd.read_csv(file_1)
df2 = pd.read_csv(file_2)

# Map the emotion labels to numbers
# Reversed emotion map
emotion_map_reversed = {
    'Angry': 0,
    'Disgust': 1,
    'Fear': 2,
    'Happy': 3,
    'Sad': 4,
    'Surprise': 5,
    'Neutral': 6
}

# Map the 'emotion_name' from df2 to the corresponding number
df2['emotion'] = df2['emotion_name'].map(emotion_map_reversed)


# Create a new DataFrame with 'emotion', 'pixels', and 'usage'
final_df = pd.DataFrame({
    'emotion': df2['emotion'],
    'pixels': df1['pixels'],  # Assuming 'pixels' exists in df1
    'Usage': df1['Usage']     # Assuming 'usage' exists in df1
})

# Save the new DataFrame to a new CSV file
final_df.to_csv('final_merged_file.csv', index=False)

print("CSV files merged and saved successfully as 'final_merged_file.csv'.")
