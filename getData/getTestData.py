import pandas as pd

# This script formats the golden dataset to prepare it for use as a test set,
# for example, in a classification tool like Label Sleuth.
df = pd.read_csv("software_and_creative_outcomes.csv")

def merge_text(row):
    """
    Merges 'title' and 'description' columns into a single 'text' field.
    Handles cases where one or both of the source fields might be empty.
    """
    title = str(row.get("title", "")).strip()
    description = str(row.get("description", "")).strip()
    
    if title and description:
        return f"{title}. {description}"
    elif title:
        return title
    else:
        return description

# Create a new DataFrame with the required 'text' and 'label' columns.
new_df = pd.DataFrame()
new_df["text"] = df.apply(merge_text, axis=1)
new_df["label"] = df["is_software"]

# Save an intermediate file with the combined text and original label.
intermediate_filename = "label_sleuth_gold.csv"
new_df.to_csv(intermediate_filename, index=False, encoding="utf-8")
print(f"Intermediate file created: {intermediate_filename}, containing {len(new_df)} samples.")

# Add a category column for final formatting.
gold_df = pd.read_csv(intermediate_filename)

# Add a 'category_name' column, often required by labeling platforms.
gold_df["category_name"] = "Software"

gold_df = gold_df[["text", "category_name", "label"]]

# Save the final file
final_filename = "labelsleuth.csv"
gold_df.to_csv(final_filename, index=False, encoding="utf-8")
print(f"Conversion complete. Final file created: {final_filename}, containing {len(gold_df)} samples.")
