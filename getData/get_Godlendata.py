import sqlite3
import pandas as pd

# Define the path to the database.
# Note: This should be the database file that contains the 'outcomes' table
# populated by the fetch_outcomes.py script.
db_path = "projects_sample.db" 

# Connect to the SQLite database.
conn = sqlite3.connect(db_path)

# Define the SQL query to build the golden dataset.
# This query performs two main functions:
# 1. Filters for a specific set of outcome types (software, creative, biological) 
#    and protections (Trade Mark).
# 2. Creates a new column 'is_software' to act as a binary label, where
#    1 indicates a software-related outcome and 0 indicates a non-software outcome.
query = """
SELECT *,
       CASE
           -- Label software and e-business platforms as 1 (True)
           WHEN outcome_type IN ('Software', 'e-Business Platform') THEN 1
           
           -- Label various creative, artistic, and biological outcomes as 0 (False)
           WHEN outcome_type IN (
               'Antibody',
               'Artistic/Creative Exhibition',
               'Artwork',
               'Biological samples',
               'Cell line',
               'Composition/Score',
               'Creative Writing',
               'Film/Video/Animation',
               'Image',
               'Performance (Music, Dance, Drama, etc)',
               'Physical Model/Kit'
           ) THEN 0
           
           -- Also label Trade Marks as 0 (False), as they are not software
           WHEN protection = 'Trade Mark' THEN 0
           
           -- All other cases within the filtered set will be NULL
           ELSE NULL
       END AS is_software
FROM outcomes
WHERE 
    -- The WHERE clause ensures we only retrieve rows that will be labeled.
    outcome_type IN (
        'Software', 'e-Business Platform',
        'Antibody', 'Artistic/Creative Exhibition', 'Artwork', 'Biological samples',
        'Cell line', 'Composition/Score', 'Creative Writing',
        'Film/Video/Animation', 'Image', 'Performance (Music, Dance, Drama, etc)',
        'Physical Model/Kit'
    )
   OR protection = 'Trade Mark'
"""

# Execute the query and load the results directly into a pandas DataFrame.
print(f"Executing query on {db_path}...")
df = pd.read_sql_query(query, conn)

# close the database connection
conn.close()

output_filename = "software_and_creative_outcomes.csv"

# Export the resulting DataFrame to a CSV file
df.to_csv(output_filename, index=False)

# Print a confirmation message to the console
print(f"Successfully exported {len(df)} records to {output_filename}")