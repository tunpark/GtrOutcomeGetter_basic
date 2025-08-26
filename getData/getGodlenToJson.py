import sqlite3
import pandas as pd

# This script extracts descriptions for specific software-related outcomes
# from the database and saves them in JSONL format, suitable for annotation.

# Connect to the SQLite database.
conn = sqlite3.connect("projects_sample.db")

# A query to select only the descriptions from software-related outcomes.
query = """
SELECT description AS text
FROM outcomes
WHERE outcome_type IN ('Software', 'Webtool/Application', 'e-Business Platform')
"""

# Execute the query and load the results into a pandas DataFrame.
df = pd.read_sql_query(query, conn)
conn.close()

# Export the DataFrame to a JSON Lines (.jsonl) file.
# This format is common for NLP tools like doccano.
output_filename = "software_descriptions.jsonl"
df.to_json(output_filename, orient='records', lines=True, force_ascii=False)

# Print a confirmation message.
print(f"Exported {len(df)} software-related outcome descriptions to {output_filename}")