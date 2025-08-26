import pandas as pd
keywords = list(set([
    # Code hosting and version control platforms
    "github", "gitlab", "bitbucket", "sourceforge", "gitee", "codeberg",
    "source.cloud.google", "azure.microsoft.com/en-us/products/devops/repos",
    "dev.azure", "visualstudio", "aws.amazon.com/codesuite", "code.google", 
    "launchpad", "savannah", "sr.ht",

    # Package managers and repositories
    "npmjs", "pypi", "repo1.maven", "nuget", "rubygems", "packagist",
    "crates.io", "hub.docker", "cran",

    # Data and model sharing platforms
    "huggingface", "zenodo", "figshare"
]))

# Read the input CSV file
try:
    df = pd.read_csv("outcomes_selected.csv")
except FileNotFoundError:
    print("Error: The input file 'outcomes_selected.csv' was not found.")
    exit()

#Define the weak labeling function.
def weak_label(url):
    """
    Applies a weak label based on the presence of software-related keywords in the URL.
    - Returns 1 if a keyword is found.
    - Returns 0 if no keyword is found.
    - Returns None for invalid or empty URLs.
    """
    if not isinstance(url, str) or url.strip() == "":
        return None  
    
    # Convert URL to lowercase for case-insensitive matching
    url_lower = url.lower()
    
    # Check if any keyword exists in the URL
    return 1 if any(keyword in url_lower for keyword in keywords) else 0


df["label"] = df["support_url"].apply(weak_label)
df = df.dropna(subset=["label"])

df['label'] = df['label'].astype(int)

#Add a 'category_name' column.
df["category_name"] = "Software"

df_weak = df[["text", "category_name", "label"]]

# Save the processed DataFrame to a new CSV file.
output_filename = "weakly_labeled_outcomes.csv"
df_weak.to_csv(output_filename, index=False, encoding="utf-8")

print(f"Weak labeling complete. Generated {len(df_weak)} samples, saved to {output_filename}")




