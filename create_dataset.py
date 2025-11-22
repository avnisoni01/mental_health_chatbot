import os

# Step 1: Make sure data folder exists
os.makedirs("data", exist_ok=True)

# Step 2: File path
file_path = "data/mental_health.csv"

# Step 3: Dataset content
dataset = """text,label
I feel sad and unmotivated lately,depression
I am constantly worried about everything,anxiety
I feel good today,happy
I am stressed because of work,stress
Nothing feels enjoyable anymore,depression
I feel calm and relaxed,happy
I cannot sleep properly at night,anxiety
I am having frequent panic attacks,anxiety
I feel positive about my life,happy
I feel overwhelmed with everything,stress
"""

# Step 4: Write file
with open(file_path, "w", encoding="utf-8") as f:
    f.write(dataset)

print("Dataset created at:", file_path)
