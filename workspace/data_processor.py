import csv
import random
import os

# --- PART 1: DATA GENERATION AND WRITING ---

def generate_user_data(num_records=100):
    """Generates a list of dictionaries containing simulated user data."""
    data = []
    first_names = ["Alice", "Bob", "Charlie", "Diana", "Edward", "Fiona"]
    last_names = ["Smith", "Jones", "Brown", "Davis", "Miller", "Wilson"]
    
    print(f"-> Generating {num_records} records of user data...")
    for i in range(num_records):
        name = f"{random.choice(first_names)} {random.choice(last_names)} {i+1}"
        age = random.randint(18, 65)
        score = round(random.uniform(60.0, 100.0), 2)
        data.append({
            'user_name': name,
            'age': age,
            'score': score
        })
    return data

def write_to_csv(data, filename="user_data.csv"):
    """Writes the list of dictionaries to a CSV file."""
    keys = data[0].keys()
    print(f"-> Writing data to {filename}...")
    try:
        with open(filename, 'w', newline='') as output_file:
            dict_writer = csv.DictWriter(output_file, fieldnames=keys)
            dict_writer.writeheader()
            dict_writer.writerows(data)
        print(f"✅ Data successfully written to {filename}.")
        return filename
    except Exception as e:
        print(f"❌ Error writing CSV: {e}")
        return None

# --- PART 2: DATA READING AND ANALYSIS ---

def analyze_csv(filename="user_data.csv"):
    """Reads the CSV and calculates summary statistics."""
    if not os.path.exists(filename):
        print(f"❌ Analysis failed: File '{filename}' not found.")
        return

    print("\n--- Starting Data Analysis ---")
    total_score = 0
    record_count = 0
    
    try:
        with open(filename, 'r', newline='') as input_file:
            reader = csv.DictReader(input_file)
            for row in reader:
                try:
                    # Assuming 'score' is always present and valid
                    score = float(row['score'])
                    total_score += score
                    record_count += 1
                except ValueError:
                    print(f"Skipping row due to invalid score format: {row['user_name']}")
                except KeyError:
                    print("Error: Missing 'score' column in the CSV.")
                    return

        if record_count > 0:
            average_score = total_score / record_count
            print("\n📊 Analysis Complete:")
            print(f"Total records processed: {record_count}")
            print(f"Sum of all scores: {total_score:.2f}")
            print(f"Average score across all users: {average_score:.2f}")
        else:
            print("⚠️ No valid records found to analyze.")

    except Exception as e:
        print(f"❌ An unexpected error occurred during analysis: {e}")


# --- MAIN EXECUTION ---
if __name__ == "__main__":
    NUM_RECORDS = 50 # Using a smaller number for faster execution, but simulating a 'big' dataset process.
    
    # 1. Generate and write the data
    user_data = generate_user_data(NUM_RECORDS)
    output_file = write_to_csv(user_data)
    
    # 2. Analyze the written data
    if output_file:
        analyze_csv(output_file)

    # Cleanup (optional)
    # print(f"\nCleaning up {output_file}...")
    # os.remove(output_file)
    # print("Cleanup complete.")