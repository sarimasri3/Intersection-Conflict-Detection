# generate_data.py

from src.data_generation import generate_dataset

def main():
    # Generate the dataset
    dataset = generate_dataset(
        total_records=100,     # Number of records to generate
        num_vehicles=5,         # Maximum number of vehicles per scenario
        fixed_vehicle_count=False  # Set to True for a fixed number of vehicles
    )

    # Save the dataset to a CSV file
    dataset.to_csv('data/generated_dataset.csv', index=False)

    print("Dataset generated and saved to 'data/generated_dataset.csv'")

if __name__ == '__main__':
    main()
