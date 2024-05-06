import pandas as pd
import numpy as np
import json


def process_noise_data(file_path):
    # Read the CSV file into a DataFrame
    noise_data = pd.read_csv(file_path)
    
    # Convert 'Created Date' and 'Closed Date' columns to datetime format
    noise_data['Created Date'] = pd.to_datetime(noise_data['Created Date'], format="%m/%d/%Y %I:%M:%S %p")
    noise_data['Closed Date'] = pd.to_datetime(noise_data['Closed Date'], format="%m/%d/%Y %I:%M:%S %p")
    
    # Define list of construction related descriptors
    construction_noise_complains = ['Noise: Construction Before/After Hours (NM1)', 'Noise: Construction Equipment (NC1)', 'Noise: Jack Hammering (NC2)']
    
    # Filter noise_data based on construction-related descriptors
    noise_data = noise_data[noise_data['Descriptor'].isin(construction_noise_complains)]
    
    # Drop rows with missing Latitude or Longitude values
    noise_data = noise_data.dropna(subset=['Latitude', 'Longitude'])
    
    return noise_data

def process_permit_data(file_path):
    # Read the CSV file into a DataFrame
    permit_data = pd.read_csv(file_path)

    # Convert date columns to datetime format
    permit_data['Filing Date'] = pd.to_datetime(permit_data['Filing Date'], format="%m/%d/%Y")
    permit_data['Expiration Date'] = pd.to_datetime(permit_data['Expiration Date'], format="%m/%d/%Y")
    permit_data['Issuance Date'] = pd.to_datetime(permit_data['Issuance Date'], format="%m/%d/%Y")
    permit_data['Job Start Date'] = pd.to_datetime(permit_data['Job Start Date'], format="%m/%d/%Y")
    
    # Filter permit_data where Job Start Date is within 2023
    permit_data = permit_data[permit_data['Job Start Date'].dt.year == 2023]

    # Drop rows with missing Latitude or Longitude values
    permit_data = permit_data.dropna(subset=['LATITUDE', 'LONGITUDE'])

    return permit_data

def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371  
    return c * r

def process_data(noise_data, permit_data):
    result_dict = {}

    # Iterate over each row in noise_data DataFrame
    for index, noise_row in noise_data.iterrows():
        noise_lat = noise_row['Latitude']
        noise_lon = noise_row['Longitude']
        noise_key = noise_row['Unique Key']
        noise_zip = noise_row['Incident Zip']
        noise_created_date = str(noise_row['Created Date'])
        noise_complaint = noise_row['Complaint Type']
        noise_desc = noise_row['Descriptor']

        # Iterate over each row in permit_data DataFrame
        for index, permit_row in permit_data[:1].iterrows():
            permit_lat = permit_row['LATITUDE']
            permit_lon = permit_row['LONGITUDE']
            permit_job = str(permit_row['Job #'])
            permit_start_date = permit_row['Job Start Date']

            # Ensure permit start date is less than noise created date
            if permit_start_date < noise_row['Created Date']:
                # Calculate the distance between the two points
                distance = haversine(noise_lat, noise_lon, permit_lat, permit_lon)

                # If distance is smaller than 0.3km, add to dictionary
                if distance < 0.3:
                    if permit_job not in result_dict:
                        result_dict[permit_job] = []
                    result_dict[str(permit_job)].append({
                        'Unique Key': noise_key,
                        'Incident Zip': noise_zip,
                        'Complaint Type': noise_complaint,
                        'Descriptor': noise_desc,
                        'Latitude': noise_lat, 
                        'Longitude': noise_lon,
                        'Created Date': noise_created_date,
                    })

    return result_dict

def save_to_json(result_dict, output_file_path):
    # Save the dictionary to a JSON file
    with open(output_file_path, 'w') as f:
        json.dump(result_dict, f)
    # print(f"Final results saved to {output_file_path}")


def main():
    noise_file_path = '311_Service_Requests_from_2010_to_Present_20240417.csv'
    permit_file_path = 'DOB_Permit_Issuance_20240421.csv'
    output_file_path = "final_results.json"

    processed_noise_data = process_noise_data(noise_file_path)
    processed_permit_data = process_permit_data(permit_file_path)
    result_dict = process_data(processed_noise_data, processed_permit_data)
    save_to_json(result_dict, output_file_path)

if __name__ == "__main__":
    
    main()
