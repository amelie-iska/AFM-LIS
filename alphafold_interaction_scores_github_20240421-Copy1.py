# Local Interaction Score calculation
# It's designed for ColabFold-derived outputs (json and pdb files)

## Imports
import os
import json
import statistics
import numpy as np
from Bio import PDB
import pandas as pd
from multiprocessing import Pool
from pandas.errors import EmptyDataError
import pandas as pd
from pandas.errors import EmptyDataError
import os

## Functions
def calculate_pae(pdb_file_path: str, print_results: bool = True, pae_cutoff: float = 12.0, name_separator: str = "___"):
    parser = PDB.PDBParser()
    file_name = pdb_file_path.split("/")[-1]
    data_folder = pdb_file_path.split("/")[-2]

    
    ## comment since it's not following rank naming convention
    if 'rank' not in file_name:
        if print_results:
            print(f"Skipping {file_name} as it does not contain 'rank' in the file name.")
        return None

    ## comment since it's not following rank naming convention
    ##  understand which info is in parts to replicate it in a different way  X
    
    # Splitting the file name first with '_unrelaxed'
    # parts = file_name.split('_unrelaxed')
    # if len(parts) < 2:
    #     if print_results:
    #         print(f"Warning: File {file_name} does not follow expected '_unrelaxed' naming convention. Skipping this file.")
    #     return None

    # comment since it uses parts
    # understand how to extract rank in an alternative way X
    #protein_2_temp = parts[1]  # Defining protein_2_temp here to use later for rank extraction


    # comments since parts is not available
    # gigure out alternative way to generate protein_1, protein_2, pae_file_name
    # Using name_separator to separate protein_1 and protein_2 from the first part
    # if parts[0].count(name_separator) == 1:
    #     protein_1, protein_2 = parts[0].split(name_separator)
    #     pae_file_name = data_folder + '+' + protein_1 + name_separator + protein_2 + '_pae.png'
    # elif parts[0].count(name_separator) > 1:
    #     protein_1 = parts[0]
    #     protein_2 = parts[0]
    #     pae_file_name = data_folder + '+' + protein_1 + '_pae.png'
    # else:
    #     if print_results:
    #         print(f"Warning: Unexpected file naming convention for {file_name}. Skipping this file.")
    #     return None

    # Extract rank information from protein_2_temp
    # same missing files are not available
    # if "_unrelaxed_rank_00" in file_name:
    #     rank_temp = file_name.split("_unrelaxed_rank_00")[1]
    #     rank = rank_temp.split("_alphafold2")[0]
    # else:
    #     rank = "Not Available"  # or any default value you prefer
    protein_1 = "A"
    protein_2 = "B"
    rank = "pLDDT"

    if print_results:
        print("Protein 1:", protein_1)
        print("Protein 2:", protein_2)
        print("Rank:", rank)
    
    json_file = pdb_file_path.replace(".pdb", ".json").replace("unrelaxed", "scores")
    structure = parser.get_structure("example", pdb_file_path)

    for model in structure:
        for chain in model:
            chain_id = chain.get_id()
            chain_length = sum(1 for _ in chain.get_residues())
            if chain_id == 'A':
                protein_a_len = chain_length
            # print(f"Chain {chain_id} length : {chain_length}")


    # Load the JSON file
    with open(json_file, 'r') as file:
        json_data = json.load(file)
    
    plddt = statistics.mean(json_data["plddt"])
    ptm = json_data["ptm"]
    iptm = json_data["iptm"]
    pae = np.array(json_data['pae'])

    # Calculate thresholded_pae
    thresholded_pae = np.where(pae < pae_cutoff, 1, 0)

    # Calculate the interaction amino acid numbers
    local_interaction_protein_a = np.count_nonzero(thresholded_pae[:protein_a_len, :protein_a_len])
    local_interaction_protein_b = np.count_nonzero(thresholded_pae[protein_a_len:, protein_a_len:])
    local_interaction_interface_1 = np.count_nonzero(thresholded_pae[:protein_a_len, protein_a_len:])
    local_interaction_interface_2 = np.count_nonzero(thresholded_pae[protein_a_len:, :protein_a_len])
    local_interaction_interface_avg = (
        local_interaction_interface_1 + local_interaction_interface_2
    )

    
    # Calculate average thresholded_pae for each region
    average_thresholded_protein_a = thresholded_pae[:protein_a_len,:protein_a_len].mean() * 100
    average_thresholded_protein_b = thresholded_pae[protein_a_len:,protein_a_len:].mean() * 100
    average_thresholded_interaction1 = thresholded_pae[:protein_a_len,protein_a_len:].mean() * 100
    average_thresholded_interaction2 = thresholded_pae[protein_a_len:,:protein_a_len].mean() * 100
    average_thresholded_interaction_total = (average_thresholded_interaction1 + average_thresholded_interaction2) / 2
    

    pae_protein_a = np.mean( pae[:protein_a_len,:protein_a_len] )
    pae_protein_b = np.mean( pae[protein_a_len:,protein_a_len:] )
    pae_interaction1 = np.mean(pae[:protein_a_len,protein_a_len:])
    pae_interaction2 = np.mean(pae[protein_a_len:,:protein_a_len])
    pae_interaction_total = ( pae_interaction1 + pae_interaction2 ) / 2

    # For pae_A
    selected_values_protein_a = pae[:protein_a_len, :protein_a_len][thresholded_pae[:protein_a_len, :protein_a_len] == 1]
    average_selected_protein_a = np.mean(selected_values_protein_a)

    # For pae_B
    selected_values_protein_b = pae[protein_a_len:, protein_a_len:][thresholded_pae[protein_a_len:, protein_a_len:] == 1]
    average_selected_protein_b = np.mean(selected_values_protein_b)

    # For pae_interaction1
    selected_values_interaction1 = pae[:protein_a_len, protein_a_len:][thresholded_pae[:protein_a_len, protein_a_len:] == 1]
    average_selected_interaction1 = np.mean(selected_values_interaction1) if selected_values_interaction1.size > 0 else pae_cutoff

    # For pae_interaction2
    selected_values_interaction2 = pae[protein_a_len:, :protein_a_len][thresholded_pae[protein_a_len:, :protein_a_len] == 1]
    average_selected_interaction2 = np.mean(selected_values_interaction2) if selected_values_interaction2.size > 0 else pae_cutoff

    # For pae_interaction_total
    average_selected_interaction_total = (average_selected_interaction1 + average_selected_interaction2) / 2

    if print_results:
        # Print the total results
        print("Total pae_A : {:.2f}".format(pae_protein_a))
        print("Total pae_B : {:.2f}".format(pae_protein_b))
        print("Total pae_i_1 : {:.2f}".format(pae_interaction1))
        print("Total pae_i_2 : {:.2f}".format(pae_interaction2))
        print("Total pae_i_avg : {:.2f}".format(pae_interaction_total))

        # Print the local results
        print("Local pae_A : {:.2f}".format(average_selected_protein_a))
        print("Local pae_B : {:.2f}".format(average_selected_protein_b))
        print("Local pae_i_1 : {:.2f}".format(average_selected_interaction1))
        print("Local pae_i_2 : {:.2f}".format(average_selected_interaction2))
        print("Local pae_i_avg : {:.2f}".format(average_selected_interaction_total))

        # Print the >PAE-cutoff area
        print("Local interaction area (Protein A):", local_interaction_protein_a)
        print("Local interaction area (Protein B):", local_interaction_protein_b)
        print("Local interaction area (Interaction 1):", local_interaction_interface_1)
        print("Local interaction area (Interaction 2):", local_interaction_interface_2)
        print("Total Interaction area (Interface):", local_interaction_interface_avg)


    # Transform the pae matrix
    scaled_pae = reverse_and_scale_matrix(pae, pae_cutoff)

    # For local interaction score for protein_a
    selected_values_protein_a = scaled_pae[:protein_a_len, :protein_a_len][thresholded_pae[:protein_a_len, :protein_a_len] == 1]
    average_selected_protein_a_score = np.mean(selected_values_protein_a)

    # For local interaction score for protein_b
    selected_values_protein_b = scaled_pae[protein_a_len:, protein_a_len:][thresholded_pae[protein_a_len:, protein_a_len:] == 1]
    average_selected_protein_b_score = np.mean(selected_values_protein_b)

    # For local interaction score1
    selected_values_interaction1_score = scaled_pae[:protein_a_len, protein_a_len:][thresholded_pae[:protein_a_len, protein_a_len:] == 1]
    average_selected_interaction1_score = np.mean(selected_values_interaction1_score) if selected_values_interaction1_score.size > 0 else 0

    # For local interaction score2
    selected_values_interaction2_score = scaled_pae[protein_a_len:, :protein_a_len][thresholded_pae[protein_a_len:, :protein_a_len] == 1]
    average_selected_interaction2_score = np.mean(selected_values_interaction2_score) if selected_values_interaction2_score.size > 0 else 0

    # For average local interaction score
    average_selected_interaction_total_score = (average_selected_interaction1_score + average_selected_interaction2_score) / 2
    
    if print_results:
        # Print the local interaction scores
        print("Local Interaction Score_A : {:.3f}".format(average_selected_protein_a_score))
        print("Local Interaction Score_B : {:.3f}".format(average_selected_protein_b_score))
        print("Local Interaction Score_i_1 : {:.3f}".format(average_selected_interaction1_score))
        print("Local Interaction Score_i_2 : {:.3f}".format(average_selected_interaction2_score))
        print("Local Interaction Score_i_avg : {:.3f}".format(average_selected_interaction_total_score))

    COLUMNS_ORDER = [
        'Protein_1', 'Protein_2', 'pLDDT', 'pTM', 'ipTM',
        'Local_Score_A', 'Local_Score_B', 'Local_Score_i_1', 'Local_Score_i_2', 'Local_Score_i_avg',
        'Local_Area_A', 'Local_Area_B', 'Local_Area_i_1', 'Local_Area_i_2', 'Local_Area_i_avg', 
        'Total_pae_A', 'Total_pae_B', 'Total_pae_i_1', 'Total_pae_i_2', 'Total_pae_i_avg',
        'Local_pae_A', 'Local_pae_B', 'Local_pae_i_1', 'Local_pae_i_2', 'Local_pae_i_avg',
        'Rank', 'saved folder', 'pdb', 'pae_file_name'
    ]

    if parts[0].count(name_separator) == 1:
        pae_file_name =  data_folder + '+' + protein_1 + name_separator + protein_2 + '_pae.png'
    elif parts[0].count(name_separator) > 1:
        pae_file_name = data_folder + '+' + protein_1 + '_pae.png'

    data = {
        'Protein_1': protein_1,
        'Protein_2': protein_2,
        'pLDDT': round(plddt, 2),
        'pTM': ptm,
        'ipTM': iptm,
        'Total_pae_A': round(pae_protein_a, 2),
        'Total_pae_B': round(pae_protein_b, 2),
        'Total_pae_i_1': round(pae_interaction1, 2),
        'Total_pae_i_2': round(pae_interaction2, 2),
        'Total_pae_i_avg': round(pae_interaction_total, 2),
        'Local_pae_A': round(average_selected_protein_a, 2),
        'Local_pae_B': round(average_selected_protein_b, 2),
        'Local_pae_i_1': round(average_selected_interaction1, 2),
        'Local_pae_i_2': round(average_selected_interaction2, 2),
        'Local_pae_i_avg': round(average_selected_interaction_total, 2),
        'Local_Score_A': round(average_selected_protein_a_score, 3),
        'Local_Score_B': round(average_selected_protein_b_score, 3),
        'Local_Score_i_1': round(average_selected_interaction1_score, 3),
        'Local_Score_i_2': round(average_selected_interaction2_score, 3),
        'Local_Score_i_avg': round(average_selected_interaction_total_score, 3),
        'Local_Area_A': local_interaction_protein_a,
        'Local_Area_B': local_interaction_protein_b,
        'Local_Area_i_1': local_interaction_interface_1,
        'Local_Area_i_2': local_interaction_interface_2,
        'Local_Area_i_avg': local_interaction_interface_avg,
        'Rank': rank,
        'saved folder': os.path.dirname(pdb_file_path),  # Gets the parent directory of the file path
        'pdb': os.path.basename(pdb_file_path),  # Extracts just the base name of the pdb file
        'pae_file_name': pae_file_name
    }

    return pd.DataFrame(data, index=[file_name])[COLUMNS_ORDER]


def reverse_and_scale_matrix(matrix: np.ndarray, pae_cutoff: float = 12.0) -> np.ndarray:
    """
    Scale the values in the matrix such that:
    0 becomes 1, pae_cutoff becomes 0, and values greater than pae_cutoff are also 0.
    
    Args:
    - matrix (np.ndarray): Input numpy matrix.
    - pae_cutoff (float): Threshold above which values become 0.
    
    Returns:
    - np.ndarray: Transformed matrix.
    """
    
    # Scale the values to [0, 1] for values between 0 and cutoff
    scaled_matrix = (pae_cutoff - matrix) / pae_cutoff
    scaled_matrix = np.clip(scaled_matrix, 0, 1)  # Ensures values are between 0 and 1
    
    return scaled_matrix



def process_pdb_files(directory_path: str, processed_files=[], pae_cutoff: float = 12.0, name_separator: str = "___") -> pd.DataFrame:
    all_files = os.listdir(directory_path)
    pdb_files = [f for f in all_files if f.endswith(".pdb") and f not in processed_files]

    # Start with an empty DataFrame with columns explicitly defined if necessary
    df_results = pd.DataFrame()  
    results_list = []  # Use a list to collect data frames or series

    for pdb_file in pdb_files:
        pdb_file_path = os.path.join(directory_path, pdb_file)
        try:
            result = calculate_pae(pdb_file_path, print_results=False, pae_cutoff=pae_cutoff, name_separator=name_separator)
            if result is not None:
                results_list.append(result)
        except FileNotFoundError:
            print(f"Error: File {pdb_file_path} not found. Skipping...")

    if results_list:
        df_results = pd.concat(results_list, axis=0, ignore_index=True)
    return df_results


## Main
def main(base_path, saving_base_path, cutoff, folders_to_analyze, num_processes, name_separator: str = "___"):
    if not os.path.exists(saving_base_path):
        os.makedirs(saving_base_path)

    for data_folder in folders_to_analyze:
        directory_path = os.path.join(base_path, data_folder)
        if os.path.exists(directory_path):
            output_filename = f"{data_folder}_alphafold_analysis.csv"
            full_saving_path = os.path.join(saving_base_path, output_filename)
            print(f"Processing data from {directory_path}")
            print(f"Saving to {full_saving_path}")

            # Check for existing processed files
            if os.path.exists(full_saving_path):
                try:
                    existing_df = pd.read_csv(full_saving_path)
                    processed_files = existing_df['pdb'].tolist() if 'pdb' in existing_df.columns else []
                except EmptyDataError:
                    # Handle the empty file situation
                    print(f"File {full_saving_path} is empty. Starting from scratch.")
                    existing_df = pd.DataFrame()
                    processed_files = []
            else:
                existing_df = pd.DataFrame()
                processed_files = []

            new_data = process_pdb_files(directory_path, processed_files, cutoff, name_separator)

            # Combine old and new data only if there's new data
            if not new_data.empty:
                combined_df = pd.concat([existing_df, new_data])
                
                # Save the combined DataFrame
                combined_df.to_csv(full_saving_path, index=False)
                print(f"Saved processed data to {full_saving_path}")
            else:
                print(f"No new data to append. CSV remains unchanged.")

        else:
            print(f"Directory {directory_path} does not exist! Skipping...")


## Process Files
def process_pdb_file(pdb_file, directory_path, processed_files, cutoff, name_separator):
    pdb_file_path = os.path.join(directory_path, pdb_file)
    print("\nProcessing:", pdb_file)
    try:
        results = calculate_pae(pdb_file_path, False, cutoff, name_separator)
        return results
    except FileNotFoundError:
        print(f"Error: File {pdb_file_path} not found. Skipping...")
        return None

def process_pdb_files_parallel(directory_path: str, processed_files=[], pae_cutoff: float = 12.0, num_processes=1) -> pd.DataFrame:
    all_files = os.listdir(directory_path)
    pdb_files = [f for f in all_files if f.endswith(".pdb") and f not in processed_files]

    df_results = pd.DataFrame()
    
    with Pool(num_processes) as pool:
        results = pool.starmap(calculate_pae, [(os.path.join(directory_path, f), False, pae_cutoff) for f in pdb_files])
        for res in results:
            if res is not None:
                df_results = df_results.append(res, ignore_index=True)
    
    return df_results

def get_subdirectories(base_path):
    """
    Get a list of subdirectories in the given base path.

    Parameters:
    - base_path: The path where to look for subdirectories.

    Returns:
    - List of subdirectories as strings.
    """
    return [d.name for d in os.scandir(base_path) if d.is_dir()]


def get_num_cpu_cores():
    try:
        return os.cpu_count() or 1
    except AttributeError:
        return multiprocessing.cpu_count() or 1

num_processes = get_num_cpu_cores()
print(f"Number of available CPU cores: {num_processes}")

## Post Processing
def process_dataframe(df):
    """
    Process the given DataFrame to extract rank1 rows and compute average values.
    Returns a new DataFrame with both rank1 and average information.
    """

    # Drop rows with any NaN values in the relevant columns
    df = df.dropna(subset=['Local_Score_i_avg', 'Local_Area_i_avg', 'ipTM', 'pTM', 'pLDDT'])
    
    # Extract rank 1 rows and rename columns accordingly
    rank1_rows = df[df['Rank'] == 1][['Protein_1', 'Protein_2', 'Local_Score_i_avg', 'Local_Area_i_avg', 'ipTM', 'pTM', 'pLDDT', 'pae_file_name']].copy()
    rank1_rows.rename(columns={
        'Local_Score_i_avg': 'Best LIS',
        'Local_Area_i_avg': 'Best LIA',
        'ipTM': 'Best ipTM',
        'pTM': 'Best pTM',
        'pLDDT': 'Best pLDDT'
    }, inplace=True)

    # Make sure to also keep 'pae_file_name' when you calculate the averages
    average_values = df.groupby('pae_file_name', as_index=False)[['Local_Score_i_avg', 'Local_Area_i_avg', 'ipTM', 'pTM', 'pLDDT']].mean()
    average_values.rename(columns={
        'Local_Score_i_avg': 'Average LIS',
        'Local_Area_i_avg': 'Average LIA',
        'ipTM': 'Average ipTM',
        'pTM': 'Average pTM',
        'pLDDT': 'Average pLDDT'
    }, inplace=True)

    
    # Merge rank 1 rows with the average values using 'pae_file_name' as the key
    final_df = pd.merge(rank1_rows, average_values, on='pae_file_name', how='left')

    # Define the columns of interest and their order
    columns_of_interest = [
        'Protein_1', 'Protein_2', 
        'Best LIS', 'Average LIS',
        'Best LIA', 'Average LIA',
        'Best ipTM', 'Average ipTM',
        'Best pTM', 'Average pTM',
        'Best pLDDT', 'Average pLDDT',
        'pae_file_name'
    ]

    final_df = final_df[columns_of_interest]

    return final_df


#  Run Step 1
# Define your base paths, cutoff value, and folder list
cutoff = 20000
name_separator = "___"  # add separator that distinguishes protein_1 and protein_2

base_path = "/home/lily/amelie/Workspace/ColabFold/work/final_outputs/output-5TPN-all"#"AFM output folder"
saving_base_path = "/home/lily/amelie/Workspace/AFM-LIS/output/analysis_pae_" + str(cutoff)  

# Generate folders_to_analyze list
folders_to_analyze = get_subdirectories(base_path)
# folders_to_analyze = ["test_jupyter"]

# Call the main function with the folder list
main(base_path, saving_base_path, cutoff, folders_to_analyze, num_processes, name_separator)


#  Run post processing
# Path to the specific folder where the original files are located
folder_path = saving_base_path  

# Path to the folder where you want to save the processed files
saving_path = saving_base_path   + "/averaged"

# Ensure the saving path directory exists, if not, create it
if not os.path.exists(saving_path):
    os.makedirs(saving_path)

file_names = [f for f in os.listdir(folder_path) if f.endswith("alphafold_analysis.csv")]

for file_name in file_names:
    file_path = os.path.join(folder_path, file_name)
    print(file_name)

    try:
        df = pd.read_csv(file_path)

        # Check if DataFrame is empty
        if df.empty:
            print(f"File {file_name} is empty. Skipping...")
            continue

        # Process the DataFrame
        processed_df = process_dataframe(df)

        # Constructing the new file name
        base_name = os.path.splitext(file_name)[0]
        new_file_name = f"{base_name}_processed.xlsx"
        new_file_path = os.path.join(saving_path, new_file_name)

        # Save the processed DataFrame to a new file
        processed_df.to_excel(new_file_path, index=False)
        print(f"Processed {file_name} and saved to {new_file_name}")

    except EmptyDataError:
        print(f"File {file_path} is empty and was skipped.")
    except Exception as e:  # General exception to catch unexpected errors
        print(f"Error processing {file_name}: {str(e)}")

print("Processing completed.")



