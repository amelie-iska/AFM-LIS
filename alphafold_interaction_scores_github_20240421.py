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
from datetime import datetime


## Functions
def calculate_pae(pdb_file_path: str, print_results: bool = True, pae_cutoff: float = 12.0, name_separator: str = "___"):
    print(f"Debug: Starting calculate_pae function with file: {pdb_file_path}")
    parser = PDB.PDBParser()
    file_name = pdb_file_path.split("/")[-1]
    data_folder = pdb_file_path.split("/")[-2]

    print(f"Debug: File name: {file_name}, Data folder: {data_folder}")

    if 'rank' not in file_name:
        print(f"Debug: Skipping {file_name} as it does not contain 'rank' in the file name.")
        return None

    # Splitting the file name first with '_unrelaxed'
    parts = file_name.split('_unrelaxed')
    if len(parts) < 2:
        print(f"Debug: Warning: File {file_name} does not follow expected '_unrelaxed' naming convention. Skipping this file.")
        return None

    protein_2_temp = parts[1]
    print(f"Debug: protein_2_temp: {protein_2_temp}")

    # Using name_separator to separate protein_1 and protein_2 from the first part
    if parts[0].count(name_separator) == 1:
        protein_1, protein_2 = parts[0].split(name_separator)
        pae_file_name = data_folder + '+' + protein_1 + name_separator + protein_2 + '_pae.png'
    elif parts[0].count(name_separator) > 1:
        protein_1 = parts[0]
        protein_2 = parts[0]
        pae_file_name = data_folder + '+' + protein_1 + '_pae.png'
    else:
        print(f"Debug: Warning: Unexpected file naming convention for {file_name}. Skipping this file.")
        return None

    print(f"Debug: protein_1: {protein_1}, protein_2: {protein_2}, pae_file_name: {pae_file_name}")

    # Extract rank information from protein_2_temp
    if "_unrelaxed_rank_00" in file_name:
        rank_temp = file_name.split("_unrelaxed_rank_00")[1]
        rank = rank_temp.split("_alphafold2")[0]
    else:
        rank = "Not Available"

    print(f"Debug: Rank: {rank}")

    # protein_1 = "A"
    # protein_2 = "B"
    # rank = "pLDDT"

    if print_results:
        print("Protein 1:", protein_1)
        print("Protein 2:", protein_2)
        print("Rank:", rank)

    #############
    #############
    
    json_file = pdb_file_path.replace(".pdb", ".json").replace("unrelaxed", "scores")
    print(f"Debug: JSON file path: {json_file}")

    try:
        structure = parser.get_structure("example", pdb_file_path)
        print("Debug: PDB structure parsed successfully")
    except Exception as e:
        print(f"Debug: Error parsing PDB file: {e}")
        return None

    protein_a_len = 0
    for model in structure:
        for chain in model:
            chain_id = chain.get_id()
            chain_length = sum(1 for _ in chain.get_residues())
            print(f"Debug: Chain {chain_id} length : {chain_length}")
            if chain_id == 'A':
                protein_a_len = chain_length

    print(f"Debug: protein_a_len: {protein_a_len}")

    # Load the JSON file
    try:
        with open(json_file, 'r') as file:
            json_data = json.load(file)
        print("Debug: JSON file loaded successfully")
    except Exception as e:
        print(f"Debug: Error loading JSON file: {e}")
        return None

    plddt = statistics.mean(json_data["plddt"])
    ptm = json_data["ptm"]
    iptm = json_data["iptm"]
    pae = np.array(json_data['pae'])

    print(f"Debug: plddt: {plddt}, ptm: {ptm}, iptm: {iptm}, pae shape: {pae.shape}")

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

    # if parts[0].count(name_separator) == 1:
    #     pae_file_name =  data_folder + '+' + protein_1 + name_separator + protein_2 + '_pae.png'
    # elif parts[0].count(name_separator) > 1:
    #     pae_file_name = data_folder + '+' + protein_1 + '_pae.png'
    pae_file_name =  data_folder + '+' + protein_1 + name_separator + protein_2 + '_pae.png'

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
    print("Debug: Finished processing, returning DataFrame")
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
    print(f"Debug: Starting reverse_and_scale_matrix function")
    print(f"Debug: Input matrix shape: {matrix.shape}")
    print(f"Debug: PAE cutoff: {pae_cutoff}")
    
    # Print some statistics about the input matrix
    print(f"Debug: Input matrix min: {np.min(matrix)}, max: {np.max(matrix)}, mean: {np.mean(matrix)}")
    
    # Scale the values to [0, 1] for values between 0 and cutoff
    scaled_matrix = (pae_cutoff - matrix) / pae_cutoff
    
    print(f"Debug: Scaled matrix (before clipping) min: {np.min(scaled_matrix)}, max: {np.max(scaled_matrix)}, mean: {np.mean(scaled_matrix)}")
    
    scaled_matrix = np.clip(scaled_matrix, 0, 1)  # Ensures values are between 0 and 1
    
    print(f"Debug: Final scaled matrix min: {np.min(scaled_matrix)}, max: {np.max(scaled_matrix)}, mean: {np.mean(scaled_matrix)}")
    
    return scaled_matrix



def process_pdb_files(directory_path: str, processed_files=[], pae_cutoff: float = 12.0, name_separator: str = "___") -> pd.DataFrame:
    print(f"Debug: Starting process_pdb_files function")
    print(f"Debug: Directory path: {directory_path}")
    print(f"Debug: Number of already processed files: {len(processed_files)}")
    print(f"Debug: PAE cutoff: {pae_cutoff}")
    print(f"Debug: Name separator: {name_separator}")

    all_files = os.listdir(directory_path)
    print(f"Debug: Total files in directory: {len(all_files)}")

    pdb_files = [f for f in all_files if f.endswith(".pdb") and f not in processed_files]
    print(f"Debug: Number of PDB files to process: {len(pdb_files)}")

    # Start with an empty DataFrame with columns explicitly defined if necessary
    df_results = pd.DataFrame()  
    results_list = []  # Use a list to collect data frames or series

    for i, pdb_file in enumerate(pdb_files, 1):
        pdb_file_path = os.path.join(directory_path, pdb_file)
        print(f"Debug: Processing file {i}/{len(pdb_files)}: {pdb_file}")
        try:
            result = calculate_pae(pdb_file_path, print_results=False, pae_cutoff=pae_cutoff, name_separator=name_separator)
            if result is not None:
                print(f"Debug: Successfully processed {pdb_file}")
                results_list.append(result)
            else:
                print(f"Debug: calculate_pae returned None for {pdb_file}")
        except FileNotFoundError:
            print(f"Error: File {pdb_file_path} not found. Skipping...")
        except Exception as e:
            print(f"Debug: Unexpected error processing {pdb_file}: {str(e)}")

    print(f"Debug: Total successfully processed files: {len(results_list)}")

    if results_list:
        df_results = pd.concat(results_list, axis=0, ignore_index=True)
        print(f"Debug: Final DataFrame shape: {df_results.shape}")
        print(f"Debug: DataFrame columns: {df_results.columns.tolist()}")
    else:
        print("Debug: No results to concatenate. DataFrame is empty.")

    return df_results

## Main
def main(base_path, saving_base_path, cutoff, folders_to_analyze, num_processes, name_separator: str = "___"):
    print(f"Debug: Starting main function")
    print(f"Debug: Base path: {base_path}")
    print(f"Debug: Saving base path: {saving_base_path}")
    print(f"Debug: Cutoff: {cutoff}")
    print(f"Debug: Folders to analyze: {folders_to_analyze}")
    print(f"Debug: Number of processes: {num_processes}")
    print(f"Debug: Name separator: {name_separator}")

    if not os.path.exists(saving_base_path):
        os.makedirs(saving_base_path)
        print(f"Debug: Created saving base path: {saving_base_path}")

    for data_folder in folders_to_analyze:
        directory_path = os.path.join(base_path, data_folder)
        print(f"Debug: Processing folder: {data_folder}")
        print(f"Debug: Full directory path: {directory_path}")

        if os.path.exists(directory_path):
            output_filename = f"{data_folder}_alphafold_analysis.csv"
            full_saving_path = os.path.join(saving_base_path, output_filename)
            print(f"Debug: Output file: {output_filename}")
            print(f"Debug: Full saving path: {full_saving_path}")

            # Check for existing processed files
            if os.path.exists(full_saving_path):
                print(f"Debug: Existing file found: {full_saving_path}")
                try:
                    existing_df = pd.read_csv(full_saving_path)
                    print(f"Debug: Existing DataFrame shape: {existing_df.shape}")
                    processed_files = existing_df['pdb'].tolist() if 'pdb' in existing_df.columns else []
                    print(f"Debug: Number of previously processed files: {len(processed_files)}")
                except EmptyDataError:
                    print(f"Debug: File {full_saving_path} is empty. Starting from scratch.")
                    existing_df = pd.DataFrame()
                    processed_files = []
            else:
                print(f"Debug: No existing file found. Starting from scratch.")
                existing_df = pd.DataFrame()
                processed_files = []

            print(f"Debug: Calling process_pdb_files")
            new_data = process_pdb_files(directory_path, processed_files, cutoff, name_separator)
            print(f"Debug: New data shape: {new_data.shape if not new_data.empty else 'Empty DataFrame'}")

            # Combine old and new data only if there's new data
            if not new_data.empty:
                combined_df = pd.concat([existing_df, new_data])
                print(f"Debug: Combined DataFrame shape: {combined_df.shape}")
                
                # Save the combined DataFrame
                combined_df.to_csv(full_saving_path, index=False)
                print(f"Debug: Saved processed data to {full_saving_path}")
            else:
                print(f"Debug: No new data to append. CSV remains unchanged.")
        else:
            print(f"Debug: Directory {directory_path} does not exist! Skipping...")

    print(f"Debug: Main function completed")

## Process Files
import os
from multiprocessing import Pool
import pandas as pd

def process_pdb_file(pdb_file, directory_path, processed_files, cutoff, name_separator):
    pdb_file_path = os.path.join(directory_path, pdb_file)
    print(f"\nDebug: Processing: {pdb_file}")
    print(f"Debug: Full file path: {pdb_file_path}")
    try:
        results = calculate_pae(pdb_file_path, False, cutoff, name_separator)
        if results is not None:
            print(f"Debug: Successfully processed {pdb_file}")
        else:
            print(f"Debug: calculate_pae returned None for {pdb_file}")
        return results
    except FileNotFoundError:
        print(f"Debug: Error: File {pdb_file_path} not found. Skipping...")
        return None
    except Exception as e:
        print(f"Debug: Unexpected error processing {pdb_file}: {str(e)}")
        return None

def process_pdb_files_parallel(directory_path: str, processed_files=[], pae_cutoff: float = 12.0, name_separator: str = "___", num_processes=1) -> pd.DataFrame:
    print(f"Debug: Starting process_pdb_files_parallel")
    print(f"Debug: Directory path: {directory_path}")
    print(f"Debug: Number of already processed files: {len(processed_files)}")
    print(f"Debug: PAE cutoff: {pae_cutoff}")
    print(f"Debug: Name separator: {name_separator}")
    print(f"Debug: Number of processes: {num_processes}")

    all_files = os.listdir(directory_path)
    print(f"Debug: Total files in directory: {len(all_files)}")

    pdb_files = [f for f in all_files if f.endswith(".pdb") and f not in processed_files]
    print(f"Debug: Number of PDB files to process: {len(pdb_files)}")

    df_results = pd.DataFrame()
    
    with Pool(num_processes) as pool:
        print(f"Debug: Created process pool with {num_processes} processes")
        results = pool.starmap(process_pdb_file, [(f, directory_path, processed_files, pae_cutoff, name_separator) for f in pdb_files])
        print(f"Debug: Parallel processing completed")
        print(f"Debug: Number of results: {len(results)}")
        
        valid_results = [res for res in results if res is not None]
        print(f"Debug: Number of valid results: {len(valid_results)}")
        
        if valid_results:
            df_results = pd.concat(valid_results, axis=0, ignore_index=True)
            print(f"Debug: Final DataFrame shape: {df_results.shape}")
            print(f"Debug: DataFrame columns: {df_results.columns.tolist()}")
        else:
            print("Debug: No valid results to concatenate. DataFrame is empty.")
    
    return df_results
    

def get_subdirectories(base_path):
    """
    Get a list of subdirectories in the given base path.
    Parameters:
    - base_path: The path where to look for subdirectories.
    Returns:
    - List of subdirectories as strings.
    """
    print(f"Debug: Starting get_subdirectories function")
    print(f"Debug: Base path: {base_path}")

    if not os.path.exists(base_path):
        print(f"Debug: Base path does not exist: {base_path}")
        return []

    if not os.path.isdir(base_path):
        print(f"Debug: Base path is not a directory: {base_path}")
        return []

    subdirectories = [d.name for d in os.scandir(base_path) if d.is_dir()]
    
    print(f"Debug: Number of subdirectories found: {len(subdirectories)}")
    for i, subdir in enumerate(subdirectories, 1):
        print(f"Debug: Subdirectory {i}: {subdir}")

    return subdirectories

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
saving_base_path = "/home/lily/amelie/Workspace/AFM-LIS/outputs/analysis_pae_" + str(cutoff)  

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



