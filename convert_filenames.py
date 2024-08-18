# import os
# import csv

# def generate_new_filename(old_filename, index):
#     # Extract the model number from the original filename
#     model_number = old_filename.split('model_')[1].split('_')[0]
    
#     # Determine the file extension
#     ext = '.pdb' if old_filename.endswith('.pdb') else '.json'
    
#     # Generate the new filename
#     new_filename = f"target___candidate_{index}_rank_{model_number:03d}{ext}"
    
#     return new_filename

# def create_filename_mapping(directory):
#     old_filenames = [f for f in os.listdir(directory) if f.endswith('.pdb') or f.endswith('.json')]
#     print(old_filenames)
    
#     mapping = []
    
#     for index, old_filename in enumerate(sorted(old_filenames), start=1):
#         new_filename = generate_new_filename(old_filename, index)
#         mapping.append((old_filename, new_filename))
    
#     # Write the mapping to a CSV file
#     with open('filename_mapping.csv', 'w', newline='') as csvfile:
#         writer = csv.writer(csvfile)
#         writer.writerow(['original_name', 'new_name'])  # Header
#         writer.writerows(mapping)
    
#     print("Filename mapping created in 'filename_mapping.csv'")

# # Use the script
# create_filename_mapping('/home/lily/amelie/Workspace/ColabFold/work/final_outputs/output-5TPN-all')
import os
import json
import statistics
import shutil

def parse_pdb_for_scores(pdb_file):
    b_factors = []
    with open(pdb_file, 'r') as f:
        for line in f:
            if line.startswith('ATOM'):
                b_factor = float(line[60:66])
                b_factors.append(b_factor)
    
    avg_plddt = statistics.mean(b_factors) if b_factors else 0
    return avg_plddt

def parse_json_for_scores(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    plddt = data.get('plddt', [])
    return statistics.mean(plddt) if plddt else 0

def extract_model_number(filename):
    if 'v3_model_' in filename:
        parts = filename.split('v3_model_')
        if len(parts) > 1:
            model_num = parts[1].split('_')[0]
            print(f"Found model number: {model_num}")
            return model_num
    print("No model number found, returning 'unknown'")
    return 'unknown'

def analyze_scores(directory):
    scores = {}
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        model_number = extract_model_number(filename)
        if model_number != "unknown":
        
            try:
                if filename.endswith('.pdb'):
                    score = parse_pdb_for_scores(filepath)
                    scores[model_number] = score
                    print(f"Processed PDB: (Model: {model_number}, Score: {score:.2f})")
                elif filename.endswith('.json'):
                    score = parse_json_for_scores(filepath)
                    scores[model_number] = score
                    print(f"Processed JSON: (Model: {model_number}, Score: {score:.2f})")
                else:
                    print(f"Skipping file: {filename} (not a PDB or JSON file)")
            except Exception as e:
                print(f"Error processing file {filename}: {str(e)}")
    
    print("\nScores before sorting:")
    for model, score in scores.items():
        print(f"Model: {model}, Score: {score}")
    
    ranked_models = sorted(scores.items(), key=lambda x: float(x[1]), reverse=True)
    
    print("\nModels ranked by average pLDDT score:")
    for rank, (model, score) in enumerate(ranked_models, 1):
        print(f"Rank {rank}: Model {model} (Score: {score:.2f})")
    
    return ranked_models

def duplicate_and_rename_files(directory, ranked_models):
    renamed_dir = os.path.join(directory, "renamed")
    os.makedirs(renamed_dir, exist_ok=True)
    
    print("\nDuplicating and renaming files:")
    for rank, (model, score) in enumerate(ranked_models, 1):
        old_pdb_name = f"run_1_0__T_0.075__seed_111__num_res_44__num_ligand_res_0__use_ligand_context_True__ligand_cutoff_distance_8.0__batch_size_1__number_of_batches_5__model_path_._model_params_ligandmpnn_v_32_010_25.pt_unrelaxed_alphafold2_multimer_v3_model_{model}_seed_000.pdb"
        new_pdb_name = f"target___candidate_1_rank_{rank:03d}.pdb"
        old_pdb_path = os.path.join(directory, old_pdb_name)
        new_pdb_path = os.path.join(renamed_dir, new_pdb_name)
        
        old_json_name = old_pdb_name.replace('unrelaxed_', '').replace('.pdb', '.json')
        new_json_name = f"target___candidate_1_rank_{rank:03d}.json"
        old_json_path = os.path.join(directory, old_json_name)
        new_json_path = os.path.join(renamed_dir, new_json_name)
        
        if os.path.exists(old_pdb_path):
            shutil.copy2(old_pdb_path, new_pdb_path)
            print(f"Duplicated PDB: {old_pdb_name} -> {new_pdb_name}")
        else:
            print(f"PDB file not found: {old_pdb_name}")
        
        if os.path.exists(old_json_path):
            shutil.copy2(old_json_path, new_json_path)
            print(f"Duplicated JSON: {old_json_name} -> {new_json_name}")
        else:
            print(f"JSON file not found: {old_json_name}")
        
        print()

# Use the script
directory_path = '/home/lily/amelie/Workspace/ColabFold/work/final_outputs/output-5TPN-all'
print(f"Analyzing directory: {directory_path}")
ranked_models = analyze_scores(directory_path)

# Duplicate and rename files based on ranking
duplicate_and_rename_files(directory_path, ranked_models)

print("Process completed. Duplicated files with new names are in the 'renamed' subdirectory.")