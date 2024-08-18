# AFM-LIS Environment Setup

To set up the environment for running the AlphaFold Local Interaction Score (AFM-LIS) analysis script, follow these steps:

1. Create a new Conda environment:
conda create -n afm_lis python=3.9 -y

2. Activate the new environment:
conda activate afm_lis

3. Install required packages from requirements.txt:
pip install -r requirements.txt

# Usage
After setting up the environment, always activate it before running the script:
conda activate afm_lis

# Running the script
python alphafold_interaction_scores_github_20240421.py

# Deactivating the environment
When you're done, you can deactivate the environment:
conda deactivate