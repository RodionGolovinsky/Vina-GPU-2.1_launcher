
import os
import sys
sys.path.append('..')
import pandas as pd
import re
from tqdm import tqdm
import numpy as np 
from prepare_ligand import preprocess_ligand
from datetime import datetime
from rdkit.Chem import Descriptors
from rdkit import Chem
from rdkit import RDLogger
from func_timeout import func_timeout, FunctionTimedOut
RDLogger.DisableLog('rdApp.*')
from typing import List, Tuple, Optional, Literal
import subprocess
import shutil
import time



cases_params = {
    'alzheimer': {'center': (37.36, 32.49, 30.85), 'box_size': (10.186, 11.792, 8.916), 'protein': '4j1r'},
    'parkinson': {'center': (-70.768, 34.864, -21.632),'box_size': (30, 33, 28.5), 'protein': '2xyn'},
    'canser': {'center': (11.672, -4.665, 5.763), 'box_size': (15.148, 19.624, 18.358), 'protein': '8afb'},
    'sclerosis': {'center': (22.511, 8.28, 1.426),'box_size': (23.766, 23.484, 45.742), 'protein': '5vfi'},
    'dyslipidemia': {'center': (99.497, 99.117, 58.981), 'box_size': (30, 25.5, 32.25), 'protein': '7lj9'},
    'resistance': {'center': (12.77, 56.441, -4.57), 'box_size': (26.25, 30.75, 27.75), 'protein': '6nuq'},
}


class DockingCalculator():
    """
    A class for automating the docking process of ligands to a receptor using QuickVina2-GPU.

    Attributes:
        df_smiles (pd.DataFrame): DataFrame containing the SMILES column.
        smiles_column (str): The name of the SMILES column in the DataFrame.
        path_for_results (str): Directory path for saving docking results.
        drop_tmp_files (bool): Whether to delete temporary files after docking.
        path_ligands (str): Directory path for storing ligands.
        opensl_lib (str): Path to the OpenCL binary.
        receptor_pdb_path (str): Path to the receptor's PDB file.
        receptor_pdbqt (str): Path to the receptor's PDBQT file.
        center_coordinates (tuple): Center coordinates for the docking box.
        box_size (tuple): Size of the docking box.
        path_ligands (str): Directory path for storing ligands.

    """
    def __init__(self,
                 df_path: str, 
                 smiles_column: str, 
                 disease: Optional[Literal['alzheimer', 'parkinson', 'cancer', 'sclerosis', 'dyslipidemia', 'resistance']],
                 path_for_results: str = None,
                 receptor_pdb_path: str = None,  
                 center_coordinates: Tuple = None, 
                 box_size: Tuple = None,
                 path_ligands: str = None,
                 drop_tmp_files: bool = False,
                ):
        
        self.df_smiles = pd.read_csv(df_path)
        self.dataset_name = os.path.basename(df_path).replace('.csv', '')
        self.smiles_column = smiles_column

        if self.smiles_column not in self.df_smiles.columns:
            raise ValueError(f'{smiles_column} not found in the DataFrame.')
        
        self.drop_tmp_files = drop_tmp_files
        self.need_to_convert_ligands = True
        if path_ligands is None:
            self.path_ligands = os.path.join(os.getcwd(), 'tmp', 'ligands')
        else:
            self.path_ligands = path_ligands
            self.need_to_convert_ligands = False

        self.opensl_lib = os.getcwd()
        if disease:
            try:
                protein_id = cases_params[disease]['protein']
            except:
                raise ValueError(f'Unsupported disease: {disease}')
            
            print(f'Selected disease: {disease}')
            self.receptor_pdb_path = os.path.join(os.getcwd(), 'proteins_raw',  protein_id + '_protein.pdb')
            self.center_coordinates = cases_params[disease]['center']
            self.box_size = cases_params[disease]['box_size']
        else:
            self.receptor_pdb_path = receptor_pdb_path
            self.center_coordinates = center_coordinates
            self.box_size = box_size
        
        self.receptor_pdbqt = os.path.join(os.getcwd(), 'tmp', 'proteins', f'{self.receptor_pdb_path.split(os.sep)[-1].replace(".pdb", "")}.pdbqt')

        if self.receptor_pdb_path is None:
            raise ValueError('Receptor PDB path is required.')
        
        if self.center_coordinates is None:
            self.center_coordinates = self.calculate_bounding_box()[0]
            print(f'Calculating center coordinates: {self.center_coordinates}')
        
        if self.box_size is None:
            self.box_size = self.calculate_bounding_box()[1]
            print(f'Calculating box size: {self.box_size}')
        
        if path_for_results is None:
            self.path_for_results = self.dataset_name
        else:
            self.path_for_results = path_for_results
            

    def change_opencl_path(self, new_path):
        """
        Updates the path to the OpenCL binary.

        Args:
            new_path (str): The new path to the OpenCL binary.
        """
        self.opensl_lib = new_path  

    def get_config(self, dir):
        """
        Generates a configuration file for docking.

        Args:
            dir (str): Directory to store the configuration file.

        Returns:
            str: Path to the generated configuration file.
        """

        new_path = os.path.join(os.getcwd(), 'tmp', dir, 'config.txt')

        os.makedirs(os.path.join(os.getcwd(), 'tmp', dir), exist_ok=True)

        config_content = f"""
receptor = {self.receptor_pdbqt}
ligand_directory = {os.path.join(os.getcwd(), 'tmp', dir)}
opencl_binary_path = {self.opensl_lib}
center_x = {self.center_coordinates[0]}
center_y = {self.center_coordinates[1]}
center_z = {self.center_coordinates[2]}
size_x = {self.box_size[0]}
size_y = {self.box_size[1]}
size_z = {self.box_size[2]}
thread = 8000
            """

        with open(new_path, "w", encoding='utf-8') as file:
            file.write(config_content)

        return new_path


    def convert_ligands(self): 
        """
        Prepares ligands from the SMILES column, including preprocessing and saving.
        """

        self.df_smiles.drop_duplicates(subset=[self.smiles_column], inplace=True) 
        try:
            self.df_smiles[self.smiles_column] = [
                Chem.MolToSmiles(Chem.MolFromSmiles(smiles)) if Chem.MolFromSmiles(smiles) else None
                for smiles in self.df_smiles[self.smiles_column]
            ]
            raw_length = len(self.df_smiles)
            self.df_smiles.dropna(subset=[self.smiles_column], inplace=True)
            length_dataset = len(self.df_smiles)
            print(f'Final version contains {length_dataset} molecules')
        except Exception as e:
            pass
        print(f'{raw_length - length_dataset} was deleted!')   
        self.df_smiles['ligand_name'] = [f'ligand_{number}' for number in range(length_dataset)]
        for index, row in tqdm(self.df_smiles.iterrows(), total=length_dataset):
            dir_index = index // 5000
            path_to_save = os.path.join(self.path_ligands, f'part_{dir_index}', row['ligand_name'])
            os.makedirs(os.path.dirname(path_to_save), exist_ok=True)
            try:
                func_timeout(2, preprocess_ligand, args=(row[self.smiles_column], path_to_save))
            except FunctionTimedOut:
                print(f"Слишком долго выполняется для {row[self.smiles_column]}")
    def parse_ligand_file(self, file_path):
        
        """
        Parses the ligand docking results file and extracts minimum energy data.

        Args:
            file_path (str): Path to the result file.

        Returns:
            pd.DataFrame: DataFrame containing ligand names and their minimum energies.
        """

        ligand_data = {}
        current_ligand = None
        min_energy = None

        with open(file_path, 'r') as file:
            for line in file:
                if "Refining ligand" in line:
                    if current_ligand and min_energy is not None:
                        ligand_data[current_ligand] = min_energy
                    ligand_match = re.search(r'ligand_\d+', line)
                    if ligand_match:
                        current_ligand = ligand_match.group()
                        min_energy = None 
                
                if re.match(r'\s*1\s+', line) and current_ligand:
                    try:
                        min_energy = float(line.split()[1]) 
                    except (IndexError, ValueError):
                        continue
            if current_ligand and min_energy is not None:
                ligand_data[current_ligand] = min_energy

        return pd.DataFrame(list(ligand_data.items()), columns=['ligand_name', 'Minimum Energy'])
    def run_docking_process(self, config_path: str):

        """
        Runs the docking process using QuickVina2-GPU with the given configuration.

        Args:
            config_path (str): Path to the configuration file.
        """

        result_path = os.path.join(os.path.dirname(config_path), "result.txt")
        print("START DOCKING...")

        cmd = f'./QuickVina2-GPU-2-1 --config {config_path} > {result_path}'
        # print(cmd)
        subprocess.run(cmd, shell=True, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        part_df = self.parse_ligand_file(result_path)
        result_df = pd.merge(part_df, self.df_smiles, on='ligand_name', how='left')
        result_df = result_df.loc[:, ~result_df.columns.str.contains('^Unnamed')]
        os.makedirs(self.path_for_results, exist_ok=True)
        result_df.to_csv(os.path.join(self.path_for_results, f'docked_{os.path.dirname(result_path).split(os.sep)[-1]}.csv'), index=['ligand_name'])

    def calculate_bounding_box(self):
        """
        Calculates the bounding box and center coordinates for the receptor.

        Returns:
            tuple: Center coordinates (x, y, z) and box size (x, y, z).
        """
        
        x_coords, y_coords, z_coords = [], [], []
        with open(self.receptor_pdb_path, "r") as file:
            for line in file:
                if line.startswith("ATOM") or line.startswith("HETATM"):
                    x = float(line[30:38].strip())
                    y = float(line[38:46].strip())
                    z = float(line[46:54].strip())
                    x_coords.append(x)
                    y_coords.append(y)
                    z_coords.append(z)
        
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)
        min_z, max_z = min(z_coords), max(z_coords)
        
        center_x = round((min_x + max_x) / 2)
        center_y = round((min_y + max_y) / 2)
        center_z = round((min_z + max_z) / 2)
        
        size_x = round(max_x - min_x, 2)
        size_y = round(max_y - min_y, 2)
        size_z = round(max_z - min_z, 2)

        return (center_x, center_y, center_z), (size_x, size_y, size_z)
    def convert_receptor(self):
        """
        Converts the receptor PDB file to PDBQT format using Open Babel.
        """

        os.makedirs(os.path.dirname(self.receptor_pdbqt), exist_ok=True)
        cmd = f"obabel {self.receptor_pdb_path} -xr -O {self.receptor_pdbqt}"
        subprocess.run(cmd, shell=True, text=True, capture_output=True)


    def launch_docking(self):
        """
        Launches the full docking workflow, including ligand preparation,
        receptor preparation, and docking execution.
        """

        start = time.time()
        if self.need_to_convert_ligands:
            self.convert_ligands()

        self.convert_receptor()
        end = time.time()
        print(f"The conversion took {end - start} seconds")
        
        start = time.time()
        for dir in os.listdir(self.path_ligands):
            if re.match(r'^part_\d+$', dir):
                config_path = self.get_config(os.path.join(self.path_ligands, dir))
                self.run_docking_process(config_path)
                print(f"{dir} is calculated! ")
        end = time.time()
        print("Docking process is finished!")
        all_results = []
        for file in os.listdir(self.path_for_results):
            if file.startswith('docked_part_'):
                df = pd.read_csv(os.path.join(self.path_for_results, file))
                all_results.append(df)
        df_concat = pd.concat(all_results)
        df_concat.to_csv(os.path.join(self.path_for_results, f'docked_all_{self.dataset_name}.csv'), index=False)
        print(f"The docking took {end - start} seconds")

        if self.drop_tmp_files:
            self.drop_tmp()


    def drop_tmp(self):

        """
        Deletes temporary files and directories created during the docking process.
        """

        shutil.rmtree(os.path.dirname(self.path_ligands))




