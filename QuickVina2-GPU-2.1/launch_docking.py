import os
from DockingCalculator import DockingCalculator




df_path = os.path.join(os.getcwd(), 'canser.csv')

# Example usage:
# It is highly recommended to delete files after each calculation (drop_tmp_files=True )
calculator = DockingCalculator(df_path=df_path, 
                               smiles_column='smiles', 
                               disease='sclerosis',
                               drop_tmp_files=False
)
calculator.launch_docking()