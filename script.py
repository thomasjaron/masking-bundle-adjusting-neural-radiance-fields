import os
import subprocess
import yaml

# Function to update the YAML file with multiple key-value pairs
def update_yaml(file_path, updates):
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)

    # Update the data with the provided key-value pairs
    for key, value in updates.items():
        data[key] = value

    with open(file_path, 'w') as file:
        yaml.safe_dump(data, file, default_flow_style=False)

# Parameters
yaml_file_path = 'options/planar.yaml'
dataset_values = ['batch1', 'batch2', 'batch3', 'batch4', 'batch5', 'batch6', 'batch7']

for i in range(4, 7):
    dataset = dataset_values[i]

    # Case 1: RGB loss without masks
    # updates = {
    #     'dataset': dataset,
    #     'use_masks': False,
    #     'use_edges': False,
    #     'alpha_initial': 1,
    #     'alpha_final': 1,
    # }
    # update_yaml(yaml_file_path, updates)
    # command = f"python train.py --group=alignment --model=planar --yaml=planar --name={dataset}_usemasksFalse_useedgesFalse_alpha1 --seed=3 --barf_c2f=[0,0.4]"
    # subprocess.run(command, shell=True)

    # Case 2: Edge loss without masks (RGB loss weight = 0)
    # updates = {
    #     'dataset': dataset,
    #     'use_masks': False,
    #     'use_edges': True,
    #     'alpha_initial': 1,
    #     'alpha_final': 1,
    # }
    # update_yaml(yaml_file_path, updates)
    # command = f"python train.py --group=alignment --model=planar --yaml=planar --name={dataset}_usemasksFalse_useedgesTrue_alpha1 --seed=3 --barf_c2f=[0,0.4]"
    # subprocess.run(command, shell=True)

    # # Case 3: RGB loss without masks + Edge loss without masks (weight high to low)
    # updates = {
    #     'dataset': dataset,
    #     'use_masks': False,
    #     'use_edges': True,
    #     'alpha_initial': 1,
    #     'alpha_final': 0,
    # }
    # update_yaml(yaml_file_path, updates)
    # command = f"python train.py --group=alignment --model=planar --yaml=planar --name={dataset}_usemasksFalse_useedgesTrue_alphahigh2low --seed=3 --barf_c2f=[0,0.4]"
    # subprocess.run(command, shell=True)

    # Case 4: RGB loss with masks + Edge loss with masks (weight high to low)
    updates = {
        'dataset': dataset,
        'use_masks': True,
        'use_edges': True,
        'alpha_initial': 1,
        'alpha_final': 0,
    }
    update_yaml(yaml_file_path, updates)
    command = f"python train.py --group=alignment --model=planar --yaml=planar --name={dataset}_usemasksTrue_useedgesTrue_alphahigh2low --seed=3 --barf_c2f=[0,0.4]"
    subprocess.run(command, shell=True)

    # # Case 5: RGB loss without masks + Edge loss without masks (weight low to high)
    # updates = {
    #     'dataset': dataset,
    #     'use_masks': False,
    #     'use_edges': True,
    #     'alpha_initial': 0,
    #     'alpha_final': 1,
    # }
    # update_yaml(yaml_file_path, updates)
    # command = f"python train.py --group=alignment --model=planar --yaml=planar --name={dataset}_usemasksFalse_useedgesTrue_alphalow2high --seed=3 --barf_c2f=[0,0.4]"
    # subprocess.run(command, shell=True)

    # Case 6: RGB loss with masks + Edge loss with masks (weight low to high)
    updates = {
        'dataset': dataset,
        'use_masks': True,
        'use_edges': True,
        'alpha_initial': 0,
        'alpha_final': 1,
    }
    update_yaml(yaml_file_path, updates)
    command = f"python train.py --group=alignment --model=planar --yaml=planar --name={dataset}_usemasksTrue_useedgesTrue_alphalow2high --seed=3 --barf_c2f=[0,0.4]"
    subprocess.run(command, shell=True)

    # Case 7: RGB loss without masks + Edge loss without masks (equal weight)
    # updates = {
    #     'dataset': dataset,
    #     'use_masks': False,
    #     'use_edges': True,
    #     'alpha_initial': 0.5,
    #     'alpha_final': 0.5,
    # }
    # update_yaml(yaml_file_path, updates)
    # command = f"python train.py --group=alignment --model=planar --yaml=planar --name={dataset}_usemasksFalse_useedgesTrue_alphaequal --seed=3 --barf_c2f=[0,0.4]"
    # subprocess.run(command, shell=True)

    # Case 8: RGB loss with masks + Edge loss with masks (equal weight)
    updates = {
        'dataset': dataset,
        'use_masks': True,
        'use_edges': True,
        'alpha_initial': 0.5,
        'alpha_final': 0.5,
    }
    update_yaml(yaml_file_path, updates)
    command = f"python train.py --group=alignment --model=planar --yaml=planar --name={dataset}_usemasksTrue_useedgesTrue_alphaequal --seed=3 --barf_c2f=[0,0.4]"
    subprocess.run(command, shell=True)

    # Case 9: RGB loss with masks + without Edge loss
    updates = {
        'dataset': dataset,
        'use_masks': True,
        'use_edges': False,
        'alpha_initial': 0,
        'alpha_final': 0,
    }
    update_yaml(yaml_file_path, updates)
    command = f"python train.py --group=alignment --model=planar --yaml=planar --name={dataset}_usemasksTrue_useedgesFalse_alpha0 --seed=3 --barf_c2f=[0,0.4]"
    subprocess.run(command, shell=True)

"""
1_use_masks_true     1_use_edges_true    #Beides Gleichgewichtet
1_use_masks_true     0_use_edges_false   #RGB with mask
0_use_masks_false    1_use_edges_true    #edge and rgb 
0_use_masks_false    0_use_edges_false   #only rgb

1->0_use_masks_true  0->1_use_edges_true #high rgb to low rgb
0->1_use_masks_true  1->0_use_edges_true #low rgb to high rgb

1->0_use_masks_false  0->1_use_edges_true #high rgb to low rgb
0->1_use_masks_false  1->0_use_edges_true #low rgb to high rgb
"""