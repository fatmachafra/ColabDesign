#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 14:29:07 2024

@author: fatmachafra
"""

#import csv
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import numpy as np
parser = argparse.ArgumentParser(description='Running Chimerax log parser for summary csv file and graphing')
parser.add_argument("-i", '--input_html_list', help='a comma separated string of the files to be considered during the csv file and plot generation')
parser.add_argument("-c", '--output_csv')
parser.add_argument("-p", '--output_plot')
args = parser.parse_args()
print(args)

# Specify input HTML file and output CSV file
output_csv_file = args.output_csv   # Desired output CSV file name
input_html_list = args.input_html_list
converted_list = input_html_list.split(',')
print('converted_list', converted_list)
output_plot_file = args.output_plot

# ['con', 'dgram_cce', 'exp_res', 'fape', 'helix', 'i_con', 'i_pae', 'pae', 'plddt', 'rmsd', 'seq_ent']
weights_to_test_number = {'weight_test_0': '0.0_0.133_0.0_0.0_0.0_0.0_0.133_0.267_0.267_0.133_0.067',
                          'weight_test_1': '0.0_0.148_0.0_0.0_0.0_0.0_0.074_0.296_0.296_0.148_0.037',
                          'weight_test_2': '0.0_0.151_0.0_0.0_0.0_0.0_0.075_0.302_0.302_0.151_0.019',
                          'weight_test_3': '0.0_0.152_0.0_0.0_0.0_0.0_0.076_0.304_0.304_0.152_0.013',
                          'weight_test_4': '0.0_0.078_0.0_0.0_0.0_0.0_0.039_0.392_0.392_0.078_0.02',
                          'weight_test_5': '0.0_0.131_0.0_0.0_0.0_0.0_0.066_0.328_0.328_0.131_0.016',
                          'weight_test_6': '0.0_0.211_0.0_0.0_0.0_0.0_0.105_0.211_0.211_0.211_0.053',
                          'weight_test_7': '0.0_0.229_0.0_0.0_0.0_0.0_0.057_0.229_0.229_0.229_0.029',
                          'weight_test_8': '0.0_0.19_0.0_0.0_0.0_0.0_0.19_0.19_0.19_0.19_0.048',
                          'weight_test_9': '0.0_0.216_0.0_0.0_0.0_0.0_0.108_0.216_0.216_0.216_0.027',
                          'weight_test_10': '0.0_1.0_0.0_0.0_0.0_0.0_0.0_0.0_0.0_0.0_0.0',
                          'weight_test_11': '0.0_0.0_0.0_0.0_0.0_0.0_0.0_0.0_0.0_1.0_0.0',
                          'weight_test_12': '0.0_0.0_0.0_0.0_0.0_0.0_1.0_0.0_0.0_0.0_0.0'
}

reversed_dict = {value: key for key, value in weights_to_test_number.items()}

def parse_html_to_csv(html_file_list, csv_file):
    model_names = []
    model_numbers = []
    rmsd_values = []
    for html_file in html_file_list:
        print('html_file', html_file)
        with open(html_file, 'r') as file:
            html_content = file.read()

        # Split the content by lines for easier processing
        lines = html_content.splitlines()
        
        # last_model_name = None  # Variable to hold the last model name

        for line in lines:
            # Check for the line containing the model name
            if "Matchmaker" in line:
                # Extract model name from the line
                parts = line.split("with")
                if len(parts) > 1:
                    # Get the second model name (after "with")
                    last_model_name = parts[1].strip().split(",")[0]  # Take only up to the first comma
                    model_names.append(last_model_name)  # Store the second model name

            # Check for the line containing the RMSD value
            
            # Statements for AF multimer runs with all vs all comparison of both chains
            # this if statement is for 8EE2
            # if "RMSD between 791 atom pairs is" in line:
            # this if statement is for 8H3X:
            # if "RMSD between 953 atom pairs is" in line:
            # this if statement is for 8DCE:
            # if "RMSD between 1813 atom pairs is" in line:
            
            # Statements for AF Design runs with only the nb chain Calpha comparison
            if 'rmsd</a> #' in line:
                # Extract chain information
                parts = line.split()
                print('rmsd parts:', parts)
                chain_info = parts[4].split("/")[0].strip('#').split('.')[-1]
                print('chain_info', chain_info)
                model_numbers.append(chain_info)
                
            if "RMSD between 100 atom pairs is" in line:
                # Extract the RMSD value
                # parts = line.split("RMSD between 791 atom pairs is")
                parts = line.split("RMSD between 100 atom pairs is")
                print(parts)
                if len(parts) > 1:
                    rmsd_part = parts[1].strip()
                    print(rmsd_part)
                    
                    
                    # Now split by spaces and find the second number (the RMSD value)
                    rmsd_value = rmsd_part.split("<br>")[0]  # The third element is the RMSD value
                    print(rmsd_value)
                    if last_model_name:  # Ensure there is a valid model name to associate with
                        rmsd_values.append(float(rmsd_value))  # Append RMSD value

        
    for key in model_names:
        print(f"model_names: {key}")
    for key in rmsd_values:
        print(f"rmsd_values: {key}") 
        
    # Create a DataFrame from model_names and rmsd_values
    df = pd.DataFrame({
        'Model Name': model_names,
        'Model Number': model_numbers,
        'RMSD Value': rmsd_values
    })

    # Extract recycle number, rank, and model number from Model Name
    df['Num Recycles'] = df['Model Name'].str.extract(r'num_recycles_(\d+)').astype(int)  # Get number after 'r'
    df['Extracted Weight'] = df['Model Name'].str.extract(r'(?<=weights_)([0-9_.]+)(?=_c)').astype(str)
    print(df['Extracted Weight'][0])
    df['Weight Test Number'] = df['Extracted Weight'].map(reversed_dict).fillna(0)
    df['Learning Rate'] = df['Model Name'].str.extract(r'(?<=_)([0-9.]+)(?=_models)').astype(float)
    # df['Rank'] = df['Model Name'].str.extract(r'rank_(\d+)').astype(int)
    # df['Model Number'] = df['Model Name'].str.extract(r'model_(\d+)').astype(int)


    print(df[['Model Name', 'Model Number', 'Extracted Weight', 'Weight Test Number', 'Num Recycles', 'RMSD Value', 'Learning Rate']])  # Debugging output
    # Write extracted data to CSV
    df.to_csv(csv_file, index=False)
    print(f"Data successfully written to {csv_file}")
    return df



# Run the parsing function
df_final = parse_html_to_csv(converted_list, output_csv_file)

# need to plot based on the different weights and not the different number of recycles in this case. So, have to extract weight information from the Model Name column
# also can color based on the learning rate
# Categorical scatterplot
# Print debugging information
print("Weight Test Number dtype:", df_final['Weight Test Number'].dtype)
print("Weight Test Number unique values:", df_final['Weight Test Number'].unique())

from matplotlib.colors import ListedColormap

def plot_rmsd_weights_models_learning_rates(df, outfilename=None):
    """
    Create a categorical scatterplot of RMSD values vs Weight Test Numbers,
    colored by Model Number, with different shapes for Learning Rates.
    """
    fig, ax = plt.subplots(figsize=(16, 10))

    # Convert all values in 'Weight Test Number' to strings
    df['Weight Test Number'] = df['Weight Test Number'].astype(str)

    # Sort the categories to ensure correct order
    categories = sorted(df['Weight Test Number'].unique(), key=lambda x: int(x.split('_')[-1]) if x.startswith('weight_test_') else -1)
    
    # Create a categorical type with ordered categories
    df['Weight Test Number'] = pd.Categorical(df['Weight Test Number'], categories=categories, ordered=True)

    # Get unique learning rates and assign shapes
    learning_rates = sorted(df['Learning Rate'].unique())
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x']
    marker_dict = dict(zip(learning_rates, markers[:len(learning_rates)]))

    # Create a color map for Model Numbers
    unique_models = sorted(df['Model Number'].unique())
    n_models = len(unique_models)
    colors = plt.cm.tab20(np.linspace(0, 1, n_models))
    color_dict = dict(zip(unique_models, colors))

    # Plot each learning rate separately
    for lr in learning_rates:
        df_lr = df[df['Learning Rate'] == lr]
        scatter = ax.scatter(df_lr['Weight Test Number'].cat.codes, df_lr['RMSD Value'], 
                             c=[color_dict[m] for m in df_lr['Model Number']],
                             marker=marker_dict[lr], s=70, alpha=0.7, label=f'LR: {lr}')

    ax.set_xlabel('Weight Test Number', fontsize=12)
    ax.set_ylabel('RMSD Value', fontsize=12)
    ax.set_title('RMSD Values for Nanobody chain over different\n'
                 'weight combinations, model numbers, and learning rates', fontsize=14)

    # Set x-axis ticks and labels
    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(categories, rotation=45, ha='right')

    # Create custom legend for Model Numbers
    model_legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                             label=f'Model {m}', markerfacecolor=color_dict[m], markersize=10)
                             for m in unique_models]

    # Add legends
    lr_legend = ax.legend(title='Learning Rate', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.add_artist(lr_legend)  # Add learning rate legend to the plot
    ax.legend(handles=model_legend_elements, title='Model Number', 
              bbox_to_anchor=(1.05, 0.6), loc='center left')

    plt.tight_layout()

    if outfilename:
        plt.savefig(outfilename, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {outfilename}")
    else:
        plt.show()

    plt.close()

# Run the plotting function
plot_rmsd_weights_models_learning_rates(df_final, output_plot_file)

'''
def plot_rmsd_vs_model_no(df, output_path):
    plt.figure(figsize=(10, 6))

    # Normalize the ranks for color mapping
    norm = plt.Normalize(df['Rank'].min(), df['Rank'].max())
    cmap = plt.get_cmap('tab10')  # Using same colormap used in the AF Design interation plots 

    # Group by Model Number
    grouped = df.groupby('Model Number')

    for name, group in grouped:
        # Sort values by Num Recycles to ensure lines connect correctly
        group = group.sort_values('Num Recycles')
        
        # Get colors based on normalized rank values
        colors = cmap(norm(group['Rank']))

        # Plotting the points with color mapping
        plt.scatter(group['Num Recycles'], group['RMSD Value'], 
                    label=f'Model {name}', alpha=0.7, color=colors)
        
        
        # Connecting lines between points of the same model with corresponding colors
        for i in range(len(group) - 1):
            plt.plot(group['Num Recycles'].iloc[i:i+2], 
                     group['RMSD Value'].iloc[i:i+2], 
                     linestyle='-', alpha=0.5, color=colors[i])  # Use the color of the first point
    plt.title('RMSD vs Number of Recycles')
    plt.xlabel('Number of Recycles')
    plt.ylabel('RMSD Value (Å)')
   
   # Add a color bar that indicates rank values
    cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=plt.gca())  # Use current axes for colorbar
    cbar.set_label('Rank')  # Label for the color bar
 
   
    plt.legend(title='Model Number')
    plt.grid()
    
    # Save plot to specified output path
    plt.savefig(output_path)
    plt.close()  # Close the figure to free up memory
    
# Specify output path for plot
output_plot_path = '/Users/fatmachafra/Desktop/8DCE results/8DCE_plots/plot_log_8DCE_antigen_4A_contact_MSA_num_recycles_0_multimer_v2_8043e.png'  # Desired output plot file name

# Call the plotting function
plot_rmsd_vs_recycles(df_final, output_plot_path)

print(f"Plot successfully saved to {output_plot_path}")
'''