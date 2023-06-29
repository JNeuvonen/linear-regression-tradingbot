import torch
import pandas as pd


df = pd.read_csv('column_order_for_live_env.csv', header=0)
features = df['Column Names'].tolist()

state_dict = torch.load('all_models/epoch_130.pt')


weights = state_dict['linear.weight']

sorted_weights = weights.view(-1).sort(descending=True)


sorted_weights_indices = sorted_weights.indices
sorted_weights_values = sorted_weights.values

sorted_weights_dict = {features[i]: weight.item() for i, weight in zip(
    sorted_weights_indices, sorted_weights_values)}

for feature, weight in sorted_weights_dict.items():
    print(f"Feature: {feature}, Weight: {weight}")
