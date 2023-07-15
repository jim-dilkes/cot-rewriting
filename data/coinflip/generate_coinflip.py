# Data structure from "Large Language Models are Zero-Shot Reasoners" (Kojima et. al. 2023):
# Code original

# As for Coin Flip, we use the following template.
# We randomly select human names from namesdataset library and insert them into {Name1} through {Name4}.
# We also randomly pick up “flips” or “does not flip” and insert the phrase into each {flips | does not flip} part, respectively.

# Example pattern
#"A coin is heads up. {Name1} {flips | does not flip} the coin.
# {Name2} {flips | does not flip} the coin.
# {Name3} {flips | does not flip} the coin.
# {Name4} {flips | does not flip} the coin.
# Is the coin still heads up? Note that "flip" here means "reverse".’


# Generate a dataset of coin flip examples

from names_dataset import NameDataset # pip install names-dataset
import random
import csv

nd = NameDataset()

# Variables
n_examples = 500
n_flips_options = [4]
initial_states = ['heads']
# initial_states = ['heads', 'tails']

n_per_country_gender = 20
country_list = ['US','ES','DE','IT','FR','GB', 'JP', 'BR', 'CA', 'IN', 'CN', 'MX', 'ID', 'NL', 'AT', 'BE']

name_list = []
for country in country_list:
    gen_names = nd.get_top_names(n=n_per_country_gender, country_alpha2=country)
    name_list.extend(gen_names[country]['M'])
    name_list.extend(gen_names[country]['F'])
name_list = list(set(name_list))

print(f"NAME LIST: {name_list}")


flip_list = ['flips', 'does not flip']
def generate_coinflip_example():
    
    # Generate random components
    n_flips = random.sample(n_flips_options, 1)[0]
    start_state = random.sample(initial_states, 1)[0]
    names = random.sample(name_list, n_flips)
    flips = random.choices(flip_list, k=n_flips)
    
    # Generate string
    return_string = f"A coin is {start_state} up. "
    for i in range(n_flips):
        return_string += f"{names[i]} {flips[i]} the coin. "
    return_string += "Is the coin still heads up? Note that 'flip' here means 'reverse'."
    
    # Generate final state
    total_flips = flips.count('flips')
    not_start_state = 'heads' if start_state == 'tails' else 'tails'
    final_state = start_state if total_flips % 2 == 0 else not_start_state
    
    return return_string, final_state, total_flips, n_flips


coinflip_dataset = [generate_coinflip_example() for _ in range(n_examples)]
with open('coinflip_dataset.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerow(['description', 'final_state', 'total_flips', 'total_events'])
    for example in coinflip_dataset:
        writer.writerow([example[0], example[1], example[2], example[3]])
