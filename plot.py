import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# File names
filenames = [
    'data/greedy_greedy.csv', 
    'data/greedy_respectful.csv', 
    'data/greedy_social.csv', 
    'data/conservative_greedy.csv', 
    'data/conservative_respectful.csv', 
    'data/conservative_social.csv',
    'data/considerate_greedy.csv',
    'data/considerate_respectful.csv',
    'data/considerate_social.csv'
]

# Read each file and add a scenario label
dataframes = []
for file in filenames:
    df = pd.read_csv(file)
    df = df[df['timestep'] % 20 == 0]  # Filter to keep only rows where timestep is a multiple of 10
    df['scenario'] = file.replace('data/', '').replace('.csv', '')  # Add scenario as a column with simplified names
    dataframes.append(df)

# Concatenate all dataframes into one
combined_data = pd.concat(dataframes, ignore_index=True)


plt.figure(figsize=(12, 8))
sns.lineplot(data=combined_data, x='timestep', y='alive_queen1', hue='scenario', style='scenario', markers=True)
plt.title('Number of Alive Bees in Queen Bee Over Time by Scenario')
plt.xlabel('Timestep')
plt.ylabel('Number of Alive Bees in Queen Bee')
plt.legend(title='Scenario')
plt.grid(True)
plt.savefig('plots/num_alive_bees.png')
plt.close()

plt.figure(figsize=(12, 8))
sns.lineplot(data=combined_data, x='timestep', y='dead_queen1', hue='scenario', style='scenario', markers=True)
plt.title('Number of Dead Bees in Queen Bee Over Time by Scenario')
plt.xlabel('Timestep')
plt.ylabel('Number of Dead Bees in Queen Bee')
plt.legend(title='Scenario')
plt.grid(True)
plt.savefig('plots/num_dead_bees.png')
plt.close()

plt.figure(figsize=(12, 8))
sns.lineplot(data=combined_data, x='timestep', y='food_queen1', hue='scenario', style='scenario', markers=True)
plt.title('Food Stored in Queen Bee Over Time by Scenario')
plt.xlabel('Timestep')
plt.ylabel('Food Stored in Queen Bee')
plt.legend(title='Scenario')
plt.grid(True)
plt.savefig('plots/food_stored.png')
plt.close()

plt.figure(figsize=(12, 8))
sns.lineplot(data=combined_data, x='timestep', y='health_queen1', hue='scenario', style='scenario', markers=True)
plt.title('Health of Queen Bee Over Time by Scenario')
plt.xlabel('Timestep')
plt.ylabel('Health of Queen Bee')
plt.legend(title='Scenario')
plt.grid(True)
plt.savefig('plots/health.png')
plt.close()

plt.figure(figsize=(12, 8))
sns.lineplot(data=combined_data, x='timestep', y='presence_queen1', hue='scenario', style='scenario', markers=True)
plt.title('Number of Bees Present in Queen Bee Over Time by Scenario')
plt.xlabel('Timestep')
plt.ylabel('Number of Bees Present in Queen Bee')
plt.legend(title='Scenario')
plt.grid(True)
plt.savefig('plots/num_bees_present.png')
plt.close()