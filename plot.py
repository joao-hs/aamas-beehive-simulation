import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# File names
filenames = [
    'data_powerful_wasp/greedy_greedy.csv', 
    'data_powerful_wasp/greedy_respectful.csv', 
    'data_powerful_wasp/greedy_social.csv', 
    'data_powerful_wasp/conservative_greedy.csv', 
    'data_powerful_wasp/conservative_respectful.csv', 
    'data_powerful_wasp/conservative_social.csv',
    'data_powerful_wasp/considerate_greedy.csv',
    'data_powerful_wasp/considerate_respectful.csv',
    'data_powerful_wasp/considerate_social.csv'
]

# Read each file and add a scenario label
dataframes = []
for file in filenames:
    df = pd.read_csv(file)
    df = df[df['timestep'] % 5 == 0]  # Filter to keep only rows where timestep is a multiple of 10
    df['scenario'] = file.replace('data_powerful_wasp/', '').replace('.csv', '')  # Add scenario as a column with simplified names
    dataframes.append(df)

# Concatenate all dataframes into one
combined_data = pd.concat(dataframes, ignore_index=True)


plt.figure(figsize=(12, 8))
sns.lineplot(data=combined_data, x='timestep', y='alive_queen1', hue='scenario', style='scenario', markers=False)
plt.title('Number of Alive Bees in Queen Bee Over Time by Scenario')
plt.xlabel('Timestep')
plt.ylabel('Number of Alive Bees in Queen Bee')
plt.legend(title='Scenario')
plt.grid(True)
plt.savefig('plots_powerful_wasp/num_alive_bees.png')
plt.close()

plt.figure(figsize=(12, 8))
sns.lineplot(data=combined_data, x='timestep', y='dead_queen1', hue='scenario', style='scenario', markers=False)
plt.title('Number of Dead Bees in Queen Bee Over Time by Scenario')
plt.xlabel('Timestep')
plt.ylabel('Number of Dead Bees in Queen Bee')
plt.legend(title='Scenario')
plt.grid(True)
plt.savefig('plots_powerful_wasp/num_dead_bees.png')
plt.close()

plt.figure(figsize=(12, 8))
sns.lineplot(data=combined_data, x='timestep', y='food_queen1', hue='scenario', style='scenario', markers=False)
plt.title('Food Stored in Queen Bee Over Time by Scenario')
plt.xlabel('Timestep')
plt.ylabel('Food Stored in Queen Bee')
plt.legend(title='Scenario')
plt.grid(True)
plt.savefig('plots_powerful_wasp/food_stored.png')
plt.close()

plt.figure(figsize=(12, 8))
sns.lineplot(data=combined_data, x='timestep', y='health_queen1', hue='scenario', style='scenario', markers=False)
plt.title('Health of Queen Bee Over Time by Scenario')
plt.xlabel('Timestep')
plt.ylabel('Health of Queen Bee')
plt.legend(title='Scenario')
plt.grid(True)
plt.savefig('plots_powerful_wasp/health.png')
plt.close()

plt.figure(figsize=(12, 8))
sns.lineplot(data=combined_data, x='timestep', y='presence_queen1', hue='scenario', style='scenario', markers=False)
plt.title('Number of Bees Present in Queen Bee Over Time by Scenario')
plt.xlabel('Timestep')
plt.ylabel('Number of Bees Present in Queen Bee')
plt.legend(title='Scenario')
plt.grid(True)
plt.savefig('plots_powerful_wasp/num_bees_present.png')
plt.close()