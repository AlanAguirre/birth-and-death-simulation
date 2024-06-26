import matplotlib.pyplot as plt

# Given JSON data
data = {"0": 9901, "1": 28781, "2": 25727, "3": 12439, "4": 4509, "5": 334, "-1": 2941}

# Extract keys and values
states = list(data.keys())
times = list(data.values())

# Define a color palette with a specific color for the error state
colors = ['skyblue', 'lightgreen', 'orange', 'purple', 'pink', 'yellow', 'red']
state_colors = [colors[i] if state != '-1' else 'red' for i, state in enumerate(states)]

# Plotting
plt.figure(figsize=(8, 8))
plt.pie(times, labels=states, colors=state_colors, autopct='%1.1f%%', startangle=140)
plt.title('Time in Different States')

# Display the plot
plt.show()
