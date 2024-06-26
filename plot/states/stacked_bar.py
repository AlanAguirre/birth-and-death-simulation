import matplotlib.pyplot as plt

# Given JSON data
data = {"0": 9901, "1": 28781, "2": 25727, "3": 12439, "4": 4509, "5": 334, "-1": 2941}

# Extract keys and values
states = list(data.keys())
times = list(data.values())

# Assuming all time in one category for now as we have only one category of states
categories = ['Time in States']
category_data = [times]

# Plotting
fig, ax = plt.subplots(figsize=(10, 6))
bottom = [0] * len(states)
for i, cat_times in enumerate(category_data):
    ax.bar(states, cat_times, bottom=bottom, label=categories[i], color=['blue' if state != '-1' else 'red' for state in states])
    bottom = [sum(x) for x in zip(bottom, cat_times)]

# Adding labels and title
plt.xlabel('States')
plt.ylabel('Time in Seconds')
plt.title('Time in Different States')
plt.legend()

# Display the plot
plt.show()
