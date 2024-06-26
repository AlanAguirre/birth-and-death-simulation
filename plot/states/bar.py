import matplotlib.pyplot as plt

# Given JSON data
data = {"0": 9901, "1": 28781, "2": 25727, "3": 12439, "4": 4509, "5": 334, "-1": 2941}

# Extract keys and values
states = list(data.keys())
times = list(data.values())

# Plotting
plt.figure(figsize=(10, 6))
plt.bar(states, times, color=['blue' if state != '-1' else 'red' for state in states])

# Adding labels and title
plt.xlabel('States')
plt.ylabel('Time in Seconds')
plt.title('Time in Different States')

# Highlighting the error state
plt.text(states.index('-1'), data['-1'] + 500, 'Error State', color='red', ha='center')

# Display the plot
plt.show()
