"""
Scenario:
  A chatbot send messages to a LLM private endpoint.
  The private endpoint has limitations of 240k TPM (Tokens Per Minute).
  The chatbot send messages with the average of 7k tokens and the average time 
  to the endpoint answer a message is 10 seconds.
  When the endpoint receive more than 240k token in less than 1 minute, the endpoint 
  responds with an error.
  The birth rate of messages are a random number between 0 to 20. During the simulation
  we are using the number of tokens in the interval. New slots of message are arriving 
  each 10 seconds.
  The death rate is defined as slots of 40k tokens. To be more specific, we are defining 
  6 slots of 40k to represent the 240k per minute. The unit used in the simulation is tokens,
  and the unit time is seconds.

"""

from __future__ import annotations

import matplotlib.pyplot as plt
import random
import simpy
import json

from simpy import Environment
from typing import Dict, List, NamedTuple, Optional, Tuple


RANDOM_SEED = 2024
TPM = 240  # Tokens Per Minute divided by 1000.
MESSAGE_TOKENS = 7 # Average tokens per message divided by 1000.
NUMBER_OF_SLOTS = 240 # Number of states to be used in the simulation. 
ENDPOINT_SLOT = TPM / NUMBER_OF_SLOTS # Tokens per slot. E.g.: (240 / 6) = 40 tokens.
SIM_TIME = 60*(60*24)  # Simulate until.
MESSAGES_ANSWERED = 10 # Average message response time.


class Endpoint(NamedTuple):
    """Class to represent the endpoint."""
    counter: simpy.Resource
    slots: List[str]
    tokens_available: Dict[str, int]
    slot_full: Dict[str, simpy.Event]
    when_slot_full: Dict[str, List[Optional[float]]]
    messages_error: Dict[str, int]
    slot_counter_token: Dict[str, int]
    slot_counter_token_loss: Dict[str, int]
    slot_counter_message_loss: Dict[str, int]
    slot_timer: Dict[str, float]
    global_slot: List[str]
    global_slot_timer:  List[float]
    global_message_counter:  List[int]
    global_token_counter:  List[int]



def _retrieve_slot_available(endpoint: Endpoint) -> int:
    """Method to retrieve the next slot available."""
    for slot in endpoint.slots:
        if endpoint.tokens_available[slot] > 0:
            return slot
    return endpoint.slots[-1]


def _set_slot_timer(env: Environment, endpoint: Endpoint, slot: int) -> None:
    if slot != endpoint.global_slot[0]:
        endpoint.global_slot[0] = slot
        timer = env.now
        endpoint.slot_timer[slot] += (timer - endpoint.global_slot_timer[0])
        endpoint.global_slot_timer[0] = timer


def _slot_operation(env: Environment, endpoint: Endpoint, slot: int, message_tokens: int) -> Tuple[int, int]:
    """Method to check if a slot has tokens available.
    
       If the slot has tokens available, the method complete the operation discounting the tokens.
       Otherwise, the method is moving to the next slot available.
    """
    if endpoint.tokens_available[slot] <= 0:
        _set_slot_timer(env, endpoint, slot)

        if slot == endpoint.slots[-1]:
            return slot, message_tokens
        
        new_slot = _retrieve_slot_available(endpoint=endpoint)
        return _slot_operation(env=env, endpoint=endpoint, slot=new_slot,message_tokens=message_tokens)

    if endpoint.tokens_available[slot] < message_tokens:
        message_tokens -= endpoint.tokens_available[slot]
        endpoint.slot_full[slot].succeed()
        endpoint.when_slot_full[slot].append(env.now)
        endpoint.slot_counter_token[slot] += endpoint.tokens_available[slot]
        endpoint.tokens_available[slot] = 0
        _set_slot_timer(env, endpoint, slot)
        new_slot = _retrieve_slot_available(endpoint=endpoint)
        return _slot_operation(env=env, endpoint=endpoint, slot=new_slot,message_tokens=message_tokens)

    _set_slot_timer(env, endpoint, slot)

    return slot, message_tokens


def _free_tokens(env: Environment, endpoint: Endpoint, slot: int, tokens: int) -> bool:
    """Method to free tokens used."""
    endpoint.slot_full[slot] = env.event()
    endpoint.tokens_available[slot] += tokens
    _set_slot_timer(env, endpoint, slot)

    if endpoint.tokens_available[slot] > ENDPOINT_SLOT:
        new_tokens = endpoint.tokens_available[slot] - ENDPOINT_SLOT
        endpoint.tokens_available[slot] = ENDPOINT_SLOT
        return _free_tokens(env=env, endpoint=endpoint, slot=(slot - 1), tokens=new_tokens)
    return True


def _answer_messages(env: Environment, endpoint: Endpoint):
    """This method is to represent the death process and simulate the tokens be answered
       by the endpoint.
       
    """
    t = random.expovariate(1.0 / MESSAGES_ANSWERED)

    yield env.timeout(t)
    
    slot = _retrieve_slot_available(endpoint=endpoint)
    
    _free_tokens(env=env, endpoint=endpoint, slot=slot, tokens=MESSAGE_TOKENS)


def chatbot(env: Environment, slot: int, num_messages: int, endpoint: Endpoint):
    """Method to represent the chatbot sending messages and receiving messages.

    """
    with endpoint.counter.request() as message_turn:

        result = yield message_turn | endpoint.slot_full[slot]

        message_tokens = num_messages * MESSAGE_TOKENS
        endpoint.global_message_counter[0] += num_messages
        endpoint.global_token_counter[0] += message_tokens
        slot, message_tokens = _slot_operation(env=env, endpoint=endpoint, slot=slot, message_tokens=message_tokens)
        
        if endpoint.tokens_available[slot] > 0:
            endpoint.tokens_available[slot] -= message_tokens
            endpoint.slot_counter_token[slot] += message_tokens
            for _ in range(0, num_messages):
                env.process(_answer_messages(env, endpoint))
        else:
            endpoint.slot_counter_token_loss[slot] += message_tokens
            endpoint.slot_counter_message_loss[slot] += num_messages


def message_arrivals(env: Environment, endpoint: Endpoint, interval: int):
    """Create new *messages* until the sim time reaches SIM_TIME.
    
       This method represent the birth process.
    """
    while True:
        slot = _retrieve_slot_available(endpoint)

        num_messages = random.randint(0, 20)
        env.process(chatbot(env, slot, num_messages, endpoint))
        # Send a new message in a average time of interval
        # t = random.expovariate(interval)
        yield env.timeout(interval)


# Setup and start the simulation
print('Chatbot simulation')
random.seed(RANDOM_SEED)
env = simpy.Environment()


slots = [i for i in range(0, NUMBER_OF_SLOTS, 1)]

endpoint = Endpoint(
    counter=simpy.Resource(env, capacity=1),
    slots=slots,
    tokens_available={slot: ENDPOINT_SLOT for slot in slots},
    slot_full={slot: env.event() for slot in slots},
    when_slot_full={slot: [] for slot in slots},
    messages_error={slot: 0 for slot in slots},
    slot_counter_token={slot: 0 for slot in slots},
    slot_counter_token_loss={slot: 0 for slot in slots},
    slot_counter_message_loss={slot: 0 for slot in slots},
    slot_timer={slot: 0 for slot in slots},
    global_slot= [-1],
    global_slot_timer= [0.0],
    global_message_counter= [0],
    global_token_counter= [0],
)


# Start process and run
env.process(message_arrivals(env, endpoint, 10))
env.run(until=SIM_TIME)

timer = env.now
endpoint.slot_timer[endpoint.global_slot[0]] += (timer - endpoint.global_slot_timer[0])

# Analysis/results
count_time = 0
count_tokens = 0
for slot in slots:
    if endpoint.slot_full[slot]:
        full_time = endpoint.when_slot_full[slot]
        num_errors = endpoint.messages_error[slot]
        available_tokens = endpoint.tokens_available[slot]
        slot_counter_token = endpoint.slot_counter_token[slot]
        slot_counter_token_loss = endpoint.slot_counter_token_loss[slot]
        slot_counter_message_loss = endpoint.slot_counter_message_loss[slot]
        slot_timer = endpoint.slot_timer[slot]
        count_time += slot_timer
        count_tokens += slot_counter_token + slot_counter_token_loss
        print(
            f'The slot "{slot}" was full at {full_time} seconds '
            f'after start message arrives. '
            f'Available tokens {available_tokens}'
        )
        print(f'  Number of message errors: {num_errors}')
        print(f'  Slot counter token: {slot_counter_token}')
        print(f'  Slot counter token loss: {slot_counter_token_loss}')
        print(f'  Slot counter message loss: {slot_counter_message_loss}')
        print(f'  Timer in state: {slot_timer}')

print('Total time: ', count_time)
print('Total tokens: ', count_tokens)
print('Total global messages: ', endpoint.global_message_counter[0])
print('Total global tokens: ', endpoint.global_token_counter[0])
# Simulation Plot
# Create a figure and a set of subplots
fig, ax = plt.subplots()

# Iterate over each slot and plot the times
for slot, times in endpoint.when_slot_full.items():
    # Convert slot to integer for plotting

    ax.scatter(times, [slot] * len(times), 
               label=f'Slot {slot}')

# Add labels and title
ax.set_xlabel('Time (seconds)')
ax.set_ylabel('Slot')
ax.set_title('Slot Full Times')
ax.legend(loc='lower right')


# Show plot
plt.show()


# State Plot

# Extract keys and values
# states = list(endpoint.slot_counter.keys())
# times = list(endpoint.slot_counter.values())

# Define a color palette with a specific color for the error state
# colors = ['skyblue', 'lightgreen', 'orange', 'purple', 'pink', 'yellow', 'red']
# state_colors = [colors[i] if state != '-1' else 'red' for i, state in enumerate(states)]

# Plotting
# plt.figure(figsize=(8, 8))
# plt.pie(times, labels=states, colors=state_colors, autopct='%1.1f%%', startangle=140)
# plt.title('Time in Different States')

# Display the plot
# plt.show()
