"""
Scenario:
  A chatbot send messages to a LLM private endpoint.
  The private endpoint has limitations of 240k TPM (Tokens Per Minute).
  The chatbot send messages with the average of 7k tokens and the average time 
  to the endpoint answer a message is 10 seconds.
  When the endpoint receive more than 240k token in less than 1 minute, the endpoint 
  responds with an error.
  The birth rate of messages are a random number between 0 to 10. During the simulation
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

from simpy import Environment
from typing import Dict, List, NamedTuple, Optional, Tuple


RANDOM_SEED = 2024
TPM = 240  # Tokens Per Minute divided by 1000.
MESSAGE_TOKENS = 7 # Average tokens per message divided by 1000.
NUMBER_OF_SLOTS = 6 # Number of states to be used in the simulation
ENDPOINT_SLOT = TPM / NUMBER_OF_SLOTS # Tokens per slot. E.g.: (240 / 6) = 40 tokens.
SIM_TIME = 60*(60*2)  # Simulate until.
MESSAGES_ANSWERED = 10 # Average message response time.


class Endpoint(NamedTuple):
    """Class to represent the endpoint."""
    counter: simpy.Resource
    slots: List[str]
    tokens_available: Dict[str, int]
    slot_full: Dict[str, simpy.Event]
    when_slot_full: Dict[str, List[Optional[float]]]
    messages_error: Dict[str, int]


def _retrieve_slot_available(endpoint: Endpoint) -> int:
    """Method to retrieve the next slot available."""
    for slot in endpoint.slots:
        if endpoint.tokens_available[slot] > 0:
            return slot
    return -1


def _slot_operation(env: Environment, endpoint: Endpoint, slot: int, message_tokens: int) -> Tuple[int, int]:
    """Method to check if a slot has tokens available.
    
       If the slot has tokens available, the method complete the operation discounting the tokens.
       Otherwise, the method is moving to the next slot available.
    """
    if endpoint.tokens_available[slot] <= 0:
        new_slot = _retrieve_slot_available(endpoint=endpoint)
        if new_slot == -1:
            return new_slot, message_tokens
        return _slot_operation(env=env, endpoint=endpoint, slot=new_slot,message_tokens=message_tokens)

    if endpoint.tokens_available[slot] < message_tokens:
        message_tokens -= endpoint.tokens_available[slot]
        endpoint.slot_full[slot].succeed()
        endpoint.when_slot_full[slot].append(env.now)
        endpoint.tokens_available[slot] = 0
        new_slot = _retrieve_slot_available(endpoint=endpoint)
        if new_slot == -1:
            return new_slot, message_tokens
        return _slot_operation(env=env, endpoint=endpoint, slot=new_slot,message_tokens=message_tokens)

    return slot, message_tokens


def _error_operation(env: Environment, endpoint: Endpoint, slot: int = -1) -> None:
    """Method to commit the error in the slot."""
    endpoint.when_slot_full[slot].append(env.now)
    endpoint.messages_error[slot] += 1


def _free_tokens(env: Environment, endpoint: Endpoint, slot: int, tokens: int) -> bool:
    """Method to free tokens used."""
    endpoint.slot_full[slot] = env.event()
    endpoint.tokens_available[slot] += tokens

    if endpoint.tokens_available[slot] > ENDPOINT_SLOT:
        new_tokens = endpoint.tokens_available[slot] - ENDPOINT_SLOT
        endpoint.tokens_available[slot] = ENDPOINT_SLOT
        if slot > 0:
            return _free_tokens(env=env, endpoint=endpoint, slot=(slot - 1), tokens=new_tokens)
    return True


def _answer_messages(env: Environment, endpoint: Endpoint, num_messages: int):
    """This method is to represent the death process and simulate the tokens be answered
       by the endpoint.
       
    """
    t = 0
    if num_messages > 0:
        u = num_messages * MESSAGES_ANSWERED
        t = random.expovariate(1.0 / random.randint(min(6, u), min(30, u)))

    yield env.timeout(t)
    message_tokens = num_messages * MESSAGE_TOKENS
    
    slot = _retrieve_slot_available(endpoint=endpoint)
    if slot == -1:
        slot = NUMBER_OF_SLOTS - 1
    
    _free_tokens(env=env, endpoint=endpoint, slot=slot, tokens=message_tokens)


def chatbot(env: Environment, slot: int, num_messages: int, endpoint: Endpoint):
    """Method to represent the chatbot sending messages and receiving messages.

    """
    with endpoint.counter.request() as message_turn:

        result = yield message_turn | endpoint.slot_full[slot]

        if message_turn not in result:
            _error_operation(env=env, endpoint=endpoint, slot=slot)
            return
        
        message_tokens = num_messages * MESSAGE_TOKENS
        slot, message_tokens = _slot_operation(env=env, endpoint=endpoint, slot=slot, message_tokens=message_tokens)

        if slot == -1:
            _error_operation(env=env, endpoint=endpoint, slot=slot)
            return
        
        endpoint.tokens_available[slot] -= message_tokens

        env.process(_answer_messages(env, endpoint, num_messages))


def message_arrivals(env: Environment, endpoint: Endpoint, interval: int):
    """Create new *messages* until the sim time reaches SIM_TIME.
    
       This method represent the birth process.
    """
    while True:
        slot = _retrieve_slot_available(endpoint)

        num_messages = random.randint(0, 10)
        env.process(chatbot(env, slot, num_messages, endpoint))
        # Send a new message in a average time of interval
        # t = random.expovariate(interval)
        yield env.timeout(interval)


# Setup and start the simulation
print('Chatbot simulation')
random.seed(RANDOM_SEED)
env = simpy.Environment()


slots = [i for i in range(0, NUMBER_OF_SLOTS, 1)]
slots.append(-1)
endpoint = Endpoint(
    counter=simpy.Resource(env, capacity=1),
    slots=slots,
    tokens_available={slot: ENDPOINT_SLOT for slot in slots},
    slot_full={slot: env.event() for slot in slots},
    when_slot_full={slot: [] for slot in slots},
    messages_error={slot: 0 for slot in slots},
)

endpoint.tokens_available[-1] = 0

# Start process and run
env.process(message_arrivals(env, endpoint, MESSAGES_ANSWERED))
env.run(until=SIM_TIME)

# Analysis/results
for slot in slots:
    if endpoint.slot_full[slot]:
        full_time = endpoint.when_slot_full[slot]
        num_errors = endpoint.messages_error[slot]
        available_tokens = endpoint.tokens_available[slot]
        print(
            f'The slot "{slot}" was full at {full_time} seconds '
            f'after start message arrives. '
            f'Available tokens {available_tokens}'
        )
        print(f'  Number of message errors: {num_errors}')


# Create a figure and a set of subplots
fig, ax = plt.subplots()

# Iterate over each slot and plot the times
for slot, times in endpoint.when_slot_full.items():
    # Convert slot to integer for plotting
    # Map slot -1 to 6 for plotting
    slot = NUMBER_OF_SLOTS if slot == -1 else slot
    ax.scatter(times, [slot] * len(times), 
               label=f'Slot {slot if slot != NUMBER_OF_SLOTS else str(NUMBER_OF_SLOTS) + " - Error"}')

# Add labels and title
ax.set_xlabel('Time (seconds)')
ax.set_ylabel('Slot')
ax.set_title('Slot Full Times')
ax.legend(loc='lower right')


# Show plot
plt.show()
