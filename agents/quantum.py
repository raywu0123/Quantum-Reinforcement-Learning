from collections import defaultdict

from qiskit import (
    QuantumCircuit,
    execute,
    Aer,
)

import groverIteration as GI
from .base import BaseAgent


class QuantumAgent(BaseAgent):

    def __init__(self, action_space, discount_factor=0.9, alpha=0.8, **kwargs):
        self.memory = defaultdict(tuple)
        self.discount_factor = discount_factor
        self.alpha = alpha

    def get_action(self, state, env):
        circuit = QuantumCircuit(2, 2)
        circuit.h(0)
        circuit.h(1)

        if state in self.memory:
            action, state_value, next_state, reward = self.memory[state]

            if next_state in self.memory:
                next_state_value = self.memory[next_state][1]
            else:
                next_state_value = 0.0

            circuit = groverIteration(circuit, action, reward, next_state_value)

        action = collapse_action_select_method(circuit)
        return action

    def learn(self, state, action, next_state, reward):
        if state in self.memory:
            _, state_value, _, _ = self.memory[state]
        else:
            state_value = 0.0

        if next_state in self.memory:
            _, next_state_value, _, _ = self.memory[next_state]
        else:
            next_state_value = 0.0

        # Update state value
        state_value = state_value + self.alpha * (reward + (self.discount_factor * next_state_value) - state_value)
        self.memory[state] = (action, state_value, next_state, reward)


def collapse_action_select_method(circuit):
    qr = circuit.qubits
    cr = circuit.clbits
    circuit.measure(qr, cr)
    backend = Aer.get_backend('qasm_simulator')
    result = execute(circuit, backend=backend, shots=1).result()

    # convert result to int
    classical_state = int(list(result.get_counts().keys())[0], base=2)
    return classical_state


def groverIteration(eigen_action, action, reward, next_state_value):
    # if L < 2:
    L = int(0.2 * (reward + next_state_value))  # reward + value of the next_state, k is .3 which is arbitrary
    if L > 1:
        L = 1

    qr = eigen_action.qubits
    if action == 0:
        for x in range(L):
            eigen_action, qr = GI.gIteration00(eigen_action, qr)
    elif action == 1:
        for x in range(L):
            eigen_action, qr = GI.gIteration01(eigen_action, qr)
    elif action == 2:
        for x in range(L):
            eigen_action, qr = GI.gIteration10(eigen_action, qr)
    elif action == 3:
        for x in range(L):
            eigen_action, qr = GI.gIteration11(eigen_action, qr)

    return eigen_action
