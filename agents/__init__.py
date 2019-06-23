from .quantum import QuantumAgent
from .traditional_q_learning import TraditionalQLearningAgent


AgentHub = {
    'quantum': QuantumAgent,
    'tradition-q': TraditionalQLearningAgent,
}