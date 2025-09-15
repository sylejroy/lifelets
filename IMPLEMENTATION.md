# Neural Network Evolution Simulation: Implementation Guide

This enhanced instruction set is designed to guide an agentic coder in building a clean, modular simulation of evolving neural networks (NNs). It balances clarity, best practices, and extensibility, and covers both the **baseline MLP** and **advanced topology evolution** approaches.

---

## 1. Baseline Neural Network Structure

### 1.1. Architecture

- **Type:** Multi-Layer Perceptron (MLP)
- **Layers:**
  - **Input Layer:** Sensor values (e.g., distances, energy, random noise)
  - **Hidden Layers:** 1–2 layers, customizable size (e.g., 8 neurons per layer)
  - **Output Layer:** Movement decisions (e.g., dx, dy, velocity vector)
- **Activation Functions:** Use modular, NumPy-based implementations (`tanh`, `sigmoid`, `relu`)
- **Parameter Storage:** All weights (and optionally biases) are stored as flat arrays in the genome for easy mutation/crossover.

### 1.2. Recommended Code Structure

- **Class:** `NeuralNetwork`
  - Methods: `forward(inputs)`, `get_weights()`, `set_weights(weights)`
- **Configurable Parameters:** Number of layers, neurons per layer, activation functions

---

## 2. Genome Encoding and Evolution

### 2.1. Genome Structure

```python
class Genome:
    def __init__(self, weights, sensor_range=10.0, mutation_rate=0.05):
        self.weights = np.array(weights)
        self.sensor_range = sensor_range
        self.mutation_rate = mutation_rate
```

- **Extendable:** Add more traits as needed (e.g., biases, individual mutation rates)

### 2.2. Mutation

- **Function:** `mutate(genome)`
  - Add Gaussian noise to each weight: `weight += np.random.normal(0, sigma)`
  - Mutate biases and other traits similarly
- **Parameter:** Mutation rate can be fixed or evolve as a genome parameter

### 2.3. Crossover (Optional)

- **Function:** `crossover(parent1, parent2)`
  - Combine weights (randomly select each weight from one parent)
- **Note:** Mutation-only evolution is simpler; add crossover for advanced population dynamics

### 2.4. Selection

- **Survival & Reproduction:** Only individuals that survive and reproduce pass on genomes
- **Fitness Evaluation:** Implement clean interfaces for evaluating fitness (e.g., movement efficiency, survival time)

---

## 3. Evolving Network Topology (Optional/Advanced)

### 3.1. Standard Approach: Fixed Topology

- All networks share the same structure; **only weights evolve**

### 3.2. Evolving Topology

#### 3.2.1. NEAT (NeuroEvolution of Augmenting Topologies)

- **Genome:**
  - Encodes nodes and connections as a directed graph
  - Supports mutations:
    - Add/remove connections
    - Add/remove nodes
  - Handles crossover via historical markers (for gene alignment)
- **Library:** Use [`neat-python`](https://github.com/CodeReclaimers/neat-python) for rapid prototyping and reference

#### 3.2.2. Custom Topology Mutation (Lightweight)

- **Genome:**
  - Store a connectivity matrix (boolean mask for active connections)
  - Mutate by toggling connections or adding/removing nodes
  - Initialize new connection weights randomly

---

## 4. Example Genome and Workflow

### 4.1. Baseline Genome

```python
genome = {
    'weights': np.array([...]),  # All NN weights (flattened)
    'sensor_range': 10.0,
    'mutation_rate': 0.05,
}
```

### 4.2. Advanced Genome (With Topology)

```python
genome = {
    'nodes': [...],              # Node IDs/types
    'connections': [...],        # (from, to, weight, enabled)
    'sensor_range': 10.0,
    'mutation_rate': 0.05,
}
```
- **Mutation:** Add/remove connections or nodes as part of genome evolution

---

## 5. Modular Code Design Principles

- **Separation of concerns:** Keep NN code, genome logic, and evolution mechanics in separate modules/classes
- **Config-driven:** Use config files or classes to control simulation parameters
- **Logging:** Integrate logging for genome changes, fitness evaluations, and population stats
- **Testing:** Provide unit tests for NN operations and evolutionary functions
- **Visualization:** Add simple plotting or dashboard code for population performance and topology evolution

---

## 6. Summary Table

| Feature      | Standard MLP            | Evolving Topology (NEAT/Custom)    |
|--------------|------------------------|------------------------------------|
| Structure    | Fixed                  | Evolves: nodes/connections         |
| Genome       | Weights only           | Weights + topology info            |
| Mutation     | Weights                | Weights + structure                |
| Crossover    | Simple                 | Specialized (historical markers)   |
| Complexity   | Simple, fast           | More complex, powerful             |
| Libraries    | NumPy, custom MLP      | neat-python, custom graph-based NN |

---

## 7. Implementation Checklist

- [ ] Implement `NeuralNetwork` class with configurable layers and activation functions
- [ ] Create `Genome` class for storing NN parameters and traits
- [ ] Build mutation and (optional) crossover functions
- [ ] Set up selection and fitness evaluation logic
- [ ] (Optional) Expand to evolving topology using NEAT or custom graph-based approach
- [ ] Structure code for extensibility, maintainability, and testability

---

## 8. Recommendation

- **Start Simple:** Prototype with fixed-topology MLPs and evolving weights; keep code modular
- **Extend Gradually:** Add topology evolution after your simulation works reliably
- **Use Libraries Judiciously:** For NEAT, leverage existing libraries to save time and avoid subtle bugs

---

## 9. Example File Structure

```
neural_evo_sim/
├── neural_network.py        # MLP implementation
├── genome.py                # Genome class and mutation logic
├── evolution.py             # Selection, crossover, population management
├── neat_topology.py         # NEAT or custom topology code (optional)
├── config.py                # Simulation parameters
├── main.py                  # Entry point: simulation loop
├── tests/
│   └── test_neural_network.py
│   └── test_genome.py
├── README.md                # Documentation
```

---

## 10. References

- [NEAT Algorithm Overview](https://www.cs.ucf.edu/~kstanley/neat.html)
- [neat-python library](https://github.com/CodeReclaimers/neat-python)
- [NumPy Documentation](https://numpy.org/doc/)
- [Modular Python Best Practices](https://realpython.com/python-modules-packages/)

---

**Ready to code? Start with a clean baseline MLP, then evolve your simulation as needed!**
