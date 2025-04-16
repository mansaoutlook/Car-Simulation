# AI Self-Driving Car Simulation (Pygame + Neural Networks)

This project is a simulation of self-driving cars using **Neural Networks** and **Genetic Algorithms**, created with Python and Pygame. The cars navigate a track, collect coins (rewards), and learn over multiple generations to drive better using **sensor data**, **collision feedback**, and **evolutionary learning**.

## Features

- 2D car simulation on a race track using Pygame
  
- Neural Network decision-making for AI cars
  
- Coin collection as a reward system (reinforcement learning concept)
 
- Sensor-based environment perception (5 virtual sensors per car)

- Genetic algorithm for evolving better drivers across generations
  
- Save and load the best performing models


## AI Learning Explained

### 1. **Neural Network (Feedforward)**

Each AI car has its own neural network with:

- Inputs: Distance readings from 5 sensors
- Hidden layers: [5, 6, 4] architecture
- Outputs:
  - Turn Left
  - Turn Right
  - Move Forward
  - Move Backward

These outputs are binary decisions controlling the car in each frame.


### 2. **Reinforcement Learning (Concept)**

Although there is **no reward function or policy gradient learning**, this simulation mimics reinforcement learning by:

- Encouraging the car to **collect coins**
- Penalizing inactivity (car is “damaged” if it doesn't collect coins for 5 seconds)
- Using distance from upcoming coin to guide performance


### 3. **Genetic Algorithm**

After every "epoch" (when all cars either crash or reach the finish):

- The car with the **highest points (most coins collected)** is selected
- All other cars are:
  - Cloned from the best car
  - Slightly **mutated** (random changes in weights and biases)
- This forms the next generation, improving driving performance over time

This evolutionary method ensures continuous learning without traditional backpropagation.


