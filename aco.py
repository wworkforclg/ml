#prac1
import numpy as np

X = np.array([[0.2, 0.3]])
Y = np.array([[1]])

W1 = np.array([[0.2, 0.4],
               [0.3, 0.1]])
B1 = np.array([[0.1, 0.2]])
W2 = np.array([[0.2, 0.3],
               [0.4, 0.1]])
B2 = np.array([[0.1, 0.2]])
W3 = np.array([[0.5],
               [0.6]])
B3 = np.array([[0.1]])

lr = 0.1
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

H1 = sigmoid(np.dot(X, W1) + B1)
H2 = sigmoid(np.dot(H1, W2) + B2)
predicted = sigmoid(np.dot(H2, W3) + B3)

if predicted.item() >= 0.5:
    predicted_class = 1
else:
    predicted_class = 0

error = Y - predicted

D3 = error * predicted * (1 - predicted)
D2 = H2 * (1 - H2) * np.dot(D3, W3.T)
D1 = H1 * (1 - H1) * np.dot(D2, W2.T)

W3 = W3 + lr * np.dot(H2.T, D3)
B3 = B3 + lr * D3

W2 = W2 + lr * np.dot(H1.T, D2)
B2 = B2 + lr * D2

W1 = W1 + lr * np.dot(X.T, D1)
B1 = B1 + lr * D1

print("Predicted Probability:", predicted.item())
print("Predicted Class:", predicted_class)
print("Actual Class:", Y.item())
print("Error:", error.item())

print("\nUpdated W1:")
print("W11 =", W1[0,0])
print("W12 =", W1[0,1])
print("W21 =", W1[1,0])
print("W22 =", W1[1,1])

print("\nUpdated W2:")
print("W11 =", W2[0,0])
print("W12 =", W2[0,1])
print("W21 =", W2[1,0])
print("W22 =", W2[1,1])

print("\nUpdated W3:")
print("W11 =", W3[0,0])
print("W21 =", W3[1,0])

import numpy as np

X = np.array([[0.2, 0.3]])
Y = np.array([[0.9]])

W1 = np.array([[0.2, 0.4],
[0.3, 0.1]])
W2 = np.array([[0.2, 0.3],
[0.4, 0.1]])
W3 = np.array([[0.5],
[0.6]])

B1 = np.array([[0.1, 0.2]])
B2 = np.array([[0.1, 0.2]])
B3 = np.array([[0.1]])

lr = 0.1
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


H1 = sigmoid(np.dot(X, W1) + B1)
H2 = sigmoid(np.dot(H1, W2) + B2)
predicted = sigmoid(np.dot(H2, W3) + B3)

error = Y - predicted
accuracy = (1 - abs(error / Y)) * 100

D3 = error * predicted * (1 -predicted)
D2 = H2 * (1 - H2) * np.dot(D3, W3.T)
D1 = H1 * (1 - H1) * np.dot(D2, W2.T)

W3 = W3 + lr * np.dot(H2.T, D3)
W2 = W2 + lr * np.dot(H1.T, D2)
W1 = W1 + lr * np.dot(X.T, D1)

B3 = B3 + lr * D3
B2 = B2 + lr * D2
B1 = B1 + lr * D1

print("Predicted Output:", predicted.item())
print("Target Output:", Y.item())
print("Error:", error.item())
print("Accuracy: %.2f%%" % accuracy.item())

print("\nUpdated W1:")
print("W11 =", W1[0,0])
print("W12 =", W1[0,1])
print("W21 =", W1[1,0])
print("W22 =", W1[1,1])

print("\nUpdated W2:")
print("W11 =", W2[0,0])
print("W12 =", W2[0,1])
print("W21 =", W2[1,0])
print("W22 =", W2[1,1])

print("\nUpdated W3:")
print("W11 =", W3[0,0])
print("W21 =", W3[1,0])

#prac2
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-5, 5, 200)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def elu(x):
    return np.where(x > 0, x, np.exp(x) - 1)

def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def selu(x):
    return 1.0507 * np.where(x > 0, x, 1.6733 * (np.exp(x) - 1))

def silu(x):
    return x * sigmoid(x)

def mish(x):
    return x * np.tanh(np.log(1 + np.exp(x)))

def softplus(x):
    return np.log(1 + np.exp(x))

def softsign(x):
    return x / (1 + np.abs(x))

def hardtanh(x):
    return np.clip(x, -1, 1)

def hardsigmoid(x):
    return np.clip(x / 6 + 0.5, 0, 1)

def hardswish(x):
    return x * hardsigmoid(x)

def prelu(x):
    return np.where(x > 0, x, 0.1 * x)

def threshold(x):
    return np.where(x > 0.5, 1, 0)

def softshrink(x):
    return np.where(x > 0.5, x - 0.5,
                    np.where(x < -0.5, x + 0.5, 0))

activations = [
    ("Linear", lambda x: x),
    ("ReLU", lambda x: np.maximum(0, x)),
    ("Sigmoid", sigmoid),
    ("Tanh", np.tanh),
    ("ELU", elu),
    ("GELU", gelu),
    ("Leaky ReLU", leaky_relu),
    ("RReLU", lambda x: leaky_relu(x, 0.05)),
    ("SELU", selu),
    ("SiLU", silu),
    ("Mish", mish),
    ("Softplus", softplus),
    ("Softsign", softsign),
    ("GLU", silu),
    ("Hardtanh", hardtanh),
    ("Hardsigmoid", hardsigmoid),
    ("Hardswish", hardswish),
    ("PReLU", prelu),
    ("Threshold", threshold),
    ("Softshrink", softshrink)
]

fig, axes = plt.subplots(4, 5, figsize=(15, 10))

for ax, (name, function) in zip(axes.flat, activations):
    ax.plot(x, function(x))
    ax.set_title(name)
    ax.grid()

plt.tight_layout()
plt.show()

#prac3
import pandas as pd
import matplotlib.pyplot as plt
from minisom import MiniSom
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

data = pd.read_csv("data.csv")

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

X = MinMaxScaler().fit_transform(X)

k = len(set(y))

km = KMeans(n_clusters=k, random_state=42, n_init=10)
km_labels = km.fit_predict(X)

print("K-Means WCSS:", km.inertia_)
print("K-Means Silhouette:", silhouette_score(X, km_labels))

som = MiniSom(3, 2, X.shape[1], random_seed=42)
som.random_weights_init(X)
som.train_random(X, 1000)

winners = [som.winner(x) for x in X]
weights = som.get_weights().reshape(-1, X.shape[1])

km2 = KMeans(n_clusters=k, random_state=42, n_init=10)
labels = km2.fit_predict(weights)

som_labels = [labels[x * 2 + y] for x, y in winners]

print("\nSOM Quantization Error:", som.quantization_error(X))
print("SOM Silhouette:", silhouette_score(X, som_labels))

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=km_labels, cmap="viridis")
plt.title("K-Means")

plt.subplot(1, 2, 2)
plt.scatter(X[:, 0], X[:, 1], c=som_labels, cmap="viridis")
plt.title("SOM")

plt.show()

import pandas as pd
import matplotlib.pyplot as plt
from minisom import MiniSom
from sklearn.preprocessing import MinMaxScaler

data = pd.read_csv("data.csv")

print("Dataset Shape:", data.shape)
print(data.head())

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

X = MinMaxScaler().fit_transform(X)

som = MiniSom(3, 2, X.shape[1], sigma=1.0,
              learning_rate=0.5, random_seed=42)

som.random_weights_init(X)
som.train_random(X, 1000)

print("\nSOM Training Completed.")
print("Quantization Error:", som.quantization_error(X))

plt.figure(figsize=(8, 6))
plt.pcolor(som.distance_map().T)
plt.colorbar(label="Distance")
plt.title("SOM U-Matrix")
plt.show()

plt.figure(figsize=(8, 6))
plt.pcolor(som.distance_map().T)
plt.colorbar(label="Distance")

markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'x']
classes = list(set(y))

for i in range(len(X)):
    winner = som.winner(X[i])
    index = classes.index(y[i])

    plt.plot(winner[0] + 0.5, winner[1] + 0.5,
             marker=markers[index % len(markers)],
             markerfacecolor="None",
             markeredgecolor=plt.cm.tab10(index),
             markersize=10)

plt.title("SOM Clustering")
plt.xlabel("SOM X")
plt.ylabel("SOM Y")
plt.show()

#prac4
import numpy as np
import gymnasium as gym

alpha = 0.3
gamma = 0.9
epsilon = 1.0
epsilon_decay = 0.999
min_epsilon = 0.01
episodes = 10000
max_steps = 100
tests = 1000

custom_map = [
    "SFFF",
    "FHFH",
    "FFFH",
    "HFFG"
]

for slippery in [False, True]:
    env = gym.make("FrozenLake-v1", desc=custom_map, is_slippery=slippery)
    Q = np.zeros((env.observation_space.n, env.action_space.n))
    
    rewards = []
    for episode in range(episodes):
        state, _ = env.reset()
        episode_reward = 0

        for step in range(max_steps):
            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[state])

            next_state, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward

            if terminated or truncated:
                target = reward
            else:
                target = reward + gamma * np.max(Q[next_state])

            Q[state][action] += alpha * (target - Q[state][action])

            state = next_state
            if terminated or truncated:
                break

        rewards.append(episode_reward)
        epsilon = max(min_epsilon, epsilon * epsilon_decay)

    env.close()

    test_env = gym.make("FrozenLake-v1", desc=custom_map, is_slippery=slippery)
    wins = 0
    total_reward = 0
    for i in range(tests):
        state, _ = test_env.reset()
        reward_sum = 0
        for step in range(max_steps):
            action = np.argmax(Q[state])
            state, reward, terminated, truncated, _ = test_env.step(action)
            reward_sum += reward

            if terminated or truncated:
                break

        total_reward += reward_sum

        if reward_sum == 1:
            wins += 1

    test_env.close()

    print("\nSlippery:", slippery)
    print("Training Performance:", np.mean(rewards))
    print("Testing Average Reward:", total_reward / tests)
    print("Win Rate:", (wins / tests) * 100, "%")
    print("Q Table:\n", Q)

import numpy as np
import gymnasium as gym
from tabulate import tabulate

levels = [
    ["Easy", ["SFFF", "FFFF", "FFFF", "FFFG"], 0.7, 0.003, 5000],
    ["Medium", ["SFFF", "FHFH", "FFFF", "FFFG"], 0.7, 0.003, 7000],
    ["Hard", ["SFFH", "FHFH", "FFFF", "HFFG"], 0.7, 0.002, 9000],
    ["Expert", ["SFFH", "HFFF", "FHFH", "HFFG"], 0.7, 0.001, 10000]
]

gamma = 0.99
max_steps = 100
tests = 1000
results = []

for name, custom_map, alpha, decay, episodes in levels:

    env = gym.make("FrozenLake-v1", desc=custom_map, is_slippery=False)
    Q = np.zeros((env.observation_space.n, env.action_space.n))

    epsilon = 1.0
    rewards = []

    for episode in range(episodes):
        state, _ = env.reset()
        episode_reward = 0

        for step in range(max_steps):

            if np.random.random() < epsilon or np.max(Q[state]) == 0:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[state])

            next_state, reward, terminated, truncated, _ = env.step(action)

            if terminated or truncated:
                target = reward
            else:
                target = reward + gamma * np.max(Q[next_state])

            Q[state][action] += alpha * (target - Q[state][action])

            episode_reward += reward

            if terminated or truncated:
                break

            state = next_state

        rewards.append(episode_reward)
        epsilon = max(0.01, epsilon * np.exp(-decay))

    env.close()

    test_env = gym.make("FrozenLake-v1", desc=custom_map, is_slippery=False)

    wins = 0
    total_reward = 0
    total_steps = 0

    for i in range(tests):
        state, _ = test_env.reset()
        reward_sum = 0

        for step in range(max_steps):
            action = np.argmax(Q[state])
            state, reward, terminated, truncated, _ = test_env.step(action)

            reward_sum += reward
            total_steps += step + 1

            if terminated or truncated:
                break

        total_reward += reward_sum

        if reward_sum == 1:
            wins += 1

    test_env.close()

    avg_reward = total_reward / tests
    win_rate = (wins / tests) * 100
    avg_steps = total_steps / tests

    results.append([
        name, alpha, decay, episodes,
        round(avg_reward, 3),
        round(win_rate, 2),
        round(avg_steps, 2)
    ])

    print("\nLevel:", name)
    print("Training Performance:", np.mean(rewards))
    print("Testing Average Reward:", avg_reward)
    print("Win Rate:", win_rate, "%")
    print("Average Steps:", avg_steps)
    print("Q Table:\n", Q)

print("\nFinal Results:")
print(tabulate(
    results,
    headers=["Level", "Alpha", "Decay", "Episodes", "Avg Reward", "Win Rate", "Avg Steps"],
    tablefmt="grid"
))

#prac5
import numpy as np
import gymnasium as gym

alpha = 0.3
gamma = 0.9
epsilon = 1.0
epsilon_decay = 0.99
episodes = 10000
max_steps = 100
tests = 1000

custom_map = [
    "SFFF",
    "FHFH",
    "FFFH",
    "HFFG"
]

for slippery in [False, True]:

    env = gym.make("FrozenLake-v1", desc=custom_map, is_slippery=slippery)

    Q = np.zeros((env.observation_space.n, env.action_space.n))
    epsilon = 1.0
    rewards = []

    for episode in range(episodes):

        state, _ = env.reset()
        episode_reward = 0

        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state])

        for step in range(max_steps):

            next_state, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward

            if np.random.random() < epsilon:
                next_action = env.action_space.sample()
            else:
                next_action = np.argmax(Q[next_state])

            if terminated or truncated:
                target = reward
            else:
                target = reward + gamma * Q[next_state][next_action]

            Q[state][action] += alpha * (target - Q[state][action])

            if terminated or truncated:
                break

            state = next_state
            action = next_action

        rewards.append(episode_reward)
        epsilon = max(0.01, epsilon * epsilon_decay)

    env.close()

    test_env = gym.make("FrozenLake-v1", desc=custom_map, is_slippery=slippery)

    wins = 0
    total_reward = 0

    for i in range(tests):

        state, _ = test_env.reset()
        done = False

        while not done:

            action = np.argmax(Q[state])
            state, reward, terminated, truncated, _ = test_env.step(action)

            total_reward += reward

            if terminated or truncated:
                done = True

                if reward == 1:
                    wins += 1

    test_env.close()

    print("\nSlippery:", slippery)
    print("Training Performance:", np.mean(rewards))
    print("Testing Average Reward:", total_reward / tests)
    print("Win Rate:", (wins / tests) * 100, "%")
    print("Q Table:\n", Q)

#prac6
import numpy as np
import gymnasium as gym
import tensorflow as tf
import random
from collections import deque
from tensorflow.keras import layers

episodes = 1000
gamma = 0.95
epsilon = 1.0
epsilon_decay = 0.995
learning_rate = 0.001
batch_size = 32
max_steps = 100

def one_hot(state):
    encoded = np.zeros(16)
    encoded[state] = 1
    return encoded

def build_model():
    model = tf.keras.Sequential([
        layers.Input(shape=(16,)),
        layers.Dense(24, activation="relu"),
        layers.Dense(24, activation="relu"),
        layers.Dense(4, activation="linear")
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),
                  loss="mse")
    return model

def train_dqn(env):
    global epsilon

    model = build_model()
    target = build_model()
    target.set_weights(model.get_weights())

    memory = deque(maxlen=2000)
    rewards = []

    for episode in range(episodes):
        state, _ = env.reset()
        total = 0

        for step in range(max_steps):
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                q = model.predict(one_hot(state).reshape(1, 16), verbose=0)
                action = np.argmax(q[0])

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            if terminated and reward == 0:
                reward = -0.2

            memory.append((state, action, reward, next_state, done))

            if len(memory) >= batch_size:
                batch = random.sample(memory, batch_size)

                states = np.array([one_hot(x[0]) for x in batch])
                next_states = np.array([one_hot(x[3]) for x in batch])

                actions = [x[1] for x in batch]
                rewards_batch = [x[2] for x in batch]
                dones = [x[4] for x in batch]

                current_q = model.predict(states, verbose=0)
                next_q = target.predict(next_states, verbose=0)

                for i in range(batch_size):
                    if dones[i]:
                        current_q[i][actions[i]] = rewards_batch[i]
                    else:
                        current_q[i][actions[i]] = (
                            rewards_batch[i] + gamma * max(next_q[i])
                        )

                model.train_on_batch(states, current_q)

            state = next_state
            total += reward

            if done:
                break

        epsilon = max(0.01, epsilon * epsilon_decay)

        if episode % 10 == 0:
            target.set_weights(model.get_weights())

        rewards.append(total)

    return model, np.mean(rewards)

for slippery in [False, True]:
    epsilon = 1.0

    env = gym.make("FrozenLake-v1", is_slippery=slippery)
    model, performance = train_dqn(env)

    print("\nSlippery:", slippery)
    print("Average Reward:", performance)
    print("Training Performance:", performance * 100, "%")

    print("\nFinal Q-Values:")
    for state in range(16):
        q = model.predict(one_hot(state).reshape(1, 16), verbose=0)
        print("State", state, ":", np.round(q[0], 4))

    env.close()

#prac7
import math

def rbf(x, center, gamma):
    distance = sum((a-b) ** 2 for a, b in zip(x,center))
    return math.exp(-gamma * distance)

gamma = 2.0
bias = 0.3

centers = [
    [0,0],
    [0,1],
    [1,0],
    [1,1]
]

weights = [-0.8, 0.9, 0.9, -0.8]

inputs = [
    [0,0],
    [0,1],
    [1,0],
    [1,1]
]

for x in inputs:
    activations = []
    for center in centers:
        activation = rbf(x, center, gamma)
        activations.append(activation)
    
    output = sum( w * a for a, w in zip(weights, activations)) + bias

    if output < 0:
        classification = 0
    else:
        classification = 1
    
    print("Input:", x)
    print("Activations:", activations)
    print("Output:", output)
    print("Classification:", classification)
    print()

#prac8
import random

target = "FYBSC"
population_size = 20
mutation_rate = 0.15


def create_population():
    population = []

    for i in range(population_size):
        chromosome = list(target)
        random.shuffle(chromosome)
        population.append("".join(chromosome))

    return population


def fitness(chromosome):
    score = 0
    for i in range(len(target)):
        if chromosome[i] == target[i]:
            score += 1
    return score


def roulette_selection(population):
    scores = []

    for chromosome in population:
        scores.append(fitness(chromosome))

    return random.choices(population, weights=scores, k=1)[0]


def crossover(parent1, parent2):
    point = random.randint(1, len(target) - 1)
    child = parent1[:point]

    for letter in parent2:
        if letter not in child:
            child += letter

    return child


def mutation(chromosome):
    chromosome = list(chromosome)

    pos1, pos2 = random.sample(range(len(chromosome)), 2)

    chromosome[pos1], chromosome[pos2] = chromosome[pos2], chromosome[pos1]

    return "".join(chromosome)


population = create_population()
generation = 0

while True:
    best = max(population, key=fitness)

    if best == target:
        print("Target Found:", best)
        print("Generation:", generation)
        break

    new_population = []

    while len(new_population) < population_size:
        parent1 = roulette_selection(population)
        parent2 = roulette_selection(population)

        child = crossover(parent1, parent2)

        if random.random() < mutation_rate:
            child = mutation(child)

        new_population.append(child)

    population = new_population
    generation += 1

import random
import matplotlib.pyplot as plt

target = "HELLO"
population_size = 20
mutation_rate = 0.15
number_of_runs = 50


def create_population():
    population = []

    for i in range(population_size):
        chromosome = list(target)
        random.shuffle(chromosome)
        population.append("".join(chromosome))

    return population


def fitness(chromosome):
    score = 0

    for i in range(len(target)):
        if chromosome[i] == target[i]:
            score += 1

    return score


def roulette_selection(population):
    return random.choices(
        population,
        weights=[fitness(c) + 1 for c in population],
        k=1
    )[0]


def rank_selection(population):
    ranked = sorted(population, key=fitness)
    return random.choices(ranked, weights=range(1, len(ranked) + 1), k=1)[0]


def tournament_selection(population):
    return max(random.sample(population, 3), key=fitness)


def crossover(p1, p2):
    point = random.randint(1, len(target) - 1)
    child = list(p1[:point])
    remaining = list(p2)

    for letter in child:
        remaining.remove(letter)

    return "".join(child + remaining)


def mutation(chromosome):
    chromosome = list(chromosome)
    a, b = random.sample(range(len(chromosome)), 2)
    chromosome[a], chromosome[b] = chromosome[b], chromosome[a]
    return "".join(chromosome)


def run_genetic_algorithm(method):
    population = create_population()
    generation = 0

    while True:
        if max(population, key=fitness) == target:
            return generation

        new_population = []

        while len(new_population) < population_size:

            if method == "roulette":
                p1 = roulette_selection(population)
                p2 = roulette_selection(population)
            elif method == "rank":
                p1 = rank_selection(population)
                p2 = rank_selection(population)
            else:
                p1 = tournament_selection(population)
                p2 = tournament_selection(population)

            child = crossover(p1, p2)

            if random.random() < mutation_rate:
                child = mutation(child)

            new_population.append(child)

        population = new_population
        generation += 1


methods = ["roulette", "rank", "tournament"]
averages = []

for method in methods:
    results = []

    for i in range(number_of_runs):
        results.append(run_genetic_algorithm(method))

    average = sum(results) / number_of_runs
    print(method.capitalize(), "Average:", round(average, 2),
          "Best:", min(results), "Worst:", max(results))

    averages.append(average)


plt.bar(["Roulette", "Rank", "Tournament"], averages)
plt.title("Genetic Algorithm Selection Comparison")
plt.xlabel("Selection Method")
plt.ylabel("Average Generations")
plt.show()

#prac9
import random
import matplotlib.pyplot as plt

target = "FYBSC"
population_size = 20
mutation_rate = 0.15
number_of_runs = 50


def create_population():
    population = []

    for i in range(population_size):
        chromosome = list(target)
        random.shuffle(chromosome)
        population.append("".join(chromosome))

    return population


def fitness(chromosome):
    distance = 0

    for i in range(len(target)):
        distance += abs(i - target.index(chromosome[i]))

    return 100 - distance


def roulette_selection(population):
    return random.choices(
        population,
        weights=[fitness(c) for c in population],
        k=1
    )[0]


def rank_selection(population):
    ranked = sorted(population, key=fitness)
    return random.choices(ranked, weights=range(1, len(ranked) + 1), k=1)[0]


def tournament_selection(population):
    return max(random.sample(population, 3), key=fitness)


def crossover(p1, p2):
    point = random.randint(1, len(target) - 1)
    child = p1[:point]

    for letter in p2:
        if letter not in child:
            child += letter

    return child


def mutation(chromosome):
    chromosome = list(chromosome)
    a, b = random.sample(range(len(chromosome)), 2)
    chromosome[a], chromosome[b] = chromosome[b], chromosome[a]
    return "".join(chromosome)


def run_genetic_algorithm(method):
    population = create_population()
    generation = 0

    while True:
        if max(population, key=fitness) == target:
            return generation

        new_population = []

        while len(new_population) < population_size:

            if method == "roulette":
                p1 = roulette_selection(population)
                p2 = roulette_selection(population)

            elif method == "rank":
                p1 = rank_selection(population)
                p2 = rank_selection(population)

            else:
                p1 = tournament_selection(population)
                p2 = tournament_selection(population)

            child = crossover(p1, p2)

            if random.random() < mutation_rate:
                child = mutation(child)

            new_population.append(child)

        population = new_population
        generation += 1


methods = ["roulette", "rank", "tournament"]
averages = []

for method in methods:
    results = []

    for i in range(number_of_runs):
        results.append(run_genetic_algorithm(method))

    average = sum(results) / number_of_runs

    print(method.capitalize(), "Average:", round(average, 2),
          "Best:", min(results), "Worst:", max(results))

    averages.append(average)


plt.bar(["Roulette", "Rank", "Tournament"], averages)
plt.title("Genetic Algorithm Selection Comparison")
plt.xlabel("Selection Method")
plt.ylabel("Average Generations")
plt.show()

import random
import matplotlib.pyplot as plt

target = "HELLO"
population_size = 20
mutation_rate = 0.15
number_of_runs = 50


def create_population():
    population = []

    for i in range(population_size):
        chromosome = list(target)
        random.shuffle(chromosome)
        population.append("".join(chromosome))

    return population


def fitness(chromosome):
    score = 0

    for i in range(len(target)):
        if chromosome[i] == target[i]:
            score += 1

    return score


def roulette_selection(population):
    return random.choices(
        population,
        weights=[fitness(c) + 1 for c in population],
        k=1
    )[0]


def rank_selection(population):
    ranked = sorted(population, key=fitness)
    return random.choices(ranked, weights=range(1, len(ranked) + 1), k=1)[0]


def tournament_selection(population):
    return max(random.sample(population, 3), key=fitness)


def crossover(p1, p2):
    point = random.randint(1, len(target) - 1)
    child = list(p1[:point])
    remaining = list(p2)

    for letter in child:
        remaining.remove(letter)

    return "".join(child + remaining)


def mutation(chromosome):
    chromosome = list(chromosome)
    a, b = random.sample(range(len(chromosome)), 2)
    chromosome[a], chromosome[b] = chromosome[b], chromosome[a]
    return "".join(chromosome)


def run_genetic_algorithm(method):
    population = create_population()
    generation = 0

    while True:
        if max(population, key=fitness) == target:
            return generation

        new_population = []

        while len(new_population) < population_size:

            if method == "roulette":
                p1 = roulette_selection(population)
                p2 = roulette_selection(population)
            elif method == "rank":
                p1 = rank_selection(population)
                p2 = rank_selection(population)
            else:
                p1 = tournament_selection(population)
                p2 = tournament_selection(population)

            child = crossover(p1, p2)

            if random.random() < mutation_rate:
                child = mutation(child)

            new_population.append(child)

        population = new_population
        generation += 1


methods = ["roulette", "rank", "tournament"]
averages = []

for method in methods:
    results = []

    for i in range(number_of_runs):
        results.append(run_genetic_algorithm(method))

    average = sum(results) / number_of_runs
    print(method.capitalize(), "Average:", round(average, 2),
          "Best:", min(results), "Worst:", max(results))

    averages.append(average)


plt.bar(["Roulette", "Rank", "Tournament"], averages)
plt.title("Genetic Algorithm Selection Comparison")
plt.xlabel("Selection Method")
plt.ylabel("Average Generations")
plt.show()

import random

cities = [
    "Delhi", "Mumbai", "Bengaluru", "Chennai", "Hyderabad",
    "Kolkata", "Pune", "Ahmedabad", "Jaipur", "Lucknow"
]

distance = {
    "Delhi": {"Mumbai": 1400, "Bengaluru": 2150, "Chennai": 2200, "Hyderabad": 1550, "Kolkata": 1500, "Pune": 1450, "Ahmedabad": 950, "Jaipur": 270, "Lucknow": 550},
    "Mumbai": {"Bengaluru": 980, "Chennai": 1330, "Hyderabad": 710, "Kolkata": 2050, "Pune": 150, "Ahmedabad": 530, "Jaipur": 1150, "Lucknow": 1400},
    "Bengaluru": {"Chennai": 350, "Hyderabad": 570, "Kolkata": 1870, "Pune": 840, "Ahmedabad": 1490, "Jaipur": 2000, "Lucknow": 1900},
    "Chennai": {"Hyderabad": 630, "Kolkata": 1660, "Pune": 1200, "Ahmedabad": 1830, "Jaipur": 2050, "Lucknow": 1950},
    "Hyderabad": {"Kolkata": 1180, "Pune": 560, "Ahmedabad": 1430, "Jaipur": 1750, "Lucknow": 1250},
    "Kolkata": {"Pune": 1900, "Ahmedabad": 2050, "Jaipur": 1500, "Lucknow": 1000},
    "Pune": {"Ahmedabad": 660, "Jaipur": 1050, "Lucknow": 1400},
    "Ahmedabad": {"Jaipur": 670, "Lucknow": 1200},
    "Jaipur": {"Lucknow": 570}
}

population_size = 50
mutation_rate = 0.15
tournament_size = 3
number_of_generations = 100


def get_distance(city1, city2):
    if city1 == city2:
        return 0
    if city2 in distance[city1]:
        return distance[city1][city2]
    return distance[city2][city1]


def calculate_route_distance(route):
    total = 0

    for i in range(len(route)):
        total += get_distance(route[i], route[(i + 1) % len(route)])

    return total


def create_population():
    population = []

    for i in range(population_size):
        route = cities.copy()
        random.shuffle(route)
        population.append(route)

    return population


def tournament_selection(population):
    contestants = random.sample(population, tournament_size)
    return min(contestants, key=calculate_route_distance).copy()


def crossover(p1, p2):
    a, b = sorted(random.sample(range(len(p1)), 2))

    child = [None] * len(p1)
    child[a:b] = p1[a:b]

    position = b

    for city in p2:
        if city not in child:
            if position == len(p1):
                position = 0

            child[position] = city
            position += 1

    return child


def mutation(route):
    route = route.copy()
    a, b = random.sample(range(len(route)), 2)
    route[a], route[b] = route[b], route[a]
    return route


population = create_population()
best_route = None
best_distance = float("inf")

for generation in range(1, number_of_generations + 1):

    current_best = min(population, key=calculate_route_distance)
    current_distance = calculate_route_distance(current_best)

    if current_distance < best_distance:
        best_distance = current_distance
        best_route = current_best.copy()
        print("Generation", generation, "New Best =", best_distance, "km")

    new_population = []

    while len(new_population) < population_size:
        p1 = tournament_selection(population)
        p2 = tournament_selection(population)

        child = crossover(p1, p2)

        if random.random() < mutation_rate:
            child = mutation(child)

        new_population.append(child)

    population = new_population


print("\nFINAL RESULT")
print("Best Route:")
print(" -> ".join(best_route + [best_route[0]]))
print("Total Distance:", best_distance, "km")

import random

weights = [2, 3, 4, 5, 1]
values = [10, 20, 30, 40, 5]

capacity = 8
population_size = 50
mutation_rate = 0.15
tournament_size = 3
generations = 100


def create_population():
    return [[random.randint(0, 1) for i in weights]
            for _ in range(population_size)]


def fitness(chromosome):
    total_weight = sum(weights[i] for i in range(5) if chromosome[i] == 1)
    total_value = sum(values[i] for i in range(5) if chromosome[i] == 1)

    if total_weight > capacity:
        return 0
    return total_value


def tournament_selection(population):
    contestants = random.sample(population, tournament_size)
    return max(contestants, key=fitness)


def crossover(p1, p2):
    point = random.randint(1, 4)
    return p1[:point] + p2[point:]


def mutation(chromosome):
    chromosome = chromosome.copy()
    i = random.randint(0, 4)
    chromosome[i] = 1 - chromosome[i]
    return chromosome


population = create_population()
best_chromosome = None
best_fitness = 0

for generation in range(generations):
    new_population = []

    for _ in range(population_size):
        p1 = tournament_selection(population)
        p2 = tournament_selection(population)

        child = crossover(p1, p2)

        if random.random() < mutation_rate:
            child = mutation(child)

        new_population.append(child)

    population = new_population
    current_best = max(population, key=fitness)

    if fitness(current_best) > best_fitness:
        best_fitness = fitness(current_best)
        best_chromosome = current_best.copy()


total_weight = sum(weights[i] for i in range(5) if best_chromosome[i] == 1)

print("Best Chromosome:", best_chromosome)
print("Total Weight:", total_weight)
print("Maximum Value:", best_fitness)

import random

jobs = [2, 4, 6, 8, 3]

population_size = 50
mutation_rate = 0.15
tournament_size = 3
generations = 100


def create_population():
    return [[random.randint(0, 1) for i in jobs]
            for _ in range(population_size)]


def fitness(chromosome):
    machine1 = sum(jobs[i] for i in range(5) if chromosome[i] == 0)
    machine2 = sum(jobs[i] for i in range(5) if chromosome[i] == 1)
    return 1 / max(machine1, machine2)


def tournament_selection(population):
    contestants = random.sample(population, tournament_size)
    return max(contestants, key=fitness)


def crossover(p1, p2):
    point = random.randint(1, 4)
    return p1[:point] + p2[point:]


def mutation(chromosome):
    chromosome = chromosome.copy()
    i = random.randint(0, 4)
    chromosome[i] = 1 - chromosome[i]
    return chromosome


population = create_population()

for generation in range(generations):
    best = max(population, key=fitness)
    new_population = [best]

    while len(new_population) < population_size:
        p1 = tournament_selection(population)
        p2 = tournament_selection(population)

        child = crossover(p1, p2)

        if random.random() < mutation_rate:
            child = mutation(child)

        new_population.append(child)

    population = new_population


best = max(population, key=fitness)

machine1 = [jobs[i] for i in range(5) if best[i] == 0]
machine2 = [jobs[i] for i in range(5) if best[i] == 1]

print("Best Assignment:", best)
print("Machine 1 Jobs:", machine1)
print("Machine 2 Jobs:", machine2)
print("Machine 1 Load:", sum(machine1))
print("Machine 2 Load:", sum(machine2))
print("Minimum Makespan:", max(sum(machine1), sum(machine2)))

import random

population_size = 20
mutation_rate = 0.15
generations = 100

weights = [0.3, 0.2, 0.2, 0.1, 0.2]

# [Python, Java, SQL, Communication, Experience]

def create_population():
    return [[random.randint(1, 10) for i in range(5)]
            for _ in range(population_size)]


def fitness(candidate):
    return sum(candidate[i] * weights[i] for i in range(5))


def tournament_selection(population):
    contestants = random.sample(population, 3)
    return max(contestants, key=fitness)


def crossover(p1, p2):
    point = random.randint(1, 4)
    return p1[:point] + p2[point:]


def mutation(candidate):
    candidate = candidate.copy()
    i = random.randint(0, 4)
    candidate[i] = random.randint(1, 10)
    return candidate


population = create_population()

for generation in range(generations):
    best = max(population, key=fitness)
    new_population = [best]

    while len(new_population) < population_size:
        p1 = tournament_selection(population)
        p2 = tournament_selection(population)

        child = crossover(p1, p2)

        if random.random() < mutation_rate:
            child = mutation(child)

        new_population.append(child)

    population = new_population


best_candidate = max(population, key=fitness)

print("Best Candidate:", best_candidate)
print("Fitness Score:", round(fitness(best_candidate), 2))

#prac11
import random
import math

target = [7.5, 7.5]

pos = [[random.uniform(0, 10), random.uniform(0, 10)] for i in range(10)]
vel = [[0, 0] for i in range(10)]
p_best = [p.copy() for p in pos]

for iteration in range(50):

    g_best = min(p_best, key=lambda p: math.dist(p, target))

    print("Iteration:", iteration + 1,
          "Best Distance:", round(math.dist(g_best, target), 4))

    if math.dist(g_best, target) < 1e-4:
        print("Food successfully found!")
        break

    for i in range(10):
        for d in range(2):
            vel[i][d] = (0.5 * vel[i][d]
                         + random.random() * (p_best[i][d] - pos[i][d])
                         + random.random() * (g_best[d] - pos[i][d]))

            pos[i][d] += vel[i][d]

        if math.dist(pos[i], target) < math.dist(p_best[i], target):
            p_best[i] = pos[i].copy()

g_best = min(p_best, key=lambda p: math.dist(p, target))

print("\nFinal Result")
print("Target:", target)
print("Best Position:", g_best)
print("Distance:", math.dist(g_best, target))

import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

data, _ = make_blobs(n_samples=150, centers=3, random_state=42)

k = 3
particles = 10
iterations = 50

def fitness(centers):
    error = 0
    for point in data:
        distances = np.linalg.norm(point - centers, axis=1)
        error += min(distances) ** 2
    return error

km = KMeans(n_clusters=k, random_state=42, n_init=10)
km.fit(data)

print("K-Means WCSS:", km.inertia_)
print("K-Means Iterations:", km.n_iter_)

low = data.min()
high = data.max()

pos = [np.random.uniform(low, high, (k, 2)) for i in range(particles)]
vel = [np.zeros((k, 2)) for i in range(particles)]

pbest = [p.copy() for p in pos]
pbest_error = [fitness(p) for p in pos]

best = np.argmin(pbest_error)
gbest = pbest[best].copy()
gbest_error = pbest_error[best]

for iteration in range(iterations):
    for i in range(particles):
        r1 = np.random.rand(k, 2)
        r2 = np.random.rand(k, 2)

        vel[i] = (0.5 * vel[i]
                  + 1.5 * r1 * (pbest[i] - pos[i])
                  + 1.5 * r2 * (gbest - pos[i]))

        pos[i] += vel[i]
        error = fitness(pos[i])

        if error < pbest_error[i]:
            pbest[i] = pos[i].copy()
            pbest_error[i] = error

            if error < gbest_error:
                gbest = pos[i].copy()
                gbest_error = error

print("\nPSO WCSS:", gbest_error)
print("PSO Iterations:", iterations)

print("\nComparison:")
if gbest_error < km.inertia_:
    print("PSO performed better")
else:
    print("K-Means performed better")

#prac12
import random

vertices = ['A', 'B', 'C', 'D']

graph = {
    'A': ['B', 'C', 'D'],
    'B': ['A', 'C', 'D'],
    'C': ['A', 'B', 'D'],
    'D': ['A', 'B', 'C']
}

weights = {
    'AB': 10, 'AC': 10, 'AD': 30,
    'BC': 40, 'CD': 20, 'BD': 10
}

pheromone = {}
p = 0.1

def get_weight(u, v):
    return weights.get(u + v, weights.get(v + u, 0))

def get_pheromone(u, v):
    return pheromone.get(u + v, pheromone.get(v + u, 0))

def set_pheromone(u, v, value):
    key = u + v if u + v in pheromone else v + u
    pheromone[key] = round(value, 3)


for u in vertices:
    for v in graph[u]:
        if u + v not in pheromone and v + u not in pheromone:
            pheromone[u + v] = random.randint(1, 50)


for iteration in range(5):
    paths = []

    for ant in range(4):
        route = ['A']
        current = 'A'

        while len(route) < 4:
            choices = [v for v in graph[current] if v not in route]

            probabilities = []

            for v in choices:
                probabilities.append(
                    get_pheromone(current, v) / get_weight(current, v)
                )

            next_city = random.choices(choices, weights=probabilities, k=1)[0]

            route.append(next_city)
            current = next_city

        route.append('A')
        paths.append(route)

        cost = sum(get_weight(route[i], route[i+1]) for i in range(4))
        print("Ant", ant + 1, "Tour:", route, "Distance:", cost)

    for key in pheromone:
        pheromone[key] *= (1 - p)

    for route in paths:
        cost = sum(get_weight(route[i], route[i+1]) for i in range(4))
        deposit = 1 / cost

        for i in range(4):
            u, v = route[i], route[i+1]
            set_pheromone(u, v, get_pheromone(u, v) + deposit)

    print("Pheromone:", pheromone)

print("\nFinal Pheromones:", pheromone)

from random import randrange

vertices = ['A', 'B', 'C', 'D']

graph = {
    'A': ['B', 'C', 'D'],
    'B': ['A', 'C', 'D'],
    'C': ['A', 'B', 'D'],
    'D': ['A', 'B', 'C']
}

weights = {
    'AB': 10, 'AC': 10, 'AD': 30,
    'BC': 40, 'CD': 20, 'BD': 10
}

pheromone = {}
p = 0.1

def get_weight(u, v):
    return weights.get(u + v, weights.get(v + u, 0))

def get_pheromone(u, v):
    return pheromone.get(u + v, pheromone.get(v + u, 0))

def set_pheromone(u, v, value):
    key = u + v if u + v in pheromone else v + u
    pheromone[key] = round(value, 3)

for u in vertices:
    for v in graph[u]:
        if u + v not in pheromone and v + u not in pheromone:
            pheromone[u + v] = randrange(1, 51)

for iteration in range(5):
    paths = []
    used = []

    for ant in range(4):
        start = randrange(4)
        while start in used:
            start = randrange(4)

        used.append(start)
        start = vertices[start]

        route = [start]
        current = start

        while len(route) < 4:
            choices = [v for v in graph[current] if v not in route]

            probabilities = []
            for v in choices:
                probabilities.append(get_pheromone(current, v) / get_weight(current, v))

            next_city = choices[probabilities.index(max(probabilities))]
            route.append(next_city)
            current = next_city

        route.append(start)
        paths.append(route)

        cost = sum(get_weight(route[i], route[i+1]) for i in range(4))
        print("Tour:", route, "Distance:", cost)

    for key in pheromone:
        pheromone[key] *= (1 - p)

    for route in paths:
        cost = sum(get_weight(route[i], route[i+1]) for i in range(4))
        deposit = 1 / cost

        for i in range(4):
            u, v = route[i], route[i+1]
            set_pheromone(u, v, get_pheromone(u, v) + deposit)

    print("Pheromone:", pheromone)

print("\nFinal Pheromones:", pheromone)
