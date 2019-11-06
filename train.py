from agent import Agent
from functions import *
import sys

stock_name, window_size, episode_count = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])

agent = Agent(window_size)
data = getStockDataVec(stock_name)
data_len = len(data) - 1
batch_size = 32

for e in range(episode_count + 1):
    print("Episode " + str(e) + "/" + str(episode_count))
    state = getState(data, 0, window_size + 1)

    total_profit = 0
    agent.inventory = []

    for time_t in range(data_len):
        action = agent.act(state)

        next_state = getState(data, time_t + 1, window_size + 1)
        reward = 0

        if action == 1:
            agent.inventory.append(data[time_t])
            print("Buy: " + formatPrice(data[time_t]))
        elif action == 2 and len(agent.inventory) > 0:
            bought_price = agent.inventory.pop(0)
            reward = max(data[time_t] - bought_price, 0)
            total_profit += data[time_t] - bought_price
            print("Sell: " + formatPrice(data[time_t]) + " | Profit: " + formatPrice(data[time_t] - bought_price))

        done = True if time_t == data_len - 1 else False
        agent.memory.append((state, action, reward, next_state, done))
        state = next_state

        if done:
            print("--------------------------------")
            print("Total Profit: " + formatPrice(total_profit))
            print("--------------------------------")

        if len(agent.memory) > batch_size:
            agent.replay(batch_size)
    
    if e % 10 == 0:
        agent.model.save("models/model_ep" + str(e))


