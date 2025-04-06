import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from homework3 import Hw3Env, complex_state
from agent import Agent, SACAgent
import torch
from torchvision.transforms import ToPILImage


def produce_graphs(in_path, out_path, title='title', window=500):
    data = np.load(in_path)
    smooth = np.convolve(data, np.ones(window)/window, mode='valid')
    plt.figure(figsize=(8, 5))
    plt.plot(data, alpha=0.4, label='Original')
    plt.plot(np.arange(len(smooth)) + window//2, smooth, color='red', label='Smoothed')
    plt.legend()
    plt.title(title)
    plt.xlabel('Episode')
    plt.ylabel('Value')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)


def experiment(model_path, out_path, agent_type, period=8):

    device = torch.device('cpu')

    if agent_type == 'pg':
        agent = Agent(device, 0.1, None, 0, None, None, None, None)
        agent.model.load_state_dict(torch.load(model_path, map_location=device))
        agent.model.eval()
        state_manipulator = complex_state
    elif agent_type == 'sac':
        agent = SACAgent(device=device)
        checkpoint = torch.load(model_path, map_location=device)
        agent.policy.load_state_dict(checkpoint['policy_state_dict'])
        agent.policy.eval()
        state_manipulator = lambda x: x
    else:
        print("select pg or sac as the agent type")
        return

    to_pil = ToPILImage()
    env = Hw3Env(render_mode='offscreen')
    env.reset()
    state = env.high_level_state()
    state = torch.from_numpy(state).float()
    state = state_manipulator(state).to(device)
    done = False
    images = [to_pil(env.state())]
    step = 0

    while not done:
        action = agent.decide_action(state)
        next_state, reward, is_terminal, is_truncated = env.step(action[0])
        next_state = torch.from_numpy(next_state).float()
        next_state = state_manipulator(next_state).to(device)
        done = is_terminal or is_truncated
        state = next_state
        step += 1
        if step % period == 0:
            images.append(to_pil(env.state()))
            print(step)

    images[0].save(
        out_path,
        save_all=True,
        append_images=images[1:],  # Rest of the frames
        duration=100,  # Duration between frames in ms
        loop=0  # 0 means loop forever
    )


if __name__ == '__main__':
    for model_type in ['pg', 'sac']:
        for metric_type in ['cumulative_rewards', 'rps', 'success_rate']:
            out_dir = f'metrics/{model_type}'
            os.makedirs(out_dir, exist_ok=True)
            produce_graphs(f'{model_type}/{metric_type}.npy', f'{out_dir}/{metric_type}.jpg', title=f'{metric_type}', window=300)

    experiment('pg/model.pth', 'gifs/pg.gif', 'pg', period=2)
    experiment('sac/model.pth', 'gifs/sac.gif', 'sac', period=2)
