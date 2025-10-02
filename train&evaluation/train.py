from pathlib import Path
import matplotlib.pyplot as plt
from collections import defaultdict

import torch

from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.datasets.factory import resolve_delta_timestamps

def main():
    output_directory = Path("/home/dam/kkm_ws/train/towel_flattening_4_model")
    output_directory.mkdir(parents=True, exist_ok=True)

    dataset_directory = Path("/home/dam/colcon_ws/src/ros2_lerobot/demo_data/towel_flattening_4")

    device = torch.device("cuda")

    training_steps = 100000
    log_freq = 1

    dataset_metadata = LeRobotDatasetMetadata("OMY", root=dataset_directory)
    
    features = dataset_to_policy_features(dataset_metadata.features)
    output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {key: ft for key, ft in features.items() if key not in output_features}

    cfg = ACTConfig(input_features=input_features, output_features=output_features, chunk_size=100, n_action_steps=100)

    delta_timestamps = resolve_delta_timestamps(cfg, dataset_metadata)
    print(delta_timestamps)
    print(len(delta_timestamps['action']))

    policy = ACTPolicy(cfg, dataset_stats=dataset_metadata.stats)
    policy.train()
    policy.to(device)

    dataset = LeRobotDataset("OMY", root=dataset_directory, delta_timestamps=delta_timestamps)

    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=4,
        batch_size=64,
        shuffle=True,
        pin_memory=device.type != "cpu",
        drop_last=True,
    )

    step = 0
    done = False
    loss_log = defaultdict(list)
    while not done:
        for batch in dataloader:
            batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            loss, loss_dict = policy.forward(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if step % log_freq == 0:
                for k, v in loss_dict.items():
                    print(f"step: {step} | {k}: {v:.6f}")
                    loss_log[k].append((step, v))
                    
            step += 1
            
            if step >= training_steps:
                done = True
                break

    policy.save_pretrained(output_directory)

    plt.figure(figsize=(10, 6))
    for k, values in loss_log.items():
        steps, losses = zip(*values)
        plt.plot(steps, losses, label=k)

    plt.title("Loss over Training Steps")
    plt.xlabel("Training Step")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_directory / "loss_plot.png")
    plt.show()

if __name__ == "__main__":
    main()
