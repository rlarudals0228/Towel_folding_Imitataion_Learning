import torch
from pathlib import Path
import matplotlib.pyplot as plt
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.datasets.lerobot_dataset import LeRobotDataset

class MultiEpisodeSampler(torch.utils.data.Sampler):
    def __init__(self, dataset: LeRobotDataset, episode_indices: list[int]):
        self.frame_ids = []
        self.episode_bounds = []

        for ep_idx in episode_indices:
            from_idx = dataset.episode_data_index["from"][ep_idx].item()
            to_idx = dataset.episode_data_index["to"][ep_idx].item()
            self.frame_ids.extend(range(from_idx, to_idx))
            self.episode_bounds.append((from_idx, to_idx))
            
    def __iter__(self):
        return iter(self.frame_ids)

    def __len__(self) -> int:
        return len(self.frame_ids)

pretrained_policy_path = Path("/home/dam/kkm_ws/train/towel_final_fold_model")
eval_dir = pretrained_policy_path / "evaluation"
eval_dir.mkdir(parents=True, exist_ok=True)

dataset_directory = Path("/home/dam/colcon_ws/src/ros2_lerobot/demo_data/towel_final_fold")

dataset = LeRobotDataset('OMY', root=dataset_directory)

device = torch.device("cuda")

policy = ACTPolicy.from_pretrained(pretrained_policy_path)

policy.eval()

episode_indices = list(range(3))
episode_sampler = MultiEpisodeSampler(dataset, episode_indices)
eval_dataloader = torch.utils.data.DataLoader(
    dataset,
    num_workers=1,
    batch_size=1,
    shuffle=False,
    pin_memory=device.type != "cpu",
    sampler=episode_sampler,
)

policy.reset()

actions = []
gt_actions = []
images = []
for batch in eval_dataloader:
    inp_batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
    action = policy.select_action(inp_batch)
    actions.append(action)
    gt_actions.append(inp_batch["action"][0].unsqueeze(0))
    images.append(inp_batch["observation.image"])

actions = torch.cat(actions, dim=0)
gt_actions = torch.cat(gt_actions, dim=0)
print(f"Mean action error: {torch.mean(torch.abs(actions - gt_actions)).item():.5f}")

action_dim = 7

for i in range(action_dim):
    plt.figure(figsize=(8, 4))
    plt.plot(actions[:, i].cpu().numpy(), label="pred")
    plt.plot(gt_actions[:, i].cpu().numpy(), label="gt", linestyle='--')
    plt.title(f"Action Dimension {i}")
    plt.xlabel("Step")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(eval_dir / f"joint_{i}.png")
    plt.close()