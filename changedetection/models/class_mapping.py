"""
HICD V5 类别映射与数据集配置。
基于 V4 class_mapping.py，新增分支路由支持。
"""
import os
import yaml
import torch

# 默认类别定义（0617final）
TARGET_NAMES = [
    "Farmland", "Runway", "Taxiway", "Apron", "Highway",
    "Building", "Tank", "Aircraft", "Vessel", "Crater",
]
STATE_NAMES = ["NoChange", "Damaged", "Reduced", "Added", "Extended", "Replaced"]
NUM_TARGETS = len(TARGET_NAMES)
NUM_STATES = len(STATE_NAMES)

# CLIP 文本提示词（默认）
CLIP_TEXT_PROMPTS = [
    "farmland, agricultural field, crop land",
    "runway, airstrip, landing strip",
    "taxiway, aircraft taxi path",
    "apron, aircraft parking area",
    "highway, main road, expressway",
    "building, structure, house",
    "fuel tank, storage tank, oil tank",
    "aircraft, plane, airplane",
    "vessel, ship, boat",
    "crater, bomb crater, impact crater",
]

# 默认分支路由
DEFAULT_BRANCH_ROUTING = {
    "Building": "instance",
    "Aircraft": "instance",
    "Tank": "instance",
    "Vessel": "instance",
    "Crater": "instance",
    "Runway": "semantic",
    "Taxiway": "semantic",
    "Apron": "semantic",
    "Highway": "semantic",
    "Farmland": "semantic",
}


def get_valid_state_mask():
    """默认层级有效性矩阵：(num_targets, num_states)"""
    mask = torch.zeros(NUM_TARGETS, NUM_STATES)
    # Farmland: NoChange, Damaged
    mask[0, 0] = 1; mask[0, 1] = 1
    # Runway, Taxiway, Apron, Highway: NoChange-Damaged-Reduced-Added-Extended
    for i in range(1, 5):
        mask[i, 0:5] = 1
    # Building: NoChange-Damaged-Reduced-Added-Extended
    mask[5, 0:5] = 1
    # Tank: NoChange-Damaged-Reduced-Added
    mask[6, 0:4] = 1
    # Aircraft, Vessel: NoChange-Damaged-Reduced-Added-Replaced
    for i in [7, 8]:
        mask[i, 0] = 1; mask[i, 1] = 1; mask[i, 2] = 1; mask[i, 3] = 1; mask[i, 5] = 1
    # Crater: NoChange only
    mask[9, 0] = 1
    return mask


class DatasetConfig:
    """数据集配置，从 YAML 文件加载。"""
    def __init__(self, name, num_targets, num_states,
                 target_names, state_names, clip_text_prompts,
                 target_valid_states, train_id_map, state_clip_prompts=None,
                 branch_routing=None, branch_config=None):
        self.name = name
        self.num_targets = num_targets
        self.num_states = num_states
        self.target_names = target_names
        self.state_names = state_names
        self.clip_text_prompts = clip_text_prompts
        self.target_valid_states = {int(k): v for k, v in target_valid_states.items()}
        self.train_id_map = {int(k): v for k, v in train_id_map.items()}
        self.state_clip_prompts = state_clip_prompts or []
        self.branch_routing = branch_routing or DEFAULT_BRANCH_ROUTING
        self.branch_config = branch_config or {}

    @classmethod
    def load(cls, dataset_name):
        """从 YAML 文件加载数据集配置。"""
        config_dir = os.path.join(os.path.dirname(__file__), '..', 'configs', 'datasets')
        yaml_path = os.path.join(config_dir, f'{dataset_name}.yaml')
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"Dataset config not found: {yaml_path}")

        with open(yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)

        return cls(
            name=data['dataset'],
            num_targets=data['num_targets'],
            num_states=data['num_states'],
            target_names=data['target_names'],
            state_names=data['state_names'],
            clip_text_prompts=data['clip_text_prompts'],
            target_valid_states=data.get('target_valid_states', {}),
            train_id_map=data.get('train_id_map', {}),
            state_clip_prompts=data.get('state_clip_prompts', []),
            branch_routing=data.get('branch_routing', {}),
            branch_config=data.get('branch_config', {}),
        )

    def get_valid_state_mask(self):
        """根据 target_valid_states 生成有效性矩阵。"""
        mask = torch.zeros(self.num_targets, self.num_states)
        for target_idx, valid_states in self.target_valid_states.items():
            if target_idx < self.num_targets:
                for s in valid_states:
                    if s < self.num_states:
                        mask[target_idx, s] = 1.0
        return mask

    def get_branch_targets(self, branch='instance'):
        """获取指定分支负责的目标类别列表。"""
        return [t for t, b in self.branch_routing.items() if b == branch]

    def print_summary(self):
        print(f"  Dataset: {self.name}")
        print(f"  Targets: {self.num_targets} ({', '.join(self.target_names)})")
        print(f"  States: {self.num_states} ({', '.join(self.state_names)})")
        if self.branch_routing:
            inst = self.get_branch_targets('instance')
            sem = self.get_branch_targets('semantic')
            print(f"  Instance branch: {inst}")
            print(f"  Semantic branch: {sem}")
