"""NERO configuration for the installed OpenPI commit; starts from official pi0_base."""
import dataclasses
from pathlib import Path

from openpi import transforms
from openpi.models.pi0_config import Pi0Config
from openpi.training import config, optimizer, weight_loaders

from nero_pi0_adapter import CAMERAS, JOINT_MASK, NeroInputs, NeroOutputs


NAME = "pi0_nero_box_v1"
TRAIN_REPO = "local/nero_box_v1_train"
HORIZON = 50


def statistics_data_config():
    """Same pre-normalization transforms as training, without tokenizer construction."""
    return config.DataConfig(
        repo_id=TRAIN_REPO,
        asset_id=TRAIN_REPO,
        prompt_from_task=True,
        action_sequence_keys=("action",),
        use_quantile_norm=False,
        repack_transforms=transforms.Group(inputs=[transforms.RepackTransform({
            "images": {role: f"observation.images.{role}" for role in CAMERAS},
            "state": "observation.state",
            "actions": "action",
            "prompt": "prompt",
        })]),
        data_transforms=transforms.Group(
            inputs=[NeroInputs(), transforms.DeltaActions(JOINT_MASK)],
            outputs=[transforms.AbsoluteActions(JOINT_MASK), NeroOutputs()],
        ),
    )


@dataclasses.dataclass(frozen=True)
class NeroDataConfig(config.DataConfigFactory):
    def create(self, assets_dirs, model_config):
        base = self.create_base_config(assets_dirs, model_config)
        spec = statistics_data_config()
        return dataclasses.replace(
            base,
            repack_transforms=spec.repack_transforms,
            data_transforms=spec.data_transforms,
            model_transforms=config.ModelTransformFactory()(model_config),
            action_sequence_keys=spec.action_sequence_keys,
            prompt_from_task=True,
            use_quantile_norm=False,
        )


def get_train_config(project=Path("/home/clwu/vla-wu"), exp_name="nero_smoke_v1"):
    """Initial 100-step LoRA smoke preset; constructing it does not train a model."""
    project = Path(project).resolve()
    model = Pi0Config(
        pi05=False,
        action_dim=32,
        action_horizon=HORIZON,
        max_token_len=128,
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora",
    )
    return config.TrainConfig(
        name=NAME,
        exp_name=exp_name,
        model=model,
        weight_loader=weight_loaders.CheckpointWeightLoader(
            str(project / "checkpoints/openpi-assets/checkpoints/pi0_base/params")),
        freeze_filter=model.get_freeze_filter(),
        ema_decay=None,
        data=NeroDataConfig(repo_id=TRAIN_REPO),
        assets_base_dir=str(project / "assets"),
        checkpoint_base_dir=str(project / "checkpoints"),
        batch_size=2,
        num_workers=0,
        num_train_steps=100,
        log_interval=10,
        save_interval=100,
        lr_schedule=optimizer.CosineDecaySchedule(
            warmup_steps=10, peak_lr=2.5e-5, decay_steps=1000, decay_lr=2.5e-6),
        wandb_enabled=False,
        overwrite=False,
        resume=False,
        fsdp_devices=1,
    )
