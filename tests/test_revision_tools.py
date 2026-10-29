"""
Tests for the machinery behind the revision experiments: checkpoint pruning,
BatchNorm re-estimation, and heads trained on cached features.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from speedplus_pose.batchnorm import recompute_bn_statistics
from speedplus_pose.evaluate import summarise
from speedplus_pose.features import to_logits
from speedplus_pose.heads import HeadConfig, build_head, head_from_checkpoint, predict_head, subsample, train_head
from speedplus_pose.normalisation import TranslationNormaliser
from speedplus_pose.train import TrainingHistory, _prune, _record_save, read_ledger, selected_checkpoint


# --- checkpoint ledger and pruning -------------------------------------------


def _save(directory, name, epoch, history):
    (directory / name).write_bytes(b"x")
    _record_save(directory, name, epoch, history)


def test_pruning_keeps_latest_and_best(tmp_path):
    val_r = [0.9, 0.8, 0.7, 0.3, 0.5, 0.6, 0.4, 0.45]
    history = TrainingHistory(
        train_losses=[0.0] * 8,
        validation_translation_errors=[0.1] * 8,
        validation_rotation_errors=val_r,
    )
    for epoch in (1, 3, 5, 7):  # checkpoint_every = 2 saves 0-indexed odd epochs
        _save(tmp_path, f"checkpoint_{epoch:02d}.pt", epoch, history)
        _prune(tmp_path)

    on_disk = sorted(p.name for p in tmp_path.glob("checkpoint_*.pt"))
    # epoch 3 (val_r 0.3) is the best saved; epoch 7 is the latest.
    assert on_disk == ["checkpoint_03.pt", "checkpoint_07.pt"]
    assert selected_checkpoint(tmp_path)["epoch"] == 3
    assert [e["pruned"] for e in read_ledger(tmp_path)] == [True, False, True, False]


def test_selection_ignores_unsaved_epochs(tmp_path):
    """Epoch 0 has the lowest val_r but was never checkpointed, so it cannot win."""
    history = TrainingHistory([0.0] * 4, [0.1] * 4, [0.1, 0.6, 0.5, 0.4])
    for epoch in (1, 3):
        _save(tmp_path, f"checkpoint_{epoch:02d}.pt", epoch, history)
    assert selected_checkpoint(tmp_path)["epoch"] == 3


# --- BatchNorm re-estimation -------------------------------------------------


def test_recompute_matches_the_data_and_touches_no_weight():
    torch.manual_seed(0)
    backbone = nn.Sequential(nn.Conv2d(3, 4, 1), nn.BatchNorm2d(4), nn.ReLU(), nn.BatchNorm2d(4))
    before = {k: v.clone() for k, v in backbone.state_dict().items() if "running" not in k and "num_batches" not in k}

    images = torch.randn(64, 3, 5, 5) * 3 + 2
    loader = DataLoader(TensorDataset(images, torch.zeros(64)), batch_size=16)
    seen = recompute_bn_statistics(backbone, loader, torch.device("cpu"))

    assert seen == 64
    # The first BN sees conv(images); its running mean is that activation's mean.
    with torch.no_grad():
        activation = backbone[0](images)
    expected_mean = activation.mean(dim=(0, 2, 3))
    assert torch.allclose(backbone[1].running_mean, expected_mean, atol=1e-5)

    after = {k: v for k, v in backbone.state_dict().items() if k in before}
    assert all(torch.equal(before[k], after[k]) for k in before)
    assert not backbone.training
    assert all(m.momentum == 0.1 for m in backbone.modules() if isinstance(m, nn.BatchNorm2d))


def test_recompute_can_be_restricted_to_some_layers():
    # AdaBN's primary scope: only layer 4's statistics move; the rest keep theirs.
    torch.manual_seed(0)
    backbone = nn.Sequential(nn.Conv2d(3, 4, 1), nn.BatchNorm2d(4), nn.ReLU(), nn.BatchNorm2d(4))
    kept = {k: v.clone() for k, v in backbone[1].state_dict().items()}

    images = torch.randn(64, 3, 5, 5) * 3 + 2
    loader = DataLoader(TensorDataset(images, torch.zeros(64)), batch_size=16)
    recompute_bn_statistics(backbone, loader, torch.device("cpu"), layers=[backbone[3]])

    assert all(torch.equal(kept[k], v) for k, v in backbone[1].state_dict().items())
    # The adapted layer sees its input normalised by the untouched first BN.
    with torch.no_grad():
        activation = backbone[2](backbone[1](backbone[0](images)))
    assert torch.allclose(backbone[3].running_mean, activation.mean(dim=(0, 2, 3)), atol=1e-5)
    assert not backbone.training


# --- heads on cached features ------------------------------------------------


def _fake_split(n, rng, weight):
    features = rng.normal(size=(n, 32)).astype(np.float32)
    translation = features[:, :3] * 0.5 + np.array([0.0, 0.0, 6.0], dtype=np.float32)
    q = features @ weight
    q[:, 0] = np.abs(q[:, 0]) + 0.1
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    return {
        "features": features,
        "labels": np.concatenate([translation, q], axis=1).astype(np.float32),
        "filename": np.array([f"img{i:05d}.jpg" for i in range(n)]),
    }


def test_head_learns_selects_a_saved_epoch_and_writes_a_prediction_record():
    rng = np.random.default_rng(0)
    weight = rng.normal(size=(32, 4)).astype(np.float32)
    train, val = _fake_split(512, rng, weight), _fake_split(128, rng, weight)
    t = train["labels"][:, :3]
    normaliser = TranslationNormaliser(t.mean(0), t.std(0))

    result = train_head(train, val, normaliser, HeadConfig(epochs=6, hidden=64, batch_size=32, seed=1))

    assert (result.selected_epoch + 1) % 2 == 0
    assert result.validation_rotation_errors[-1] < result.validation_rotation_errors[0]

    record = predict_head(result.head, val, normaliser)
    assert set(record) == {
        "filename", "r_pred", "q_pred", "r_true", "q_true",
        "translation_relative", "translation_metres", "rotation", "range",
    }
    assert np.allclose(np.linalg.norm(record["q_pred"], axis=1), 1.0, atol=1e-5)
    assert summarise(record)["n"] == 128


def test_head_training_is_seed_deterministic():
    rng = np.random.default_rng(0)
    weight = rng.normal(size=(32, 4)).astype(np.float32)
    train, val = _fake_split(256, rng, weight), _fake_split(64, rng, weight)
    t = train["labels"][:, :3]
    normaliser = TranslationNormaliser(t.mean(0), t.std(0))
    config = HeadConfig(epochs=2, hidden=16, batch_size=32, seed=3)

    first = train_head(train, val, normaliser, config)
    second = train_head(train, val, normaliser, config)
    assert first.validation_rotation_errors == second.validation_rotation_errors


def test_published_head_loads_from_an_end_to_end_checkpoint():
    head = build_head(2048)
    checkpoint = {"model_state_dict": {f"fc.{k}": v for k, v in head.state_dict().items()} | {"resnet.conv1.weight": torch.zeros(1)}}
    loaded = head_from_checkpoint(checkpoint)
    x = torch.randn(3, 2048)
    assert torch.equal(head(x), loaded(x))


def test_subsample_is_fixed_by_its_seed():
    rng = np.random.default_rng(0)
    split = _fake_split(100, rng, rng.normal(size=(32, 4)).astype(np.float32))
    a, b = subsample(split, 40, seed=5), subsample(split, 40, seed=5)
    assert np.array_equal(a["filename"], b["filename"]) and len(a["filename"]) == 40
    assert not np.array_equal(a["filename"], subsample(split, 40, seed=6)["filename"])


def test_logits_are_fc_of_pooled():
    rng = np.random.default_rng(0)
    pooled = rng.normal(size=(5, 2048)).astype(np.float32)
    weight, bias = rng.normal(size=(1000, 2048)).astype(np.float32), rng.normal(size=1000).astype(np.float32)
    fc = nn.Linear(2048, 1000)
    with torch.no_grad():
        fc.weight.copy_(torch.from_numpy(weight))
        fc.bias.copy_(torch.from_numpy(bias))
        expected = fc(torch.from_numpy(pooled)).numpy()
    assert np.allclose(to_logits(pooled, weight, bias), expected, atol=1e-3)
