"""Unit tests for federated learning components."""
import numpy as np
import pytest
import torch


def test_model_forward():
    from fl.clients.model import MNISTNet
    model = MNISTNet()
    x = torch.randn(4, 1, 28, 28)
    out = model(x)
    assert out.shape == (4, 10)


def test_model_parameter_count():
    from fl.clients.model import MNISTNet, count_parameters
    model = MNISTNet()
    n = count_parameters(model)
    assert n > 0
    assert n < 5_000_000  # should be lightweight


def test_model_get_set_parameters():
    from fl.clients.model import MNISTNet
    m1, m2 = MNISTNet(), MNISTNet()
    params = m1.get_parameters()
    m2.set_parameters(params)
    for p1, p2 in zip(m1.parameters(), m2.parameters()):
        assert torch.allclose(p1, p2)


def test_iid_partition():
    from unittest.mock import MagicMock
    from fl.data.dataset import partition_iid
    dataset = MagicMock()
    dataset.__len__ = lambda self: 1000
    indices = partition_iid(dataset, num_clients=10)
    assert len(indices) == 10
    total = sum(len(idx) for idx in indices)
    assert total == 1000


def test_non_iid_partition():
    from unittest.mock import MagicMock
    import numpy as np
    from fl.data.dataset import partition_non_iid
    dataset = MagicMock()
    dataset.__len__ = lambda self: 1000
    dataset.targets = torch.tensor(np.random.randint(0, 10, 1000))
    indices = partition_non_iid(dataset, num_clients=5, alpha=0.5)
    assert len(indices) == 5
    total = sum(len(idx) for idx in indices)
    assert total == 1000


def test_fedavg_weighted():
    from fl.server.aggregation import federated_average
    params_a = [np.ones((3, 3)), np.zeros(3)]
    params_b = [np.zeros((3, 3)), np.ones(3)]
    updates = [(params_a, 100), (params_b, 100)]
    aggregated = federated_average(updates)
    assert np.allclose(aggregated[0], 0.5 * np.ones((3, 3)))
    assert np.allclose(aggregated[1], 0.5 * np.ones(3))


def test_fedavg_unequal_weights():
    from fl.server.aggregation import federated_average
    p1 = [np.ones((2, 2))]
    p2 = [np.zeros((2, 2))]
    result = federated_average([(p1, 300), (p2, 100)])
    assert np.allclose(result[0], 0.75 * np.ones((2, 2)))


def test_dp_noise_injection():
    from fl.privacy.dp import add_gaussian_noise
    params = [np.zeros((10, 10)), np.zeros(10)]
    noisy = add_gaussian_noise(params, noise_multiplier=1.0, max_grad_norm=1.0, num_clients=10)
    assert len(noisy) == 2
    # Should not be all zeros after noise
    assert not np.allclose(noisy[0], np.zeros((10, 10)))


def test_privacy_guarantee():
    from fl.privacy.dp import compute_privacy_guarantee
    result = compute_privacy_guarantee(
        noise_multiplier=1.0, num_clients=10, rounds=20, delta=1e-5
    )
    assert "epsilon" in result
    assert "delta" in result
    assert "guarantee" in result
    assert result["epsilon"] > 0


def test_privacy_accountant():
    from fl.privacy.dp import DPConfig, PrivacyAccountant
    config = DPConfig(enabled=True, noise_multiplier=1.0)
    acc = PrivacyAccountant(config=config, num_clients=10, dataset_size=60000)
    entry = acc.update(round_num=10)
    assert "epsilon" in entry
    assert "delta" in entry
    assert entry["epsilon"] > 0


def test_client_selection():
    from fl.server.aggregation import select_clients
    selected = select_clients(num_total=10, fraction=0.5, min_clients=2)
    assert len(selected) >= 2
    assert len(selected) <= 10
    assert len(set(selected)) == len(selected)  # no duplicates


def test_model_divergence():
    from fl.server.aggregation import compute_model_divergence
    global_params = [np.ones((3, 3))]
    client_params = [
        [np.zeros((3, 3))],
        [np.ones((3, 3)) * 0.5],
    ]
    result = compute_model_divergence(global_params, client_params)
    assert "mean_divergence" in result
    assert result["mean_divergence"] > 0
