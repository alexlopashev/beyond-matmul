import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from beyond_matmul import olmoe_route_plan


def _eager_reference(experts, hidden_states, top_k_index, top_k_weights):
    final_hidden_states = torch.zeros_like(hidden_states)
    with torch.no_grad():
        expert_mask = torch.nn.functional.one_hot(
            top_k_index, num_classes=experts.num_experts
        ).permute(2, 1, 0)
        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

    for expert_idx in expert_hit:
        expert_idx = expert_idx[0]
        top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
        current_state = hidden_states[token_idx]
        gate, up = torch.nn.functional.linear(
            current_state, experts.gate_up_proj[expert_idx]
        ).chunk(2, dim=-1)
        current_hidden_states = experts.act_fn(gate) * up
        current_hidden_states = torch.nn.functional.linear(
            current_hidden_states, experts.down_proj[expert_idx]
        )
        current_hidden_states = (
            current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
        )
        final_hidden_states.index_add_(
            0, token_idx, current_hidden_states.to(final_hidden_states.dtype)
        )
    return final_hidden_states


class _TinyExperts(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.num_experts = 4
        self.hidden_dim = 3
        self.intermediate_dim = 2
        self.gate_up_proj = torch.nn.Parameter(
            torch.arange(4 * 4 * 3, dtype=torch.float32).reshape(4, 4, 3) / 37
        )
        self.down_proj = torch.nn.Parameter(
            torch.arange(4 * 3 * 2, dtype=torch.float32).reshape(4, 3, 2) / 23
        )
        self.act_fn = torch.nn.functional.silu
        self.has_gate = True
        self.has_bias = False
        self.is_transposed = False
        self.is_concatenated = True
        self.eval()


class OlmoeStableRouteTests(unittest.TestCase):
    def setUp(self):
        self.top_k_index = torch.tensor(
            [
                [2, 1, 2],
                [0, 2, 1],
                [2, 0, 2],
                [1, 2, 0],
            ],
            dtype=torch.long,
        )
        self.top_k_weights = torch.tensor(
            [
                [0.5, 0.3, 0.2],
                [0.6, 0.3, 0.1],
                [0.4, 0.4, 0.2],
                [0.7, 0.2, 0.1],
            ],
            dtype=torch.float32,
        )

    def test_route_plan_matches_eager_expert_slot_token_traversal(self):
        plan = olmoe_route_plan.build_stable_route_plan(
            self.top_k_index,
            self.top_k_weights,
            num_experts=4,
        )

        self.assertEqual(plan.active_experts, (0, 1, 2))
        self.assertEqual(plan.expert_offsets, (0, 3, 6, 12, 12))
        self.assertEqual(
            list(zip(plan.expert_indices.tolist(), plan.top_k_positions.tolist(), plan.token_indices.tolist())),
            [
                (0, 0, 1),
                (0, 1, 2),
                (0, 2, 3),
                (1, 0, 3),
                (1, 1, 0),
                (1, 2, 1),
                (2, 0, 0),
                (2, 0, 2),
                (2, 1, 1),
                (2, 1, 3),
                (2, 2, 0),
                (2, 2, 2),
            ],
        )
        self.assertEqual(plan.metadata()["order"], "expert_then_top_k_slot_then_token")
        self.assertEqual(plan.metadata()["empty_expert_count"], 1)

    def test_plan_handles_zero_weight_interface_sentinels_without_real_expert_work(self):
        indices = torch.tensor([[0, 4], [3, 1]], dtype=torch.long)
        weights = torch.tensor([[1.0, 0.0], [0.75, 0.25]])

        plan = olmoe_route_plan.build_stable_route_plan(
            indices,
            weights,
            num_experts=4,
        )

        self.assertEqual(plan.sentinel_route_count, 1)
        self.assertEqual(plan.valid_route_count, 3)
        self.assertEqual(plan.active_experts, (0, 1, 3))
        self.assertNotIn(4, plan.expert_indices.tolist())

    def test_sentinel_execution_matches_a_zero_weight_in_range_placeholder(self):
        experts = _TinyExperts()
        hidden_states = torch.randn(2, 3, generator=torch.Generator().manual_seed(5))
        indices = torch.tensor([[0, 4], [3, 1]], dtype=torch.long)
        placeholder_indices = torch.tensor([[0, 2], [3, 1]], dtype=torch.long)
        weights = torch.tensor([[1.0, 0.0], [0.75, 0.25]])
        expected = _eager_reference(
            experts, hidden_states, placeholder_indices, weights
        )

        actual = olmoe_route_plan.stable_route_experts_forward(
            experts, hidden_states, indices, weights
        )

        self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(
            experts._beyond_matmul_last_route_plan["sentinel_route_count"], 1
        )

    def test_plan_handles_all_experts_active(self):
        indices = torch.tensor([[0, 1], [2, 3]], dtype=torch.long)
        weights = torch.full((2, 2), 0.5)

        plan = olmoe_route_plan.build_stable_route_plan(
            indices, weights, num_experts=4
        )

        self.assertEqual(plan.active_experts, (0, 1, 2, 3))
        self.assertEqual(plan.expert_offsets, (0, 1, 2, 3, 4))
        self.assertEqual(plan.metadata()["empty_expert_count"], 0)

    def test_stable_executor_is_bitwise_equal_to_eager_reference(self):
        experts = _TinyExperts()
        hidden_states = torch.tensor(
            [
                [0.2, -0.4, 0.7],
                [0.1, 0.5, -0.3],
                [-0.6, 0.8, 0.2],
                [0.9, -0.2, 0.4],
            ],
            dtype=torch.float32,
        )
        expected = _eager_reference(
            experts, hidden_states, self.top_k_index, self.top_k_weights
        )

        actual = olmoe_route_plan.stable_route_experts_forward(
            experts, hidden_states, self.top_k_index, self.top_k_weights
        )

        self.assertTrue(torch.equal(actual, expected))
        metadata = experts._beyond_matmul_last_route_plan
        self.assertEqual(metadata["execution_path"], "stable_route_plan")
        self.assertEqual(metadata["route_plan_build_count"], 1)
        self.assertFalse(metadata["fallback_used"])

    def test_executor_builds_the_route_plan_exactly_once(self):
        experts = _TinyExperts()
        hidden_states = torch.randn(4, 3, generator=torch.Generator().manual_seed(7))

        with mock.patch.object(
            olmoe_route_plan,
            "build_stable_route_plan",
            wraps=olmoe_route_plan.build_stable_route_plan,
        ) as build:
            olmoe_route_plan.stable_route_experts_forward(
                experts, hidden_states, self.top_k_index, self.top_k_weights
            )

        build.assert_called_once_with(
            self.top_k_index,
            self.top_k_weights,
            num_experts=experts.num_experts,
        )

    def test_training_mode_uses_explicit_eager_fallback_and_records_reason(self):
        experts = _TinyExperts().train()
        hidden_states = torch.randn(4, 3, generator=torch.Generator().manual_seed(11))
        expected = _eager_reference(
            experts, hidden_states, self.top_k_index, self.top_k_weights
        )

        actual = olmoe_route_plan.stable_route_experts_forward(
            experts, hidden_states, self.top_k_index, self.top_k_weights
        )

        self.assertTrue(torch.equal(actual, expected))
        self.assertEqual(
            experts._beyond_matmul_last_route_plan,
            {
                "backend": olmoe_route_plan.BACKEND_NAME,
                "execution_path": "eager_fallback",
                "fallback_used": True,
                "fallback_reason": "training_mode_not_supported",
                "route_plan_build_count": 0,
            },
        )

    def test_registration_uses_the_public_transformers_experts_interface(self):
        interface = SimpleNamespace(register=mock.Mock())

        registered = olmoe_route_plan.register_transformers_backend(interface)

        self.assertEqual(registered, olmoe_route_plan.BACKEND_NAME)
        interface.register.assert_called_once_with(
            olmoe_route_plan.BACKEND_NAME,
            olmoe_route_plan.stable_route_experts_forward,
        )

    def test_bfloat16_duplicate_accumulation_matches_eager(self):
        experts = _TinyExperts().to(dtype=torch.bfloat16)
        hidden_states = torch.tensor(
            [
                [0.2, -0.4, 0.7],
                [0.1, 0.5, -0.3],
                [-0.6, 0.8, 0.2],
                [0.9, -0.2, 0.4],
            ],
            dtype=torch.bfloat16,
        )
        weights = self.top_k_weights.to(dtype=torch.bfloat16)

        expected = _eager_reference(
            experts, hidden_states, self.top_k_index, weights
        )
        actual = olmoe_route_plan.stable_route_experts_forward(
            experts, hidden_states, self.top_k_index, weights
        )

        self.assertTrue(torch.equal(actual, expected))

    def test_execution_audit_counts_stable_and_fallback_paths(self):
        experts = _TinyExperts()
        hidden_states = torch.randn(4, 3, generator=torch.Generator().manual_seed(13))

        olmoe_route_plan.begin_execution_audit()
        olmoe_route_plan.stable_route_experts_forward(
            experts, hidden_states, self.top_k_index, self.top_k_weights
        )
        experts.train()
        olmoe_route_plan.stable_route_experts_forward(
            experts, hidden_states, self.top_k_index, self.top_k_weights
        )
        audit = olmoe_route_plan.end_execution_audit()

        self.assertEqual(audit["calls"], 2)
        self.assertEqual(audit["stable_route_plan_calls"], 1)
        self.assertEqual(audit["eager_fallback_calls"], 1)
        self.assertEqual(audit["route_plan_build_count"], 1)
        self.assertEqual(audit["fallback_reasons"], {"training_mode_not_supported": 1})


if __name__ == "__main__":
    unittest.main()
