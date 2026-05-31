"""Fuzzy relation network exports.

The network is exported as plain CSV node/edge tables to keep the MVP dependency
lightweight while still supporting import into NetworkX, Gephi, Neo4j, or graph
notebook tooling.
"""

from __future__ import annotations

from typing import Any

from .models import FuzzyRule, InferenceResult, Membership, Observation


def build_relation_network(
    observations: list[Observation],
    memberships: list[Membership],
    rules: list[FuzzyRule],
    inference_results: list[InferenceResult],
    aggregations: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    nodes: dict[str, dict[str, Any]] = {}
    edges: list[dict[str, Any]] = []

    def add_node(node_id: str, node_type: str, label: str, **attrs: Any) -> None:
        nodes.setdefault(node_id, {"node_id": node_id, "node_type": node_type, "label": label, **attrs})

    obs_by_id = {obs.observation_id: obs for obs in observations}
    for obs in observations:
        obs_node = f"observation:{obs.standard_observation}"
        source_node = f"source:{obs.source_id}"
        add_node(source_node, "source", obs.source_id)
        add_node(obs_node, "observation", obs.standard_observation or obs.feature_value)
        edges.append(
            {
                "source": source_node,
                "target": obs_node,
                "edge_type": "contains_observation",
                "weight": obs.extraction_confidence,
                "evidence": obs.evidence_text,
            }
        )

    for mem in memberships:
        variable_node = f"fuzzy_variable:{mem.variable_key}"
        obs_node = f"observation:{mem.standard_observation}"
        add_node(variable_node, "fuzzy_variable", mem.variable_key)
        add_node(obs_node, "observation", mem.standard_observation)
        edges.append(
            {
                "source": obs_node,
                "target": variable_node,
                "edge_type": "maps_to_membership",
                "weight": mem.membership,
                "evidence": mem.membership_id,
            }
        )

    for rule in rules:
        rule_node = f"rule:{rule.rule_id}"
        syndrome_node = f"conclusion:{rule.consequent_entity}"
        add_node(rule_node, "rule", rule.rule_id, rule_name=rule.rule_name)
        add_node(syndrome_node, rule.consequent_type, rule.consequent_entity)
        for antecedent in rule.antecedents:
            variable_node = f"fuzzy_variable:{antecedent.variable_key}"
            add_node(variable_node, "fuzzy_variable", antecedent.variable_key)
            edges.append(
                {
                    "source": variable_node,
                    "target": rule_node,
                    "edge_type": "antecedent_of",
                    "weight": antecedent.weight,
                    "threshold": antecedent.threshold,
                }
            )
        edges.append(
            {
                "source": rule_node,
                "target": syndrome_node,
                "edge_type": "infers",
                "weight": rule.rule_weight,
                "review_status": rule.review_status,
            }
        )

    for result in inference_results:
        if result.status != "fired":
            continue
        source_node = f"source:{result.source_id}"
        rule_node = f"rule:{result.rule_id}"
        edges.append(
            {
                "source": source_node,
                "target": rule_node,
                "edge_type": "activates_rule",
                "weight": result.activation,
                "evidence": ";".join(result.supporting_variables),
            }
        )

    for row in aggregations:
        if row.get("aggregation_level") != "global":
            continue
        conclusion_node = f"conclusion:{row['consequent_entity']}"
        global_node = f"global:{row['consequent_entity']}"
        add_node(global_node, "global_conclusion", row["consequent_entity"])
        edges.append(
            {
                "source": conclusion_node,
                "target": global_node,
                "edge_type": "aggregates_to_global",
                "weight": row["mu"],
                "evidence": row.get("evidence_count", 0),
            }
        )

    return list(nodes.values()), edges
