"""
PanelTrajectory — Records the complete trajectory of a multi-expert joint diagnostic discussion

The output trajectory format is compatible with OpenAI messages and can be directly used for SFT training.
"""

import json
import time
from typing import List, Dict, Optional


class PanelTrajectory:
    """
    Records the complete trajectory of a multi-expert joint diagnostic discussion.

    The discussion is divided into 4 phases:
      - Phase 1: Independent Review (each expert reviews images independently)
      - Phase 2: Panel Discussion (structured discussion)
      - Phase 3: Moderator Synthesis (moderator synthesis)
      - Phase 4: Final Diagnosis (final diagnosis)

    Usage:
        traj = PanelTrajectory(patient_id="1007146", question="...", options={...})
        traj.add_phase("independent_review", expert_outputs={...})
        traj.add_phase("panel_discussion", discussion_rounds=[...])
        traj.add_phase("moderator_synthesis", consensus="...")
        traj.add_phase("final_diagnosis", answer="B", reasoning="...")
        traj.finalize(correct=True)

        # Export
        traj.save("trajectory.json")
        messages = traj.to_sft_messages()
        traj.save_sft("sft_data.jsonl")
    """

    def __init__(
        self,
        patient_id: str = "",
        question: str = "",
        options: Optional[Dict[str, str]] = None,
        modalities: Optional[List[str]] = None,
        experts: Optional[List[str]] = None,
        metadata: Optional[dict] = None,
    ):
        self.patient_id = patient_id
        self.question = question
        self.options = options or {}
        self.modalities = modalities or []
        self.experts = experts or []
        self.metadata = metadata or {}
        self.phases: List[dict] = []
        self.final_result: dict = {}
        self.created_at = time.strftime("%Y-%m-%dT%H:%M:%S")
        # Track reasoning insight separately for distillation
        self.reasoning_insight: str = ""

    def add_phase(
        self,
        phase_name: str,
        expert_outputs: Optional[Dict[str, str]] = None,
        discussion_rounds: Optional[List[dict]] = None,
        consensus: Optional[str] = None,
        answer: Optional[str] = None,
        reasoning: Optional[str] = None,
        selected: Optional[List[str]] = None,
    ):
        """
        Record one discussion phase.

        Args:
            phase_name: Phase name (modality_selection / independent_review / panel_discussion / moderator_synthesis / final_diagnosis)
            expert_outputs: Phase 2 - each expert's independent analysis {expert_name: output_text}
            discussion_rounds: Phase 3 - discussion rounds [{"round": 1, "messages": [...]}]
            consensus: Phase 4 - consensus from moderator synthesis
            answer: Phase 5 - final answer
            reasoning: Phase 1 (modality selection) or Phase 5 - reasoning chain
            selected: Phase 1 - list of selected modalities
        """
        phase_data = {
            "phase_name": phase_name,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }

        if expert_outputs is not None:
            phase_data["expert_outputs"] = expert_outputs
        if discussion_rounds is not None:
            phase_data["discussion_rounds"] = discussion_rounds
        if consensus is not None:
            phase_data["consensus"] = consensus
        if answer is not None:
            phase_data["answer"] = answer
        if reasoning is not None:
            phase_data["reasoning"] = reasoning
        if selected is not None:
            phase_data["selected"] = selected

        self.phases.append(phase_data)

    def finalize(
        self,
        answer: Optional[str] = None,
        ground_truth: Optional[str] = None,
        consensus_insight: Optional[str] = None,
        total_experts: Optional[int] = None,
        total_rounds: Optional[int] = None,
        success: bool = True,
    ):
        """
        Finalize trajectory recording.

        Args:
            answer: final answer
            ground_truth: correct answer (optional)
            consensus_insight: multi-expert consensus insight (used for insight learning in the distill plan)
            total_experts: number of participating experts
            total_rounds: total number of discussion rounds
            success: whether it completed successfully
        """
        # Store reasoning insight for distillation
        self.reasoning_insight = consensus_insight or ""

        self.final_result = {
            "answer": answer,
            "ground_truth": ground_truth,
            "correct": answer == ground_truth if ground_truth and answer else None,
            "consensus_insight": consensus_insight,
            "total_experts": total_experts or len(self.experts),
            "total_rounds": total_rounds,
            "success": success,
            "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }

    def to_dict(self) -> dict:
        """
        Serialize to a dictionary for JSON storage.

        Returns:
            dict: dictionary containing the complete trajectory data
        """
        return {
            "patient_id": self.patient_id,
            "question": self.question,
            "options": self.options,
            "modalities": self.modalities,
            "experts": self.experts,
            "phases": self.phases,
            "final_result": self.final_result,
            "created_at": self.created_at,
            "metadata": self.metadata,
        }

    def to_sft_messages(self) -> List[dict]:
        """
        Convert to SFT training format (OpenAI messages).

        Distillation goal: the student learns the teacher's complete flow of executing the medpanel_discuss tool in the sandbox.

        Format:
          system: MedPanel system description
          user:   patient info + question + options + available modalities
          assistant: [Phase 1] modality selection + reasoning
          assistant: [Phase 2] each expert's independent review
          assistant: [Phase 3] structured discussion
          assistant: [Phase 4] moderator synthesis (consensus insight)
          assistant: [Phase 5] diagnostic reasoning based on insight + answer

        The student learns via DPO:
          - Dense feedback: execution quality of each phase
          - Sparse reward: insight quality + trajectory consistency + final correctness
        """
        messages = []

        # ============================================================
        # System: MedPanel system description
        # ============================================================
        experts_str = ", ".join(self.experts) if self.experts else "medical specialists"
        modalities_str = ", ".join(self.modalities) if self.modalities else "MRI"

        system_content = (
            "You are MedPanel, a multi-expert medical discussion system.\n"
            "You conduct structured panel discussions on longitudinal medical imaging data "
            "(Visit 1 vs Visit 2) to answer clinical questions.\n\n"
            f"Available experts: {experts_str}\n"
            f"Available modalities: {modalities_str}\n\n"
            "Your process has 5 phases:\n"
            "1. Modality Selection: Choose the most relevant imaging modalities for the question\n"
            "2. Independent Review: Each expert analyzes their domain images (Visit 1 vs Visit 2)\n"
            "3. Panel Discussion: Experts cross-comment and identify correlations\n"
            "4. Moderator Synthesis: Integrate findings into a consensus insight\n"
            "5. Final Diagnosis: Reason from insight to answer the clinical question"
        )
        messages.append({"role": "system", "content": system_content})

        # ============================================================
        # User: patient info + question + options
        # ============================================================
        user_content = f"Patient: {self.patient_id}\n\n"
        user_content += f"Question: {self.question}\n\n"
        if self.options:
            user_content += "Options:\n"
            for k, v in self.options.items():
                user_content += f"{k}. {v}\n"
        messages.append({"role": "user", "content": user_content})

        # ============================================================
        # Phase 1: Modality Selection
        # ============================================================
        for phase in self.phases:
            if phase["phase_name"] == "modality_selection":
                selected = phase.get("selected", [])
                reasoning = phase.get("reasoning", "")
                content = (
                    "=== Phase 1: Modality Selection ===\n\n"
                    f"**Selected modalities:** {', '.join(selected)}\n\n"
                    f"**Reasoning:** {reasoning}"
                )
                messages.append({"role": "assistant", "content": content})
                break

        # ============================================================
        # Phase 2: Independent Review
        # ============================================================
        for phase in self.phases:
            if phase["phase_name"] == "independent_review":
                expert_outputs = phase.get("expert_outputs", {})
                content = "=== Phase 2: Independent Review ===\n\n"
                for expert, output in expert_outputs.items():
                    content += f"**{expert}:**\n{output}\n\n"
                messages.append({"role": "assistant", "content": content.strip()})
                break

        # ============================================================
        # Phase 3: Panel Discussion
        # ============================================================
        for phase in self.phases:
            if phase["phase_name"] == "panel_discussion":
                discussion_rounds = phase.get("discussion_rounds", [])
                content = "=== Phase 3: Panel Discussion ===\n\n"
                for round_data in discussion_rounds:
                    round_num = round_data.get("round", 0)
                    round_msgs = round_data.get("messages", [])
                    content += f"**Round {round_num}:**\n"
                    for msg in round_msgs:
                        content += f"- {msg.get('speaker', 'Unknown')}: {msg.get('comment', '')}\n"
                    content += "\n"
                messages.append({"role": "assistant", "content": content.strip()})
                break

        # ============================================================
        # Phase 4: Moderator Synthesis
        # ============================================================
        for phase in self.phases:
            if phase["phase_name"] == "moderator_synthesis":
                consensus = phase.get("consensus", "")
                content = (
                    "=== Phase 4: Moderator Synthesis ===\n\n"
                    f"{consensus}"
                )
                messages.append({"role": "assistant", "content": content})
                break

        # ============================================================
        # Phase 5: Final Diagnosis
        # ============================================================
        for phase in self.phases:
            if phase["phase_name"] == "final_diagnosis":
                answer = phase.get("answer", "")
                reasoning = phase.get("reasoning", "")
                content = (
                    "=== Phase 5: Final Diagnosis ===\n\n"
                    f"**Reasoning:** {reasoning}\n\n"
                    f"**Answer:** {answer}"
                )
                messages.append({"role": "assistant", "content": content})
                break

        return messages

    def save(self, path: str):
        """Save the raw trajectory as JSON"""
        data = {
            "patient_id": self.patient_id,
            "question": self.question,
            "options": self.options,
            "modalities": self.modalities,
            "experts": self.experts,
            "created_at": self.created_at,
            "metadata": self.metadata,
            "phases": self.phases,
            "final_result": self.final_result,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def save_sft(self, path: str):
        """
        Save as SFT JSONL format (for OPD-DPO distillation training).

        Output contains:
        - messages: the complete 5-phase execution process (student learns the flow)
        - metadata:
          - Dense feedback: execution quality metrics of each phase
          - Sparse reward: insight quality + trajectory consistency + correctness
        """
        messages = self.to_sft_messages()

        # ============================================================
        # Dense feedback: quality metrics of each phase
        # ============================================================
        phase_metrics = {}
        for phase in self.phases:
            pname = phase["phase_name"]
            if pname == "modality_selection":
                phase_metrics["modality_selection"] = {
                    "selected_count": len(phase.get("selected", [])),
                    "has_reasoning": bool(phase.get("reasoning", "").strip()),
                }
            elif pname == "independent_review":
                expert_outputs = phase.get("expert_outputs", {})
                phase_metrics["independent_review"] = {
                    "expert_count": len(expert_outputs),
                    "avg_output_length": sum(len(v) for v in expert_outputs.values()) / max(len(expert_outputs), 1),
                    "experts": list(expert_outputs.keys()),
                }
            elif pname == "panel_discussion":
                rounds = phase.get("discussion_rounds", [])
                total_comments = sum(len(r.get("messages", [])) for r in rounds)
                phase_metrics["panel_discussion"] = {
                    "round_count": len(rounds),
                    "total_comments": total_comments,
                }
            elif pname == "moderator_synthesis":
                consensus = phase.get("consensus", "")
                phase_metrics["moderator_synthesis"] = {
                    "consensus_length": len(consensus),
                    "has_key_findings": "Key Findings" in consensus or "key findings" in consensus.lower(),
                    "has_temporal_changes": "Temporal Changes" in consensus or "temporal" in consensus.lower(),
                    "has_clinical_implications": "Clinical Implications" in consensus or "clinical" in consensus.lower(),
                }
            elif pname == "final_diagnosis":
                phase_metrics["final_diagnosis"] = {
                    "has_reasoning": bool(phase.get("reasoning", "").strip()),
                    "has_answer": bool(phase.get("answer", "").strip()),
                }

        # ============================================================
        # Sparse reward: overall quality metrics
        # ============================================================
        consensus_insight = self.final_result.get("consensus_insight", "")
        sparse_reward_metrics = {
            "insight_length": len(consensus_insight),
            "insight_quality": {
                "structured": all(k in consensus_insight for k in ["Key Findings", "Temporal Changes", "Clinical Implications"]),
                "has_confidence": "Confidence Level" in consensus_insight,
            },
            "trajectory_consistency": {
                "total_phases": len(self.phases),
                "expected_phases": 5,
                "all_phases_present": len(self.phases) >= 5,
            },
            "correctness": self.final_result.get("correct", None),
        }

        sft_data = {
            "messages": messages,
            "metadata": {
                "patient_id": self.patient_id,
                "question": self.question,
                "options": self.options,
                "modalities": self.modalities,
                "experts": self.experts,
                "has_followup": self.metadata.get("has_followup", False),
                "created_at": self.created_at,
                # Dense feedback
                "phase_metrics": phase_metrics,
                # Sparse reward
                "sparse_reward": sparse_reward_metrics,
                # Insight (teacher's surplus wisdom)
                "consensus_insight": consensus_insight,
                # Ground truth
                "answer": self.final_result.get("answer", ""),
                "ground_truth": self.final_result.get("ground_truth", ""),
                "correct": self.final_result.get("correct", None),
            },
        }
        with open(path, "w", encoding="utf-8") as f:
            f.write(json.dumps(sft_data, ensure_ascii=False) + "\n")

    def summary(self) -> str:
        """Generate a trajectory summary"""
        lines = [
            f"Patient: {self.patient_id}",
            f"Question: {self.question[:100]}...",
            f"Modalities: {', '.join(self.modalities)}",
            f"Experts: {', '.join(self.experts)}",
            f"Phases: {len(self.phases)}",
        ]
        if self.final_result:
            lines.append(f"Answer: {self.final_result.get('answer', 'N/A')}")
            lines.append(f"Correct: {self.final_result.get('correct', 'N/A')}")
        return "\n".join(lines)
