"""
MedPanelTool — Multi-expert joint diagnostic discussion tool (LLM-driven version)

Core features:
  - LLM autonomously decides which imaging modalities to view (students can learn the selection logic)
  - 2-round limit: avoids excessive calling
  - Explicitly records selection reasoning (for dense feedback)
  - Supports longitudinal follow-up comparison (Visit 1 vs Visit 2)
  - Concurrent execution improves efficiency

Supported imaging modalities:
  - Brain T1 MRI            → Neuroradiologist
  - Brain T2 FLAIR MRI      → Neuroradiologist
  - Brain T1+T2 FLAIR MRI   → Neuroradiologist
  - Cardiac Cine MRI        → Cardiologist
  - Pancreas Gradient-Echo MRI → Gastroenterologist
  - Liver IDEAL MRI         → Hepatologist
  - Whole-Body Dixon MRI    → Body Imager

Dependencies:
  - OpenAI-compatible API (e.g. DashScope qwen-vl-max)
"""

import base64
import json
import os
import re
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Semaphore
from typing import Optional, Dict, List, Any

from distill.tool.common.trajectory import PanelTrajectory


# ============================================================
# Expert role definitions (hardcoded rules)
# ============================================================

EXPERT_ROLES = {
    # ---- Neuroradiology (3 brain modalities share one expert) ----
    "Brain T1 MRI": {
        "name": "Neuroradiologist",
        "cn_name": "Neuroradiologist",
        "system_prompt": (
            "You are a senior neuroradiologist with 20 years of experience in brain MRI interpretation.\n"
            "Your expertise:\n"
            "- Brain atrophy patterns and neurodegenerative diseases\n"
            "- White matter hyperintensities and small vessel disease\n"
            "- Structural abnormalities (tumors, cysts, malformations)\n"
            "- Ventricular size and symmetry assessment\n"
            "- Cortical thickness and gyral pattern evaluation\n\n"
            "When analyzing brain T1 MRI, focus on:\n"
            "1. Overall brain morphology and symmetry\n"
            "2. Ventricular system (size, shape, symmetry)\n"
            "3. Gray-white matter differentiation\n"
            "4. Specific structures (basal ganglia, thalamus, brainstem, cerebellum)\n"
            "5. Any focal lesions or mass effects\n\n"
            "IMPORTANT: You will be shown images from TWO visits (baseline and follow-up). "
            "You MUST explicitly compare the two visits and describe any interval changes "
            "(new findings, progression, regression, or stability).\n\n"
            "Provide findings in structured format:\n"
            "## Visit 1 Findings\n## Visit 2 Findings\n## Interval Changes\n## Clinical Significance"
        ),
        "focus": ["brain atrophy", "white matter changes", "structural abnormalities",
                  "ventricular size", "mass lesions"],
    },
    "Brain T2 FLAIR MRI": {
        "name": "Neuroradiologist",
        "cn_name": "Neuroradiologist",
        "system_prompt": (
            "You are a senior neuroradiologist with 20 years of experience in brain MRI interpretation.\n"
            "Your expertise in T2 FLAIR sequences:\n"
            "- White matter hyperintensities (WMH) — Fazekas scale assessment\n"
            "- Demyelinating lesions (MS-like patterns)\n"
            "- Lacunar infarcts and small vessel disease markers\n"
            "- Periventricular and deep white matter changes\n"
            "- Cortical/subcortical signal abnormalities\n\n"
            "When analyzing brain T2 FLAIR MRI, focus on:\n"
            "1. White matter hyperintensity burden and distribution\n"
            "2. Periventricular vs deep vs juxtacortical WMH pattern\n"
            "3. Lacunar infarcts (location, number, chronicity)\n"
            "4. Any focal T2 hyperintense lesions\n"
            "5. Ventricular size and periventricular signal\n\n"
            "IMPORTANT: You will be shown images from TWO visits (baseline and follow-up). "
            "You MUST explicitly compare the two visits and describe any interval changes "
            "(new WMH, progression of existing lesions, stability).\n\n"
            "Provide findings in structured format:\n"
            "## Visit 1 Findings\n## Visit 2 Findings\n## Interval Changes\n## Clinical Significance"
        ),
        "focus": ["white matter hyperintensities", "small vessel disease",
                  "lacunar infarcts", "demyelination", "periventricular changes"],
    },
    "Brain T1+T2 FLAIR MRI": {
        "name": "Neuroradiologist",
        "cn_name": "Neuroradiologist",
        "system_prompt": (
            "You are a senior neuroradiologist with 20 years of experience in brain MRI interpretation.\n"
            "You have access to BOTH T1-weighted and T2 FLAIR sequences, allowing comprehensive assessment:\n"
            "- T1: brain morphology, atrophy, structural lesions, post-contrast enhancement\n"
            "- T2 FLAIR: white matter hyperintensities, small vessel disease, demyelination\n\n"
            "When analyzing combined T1+T2 FLAIR MRI, focus on:\n"
            "1. Structural abnormalities visible on T1\n"
            "2. White matter changes visible on T2 FLAIR\n"
            "3. Correlation between T1 and T2 findings\n"
            "4. Ventricular system assessment\n"
            "5. Any focal lesions or diffuse changes\n\n"
            "IMPORTANT: You will be shown images from TWO visits (baseline and follow-up). "
            "You MUST explicitly compare the two visits and describe any interval changes.\n\n"
            "Provide findings in structured format:\n"
            "## Visit 1 Findings\n## Visit 2 Findings\n## Interval Changes\n## Clinical Significance"
        ),
        "focus": ["structural abnormalities", "white matter changes", "lesions",
                  "ventricular changes", "T1-T2 correlation"],
    },

    # ---- Cardiology ----
    "Cardiac Cine MRI": {
        "name": "Cardiologist",
        "cn_name": "Cardiologist",
        "system_prompt": (
            "You are a senior cardiologist specializing in cardiac MRI with 18 years of experience.\n"
            "Your expertise:\n"
            "- Ventricular function assessment (EF, wall motion, strain)\n"
            "- Chamber size and morphology evaluation\n"
            "- Valvular structure and function\n"
            "- Myocardial tissue characterization\n"
            "- Pericardial assessment\n\n"
            "When analyzing cardiac cine MRI, focus on:\n"
            "1. Left and right ventricular size and function\n"
            "2. Regional wall motion abnormalities\n"
            "3. Valvular morphology and motion\n"
            "4. Pericardial thickness and effusion\n"
            "5. Great vessels (aorta, pulmonary artery)\n\n"
            "IMPORTANT: You will be shown images from TWO visits (baseline and follow-up). "
            "You MUST explicitly compare the two visits and describe any interval changes "
            "(EF changes, new wall motion abnormalities, valvular progression).\n\n"
            "Provide findings in structured format:\n"
            "## Visit 1 Findings\n## Visit 2 Findings\n## Interval Changes\n## Clinical Significance"
        ),
        "focus": ["ventricular function", "wall motion", "valvular abnormalities",
                  "chamber size", "pericardium"],
    },

    # ---- Gastroenterology ----
    "Pancreas Gradient-Echo MRI": {
        "name": "Gastroenterologist",
        "cn_name": "Gastroenterologist",
        "system_prompt": (
            "You are a senior gastroenterologist with expertise in abdominal MRI and pancreatic imaging, "
            "with 15 years of clinical experience.\n"
            "Your expertise:\n"
            "- Pancreatic morphology and size assessment\n"
            "- Fat infiltration and fatty replacement (in-phase/out-of-phase, multi-echo)\n"
            "- Pancreatic masses and cystic lesions\n"
            "- Ductal system evaluation\n"
            "- Surrounding structures (liver, spleen, kidneys, adrenal glands)\n\n"
            "When analyzing pancreas gradient-echo MRI, focus on:\n"
            "1. Pancreatic size, shape, and contour\n"
            "2. Signal intensity characteristics (especially fat-sensitive sequences)\n"
            "3. Fat content assessment (signal drop on out-of-phase vs in-phase)\n"
            "4. Focal lesions (solid masses, cysts, IPMNs)\n"
            "5. Main pancreatic duct caliber\n\n"
            "IMPORTANT: You will be shown images from TWO visits (baseline and follow-up). "
            "You MUST explicitly compare the two visits:\n"
            "- Has pancreatic fat content changed?\n"
            "- Any new lesions or size changes?\n"
            "- Ductal changes?\n\n"
            "Provide findings in structured format:\n"
            "## Visit 1 Findings\n## Visit 2 Findings\n## Interval Changes\n## Clinical Significance"
        ),
        "focus": ["pancreatic morphology", "fat infiltration", "masses", "cysts", "ductal system"],
    },

    # ---- Hepatology ----
    "Liver IDEAL MRI": {
        "name": "Hepatologist",
        "cn_name": "Hepatologist",
        "system_prompt": (
            "You are a senior hepatologist specializing in liver MRI with 16 years of clinical experience.\n"
            "Your expertise:\n"
            "- Hepatic steatosis quantification (IDEAL/PDFF techniques)\n"
            "- Liver fibrosis assessment (morphological surrogates)\n"
            "- Focal liver lesions characterization (HCC, metastases, hemangiomas, FNH)\n"
            "- Biliary system evaluation\n"
            "- Portal venous system and hepatic vasculature\n"
            "- Spleen size as portal hypertension surrogate\n\n"
            "When analyzing liver IDEAL MRI, focus on:\n"
            "1. Liver fat fraction estimation (IDEAL water/fat images)\n"
            "2. Steatosis grading (none / mild <5% / moderate 5-33% / severe >33%)\n"
            "3. Liver morphology (surface nodularity, lobar redistribution, caudate lobe ratio)\n"
            "4. Focal lesions (number, size, signal characteristics)\n"
            "5. Spleen size and portal vein caliber\n"
            "6. Biliary ducts (dilation, strictures)\n\n"
            "IMPORTANT: You will be shown images from TWO visits (baseline and follow-up). "
            "You MUST explicitly compare the two visits:\n"
            "- Has hepatic steatosis progressed or regressed?\n"
            "- Any new focal lesions?\n"
            "- Morphological changes suggesting fibrosis progression?\n"
            "- Changes in spleen size or portal vein?\n\n"
            "Provide findings in structured format:\n"
            "## Visit 1 Findings\n## Visit 2 Findings\n## Interval Changes\n## Clinical Significance"
        ),
        "focus": ["hepatic steatosis", "liver fibrosis", "focal lesions",
                  "biliary system", "portal hypertension"],
    },

    # ---- Whole-body imaging ----
    "Whole-Body Dixon MRI": {
        "name": "Body Imager",
        "cn_name": "Body Imager",
        "system_prompt": (
            "You are a senior body imaging radiologist with 17 years of experience in whole-body MRI.\n"
            "Your expertise:\n"
            "- Fat distribution and body composition analysis\n"
            "- Organ size and morphology assessment (liver, spleen, kidneys, pancreas)\n"
            "- Musculoskeletal evaluation (muscle mass, bone marrow signal)\n"
            "- Systemic disease manifestations across organ systems\n"
            "- Multi-organ correlation and syndromic patterns\n\n"
            "When analyzing whole-body Dixon MRI, focus on:\n"
            "1. Fat distribution pattern (subcutaneous vs visceral, ectopic fat)\n"
            "2. Organ sizes: liver (hepatomegaly?), spleen (splenomegaly?), kidneys\n"
            "3. Fat infiltration in organs: hepatic steatosis (Dixon fat fraction), pancreatic fat\n"
            "4. Musculoskeletal: muscle mass (sarcopenia?), bone marrow signal abnormalities\n"
            "5. Any focal lesions or masses across body regions\n"
            "6. Lymph nodes (enlarged?)\n\n"
            "IMPORTANT: You will be shown images from TWO visits (baseline and follow-up). "
            "You MUST explicitly compare the two visits:\n"
            "- Body composition changes (weight gain/loss, fat redistribution)\n"
            "- Organ size changes\n"
            "- New or resolved lesions\n"
            "- Progression of fatty infiltration\n\n"
            "Provide findings in structured format:\n"
            "## Visit 1 Findings\n## Visit 2 Findings\n## Interval Changes\n## Clinical Significance"
        ),
        "focus": ["fat distribution", "organ sizes", "systemic findings",
                  "fat infiltration", "musculoskeletal"],
    },
}

MODERATOR_SYSTEM_PROMPT = """You are the moderator of a multi-expert medical panel.

Your role is to:
1. Synthesize expert findings into actionable insights
2. Highlight key changes between Visit 1 and Visit 2
3. Provide concise, structured output

Output format:
**Key Findings:** [1-2 sentences summarizing the most relevant findings]
**Temporal Changes:** [What changed between visits, if anything]
**Clinical Implications:** [What this means for the patient's condition]
**Confidence Level:** [High/Medium/Low based on image quality and clarity of findings]
"""

MODALITY_SELECTOR_SYSTEM_PROMPT = """You are a medical imaging specialist deciding which MRI modalities to analyze for a given clinical question.

Available modalities:
{available_modalities}

Your task:
1. Analyze the clinical question
2. Select the 1-2 most relevant modalities
3. Explain your reasoning

Output format:
**Selected Modalities:** [modality1, modality2]
**Reasoning:** [Why these modalities are most relevant for this question]
"""

# ============================================================
# MedPanelTool
# ============================================================


class MedPanelTool:
    """
    Multi-expert joint diagnostic discussion tool (LLM-driven version)

    Core improvements:
    - LLM autonomously decides which modalities to view (rather than hardcoded routing)
    - Explicitly records selection reasoning (students can learn)
    - 2-round limit (avoids excessive calling)
    - Concurrent execution (improves efficiency)

    Outputs:
    - trajectory: full trajectory (dense feedback)
    - consensus_insight: condensed insight (sparse reward)
    - answer: final answer
    """

    # Global semaphore, controls the total number of concurrent API calls across all instances
    _api_semaphore = None
    _max_api_concurrent = 10  # Default maximum of 10 concurrent API calls

    @classmethod
    def set_max_api_concurrent(cls, max_concurrent: int):
        """Set the global maximum API concurrency"""
        cls._max_api_concurrent = max_concurrent
        cls._api_semaphore = Semaphore(max_concurrent)

    def __init__(
        self,
        model: str = "qwen-vl-max",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        output_dir: str = "./medpanel_output",
        max_api_concurrent: Optional[int] = None,
    ):
        """
        Initialize MedPanelTool

        Args:
            model: LLM model name
            api_key: API key (optional, defaults to environment variable)
            base_url: API base URL (optional)
            temperature: generation temperature
            max_tokens: maximum number of output tokens
            output_dir: trajectory output directory
            max_api_concurrent: maximum API concurrency (global limit, default 10)
        """
        from openai import OpenAI

        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.output_dir = output_dir

        # Initialize the global semaphore
        if max_api_concurrent is not None:
            MedPanelTool.set_max_api_concurrent(max_api_concurrent)
        elif MedPanelTool._api_semaphore is None:
            MedPanelTool._api_semaphore = Semaphore(MedPanelTool._max_api_concurrent)

        os.makedirs(self.output_dir, exist_ok=True)

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=300.0,
            max_retries=3,
        )

        # Statistics
        self.stats = {
            "total_diagnoses": 0,
            "total_llm_calls": 0,
            "errors": 0,
        }

    def diagnose(
        self,
        patient_info: str,
        images: Dict[str, Any],
        question: str,
        options: Any,
        ground_truth: Optional[str] = None,
        patient_id: Optional[str] = None,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Execute multi-expert joint diagnosis (2-round limit version)

        Args:
            patient_info: patient basic information
            images: image data {modality: {"visit_1": [imgs], "visit_2": [imgs]}}
            question: clinical question
            options: list of options
            ground_truth: correct answer (optional)
            patient_id: patient ID (optional)
            verbose: whether to print detailed process

        Returns:
            {
                "trajectory": PanelTrajectory,
                "consensus_insight": str,
                "answer": str,
                "reasoning": str,
                "correct": bool | None,
            }
        """
        self.stats["total_diagnoses"] += 1

        # Initialize trajectory (populate actual modalities and expert info)
        trajectory = PanelTrajectory(
            patient_id=patient_id or "unknown",
            question=question,
            options=options,
            modalities=list(images.keys()),  # NEW: populate actual modalities
        )

        # ====== Phase 1: LLM selects modalities ======
        if verbose:
            print(f"\n[Phase 1] Selecting imaging modalities...")

        available_modalities = list(images.keys())
        modality_selection = self._select_modalities(question, available_modalities)

        selected_modalities = modality_selection["modalities"]
        selection_reasoning = modality_selection["reasoning"]

        # Record to trajectory
        trajectory.add_phase(
            "modality_selection",
            reasoning=selection_reasoning,
            selected=selected_modalities,
        )

        if verbose:
            print(f"  Selected modalities: {selected_modalities}")
            print(f"  Selection reasoning: {selection_reasoning}")

        # ====== Phase 2: Experts independently review images (concurrent, with question and options) ======
        if verbose:
            print(f"\n[Phase 2] Experts independently reviewing images...")

        expert_outputs = {}

        # Prepare concurrent tasks
        tasks = []
        for modality in selected_modalities:
            if modality not in images:
                continue
            if modality not in EXPERT_ROLES:
                continue

            expert = EXPERT_ROLES[modality]
            expert_name = expert["name"]

            # Avoid the same expert reviewing repeatedly
            if expert_name in expert_outputs:
                continue

            tasks.append((expert_name, modality, expert))

        # Concurrent execution (passing in question and options)
        with ThreadPoolExecutor(max_workers=len(tasks)) as executor:
            futures = {}
            for expert_name, modality, expert in tasks:
                future = executor.submit(
                    self._run_independent_review,
                    expert,
                    images[modality],
                    patient_info,
                    verbose,
                    question,  # NEW: pass in clinical question
                    options,   # NEW: pass in options
                )
                futures[future] = expert_name

            for future in as_completed(futures):
                expert_name = futures[future]
                try:
                    output = future.result()
                    expert_outputs[expert_name] = output
                    if verbose:
                        print(f"  ✓ {expert_name} completed")
                except Exception as e:
                    print(f"  ✗ {expert_name} failed: {e}")
                    self.stats["errors"] += 1

        trajectory.add_phase("independent_review", expert_outputs=expert_outputs)

        # NEW: populate actual expert list (for to_sft_messages)
        trajectory.experts = list(expert_outputs.keys())

        # ====== Phase 3: Structured discussion (concurrent, 2 rounds, with question and options) ======
        if verbose:
            print(f"\n[Phase 3] Structured discussion...")

        discussion_rounds = self._run_panel_discussion(
            expert_outputs,
            patient_info,
            max_rounds=2,
            verbose=verbose,
            question=question,  # NEW: pass in clinical question
            options=options,    # NEW: pass in options
        )

        trajectory.add_phase("panel_discussion", discussion_rounds=discussion_rounds)

        # ====== Phase 4: Moderator synthesis (with options) ======
        if verbose:
            print(f"\n[Phase 4] Moderator synthesis...")

        consensus = self._run_moderator_synthesis(
            expert_outputs,
            discussion_rounds,
            question,
            verbose,
            options=options,  # NEW: pass in options
        )

        trajectory.add_phase("moderator_synthesis", consensus=consensus)

        if verbose:
            print(f"  Consensus: {consensus}")

        # ====== Phase 5: Final diagnosis (with patient info + anti-hallucination) ======
        if verbose:
            print(f"\n[Phase 5] Final diagnosis...")

        diagnosis = self._run_final_diagnosis(
            consensus,
            question,
            options,
            verbose,
            patient_info=patient_info,  # NEW: pass in patient info
        )

        trajectory.add_phase("final_diagnosis", answer=diagnosis["answer"], reasoning=diagnosis["reasoning"])

        # Finalize trajectory
        trajectory.finalize(
            answer=diagnosis["answer"],
            ground_truth=ground_truth,
            consensus_insight=consensus,
            total_experts=len(expert_outputs),
            total_rounds=len(discussion_rounds),
            success=True,
        )

        if verbose:
            print(f"\n[Done] Answer: {diagnosis['answer']}")
            if ground_truth:
                correct = "✓" if diagnosis["answer"] == ground_truth else "✗"
                print(f"  Correctness: {correct} (correct answer: {ground_truth})")

        return {
            "trajectory": trajectory,
            "consensus_insight": consensus,
            "answer": diagnosis["answer"],
            "reasoning": diagnosis["reasoning"],
            "correct": (diagnosis["answer"] == ground_truth) if ground_truth else None,
        }

    def diagnose_batch(
        self,
        patient_info: str,
        images: Dict[str, Dict[str, List[str]]],
        questions: List[str],
        options_list: List[List[str]],
        solutions: List[str],
        patient_id: str = "unknown",
        verbose: bool = True,
        max_workers: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Batch diagnosis (concurrent execution)

        Args:
            patient_info: patient basic information
            images: image data
            questions: list of questions
            options_list: list of lists of options
            solutions: list of answers
            patient_id: patient ID
            verbose: whether to print detailed process
            max_workers: number of concurrent threads (default 3)

        Returns:
            List[Dict]: result of each diagnosis
        """
        results = [None] * len(questions)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {}

            for idx, (question, options, solution) in enumerate(
                zip(questions, options_list, solutions)
            ):
                future = executor.submit(
                    self.diagnose,
                    patient_info=patient_info,
                    images=images,
                    question=question,
                    options=options,
                    ground_truth=solution,
                    patient_id=patient_id,
                    verbose=verbose,
                )
                future_to_idx[future] = idx

            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    result = future.result()
                    results[idx] = result
                    if verbose:
                        correct = "✓" if result["correct"] else "✗"
                        print(f"\n[Done {idx+1}/{len(questions)}] Answer: {result['answer']} {correct}")
                except Exception as e:
                    print(f"\n[Error {idx+1}/{len(questions)}] {e}")
                    self.stats["errors"] += 1

        return results  # Return all results including None for failed diagnoses

    # ============================================================
    # Internal methods
    # ============================================================

    def _select_modalities(
        self,
        question: str,
        available_modalities: List[str],
    ) -> Dict[str, Any]:
        """
        LLM-driven selection of the 1-2 most relevant modalities

        Returns:
            {
                "modalities": [str],
                "reasoning": str,
            }
        """
        prompt = MODALITY_SELECTOR_SYSTEM_PROMPT.format(
            available_modalities="\n".join(f"- {m}" for m in available_modalities)
        )

        messages = [
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": f"Question: {question}\n\nPlease select 1-2 modalities and explain your reasoning.",
            },
        ]

        try:
            response = self._call_llm(messages)
            self.stats["total_llm_calls"] += 1

            # Parse output
            modalities = []
            reasoning = ""

            # Extract Selected Modalities
            if "**Selected Modalities:**" in response:
                modalities_line = response.split("**Selected Modalities:**")[1].split("\n")[0]
                modalities = [m.strip() for m in modalities_line.replace("[", "").replace("]", "").split(",")]
                modalities = [m for m in modalities if m in available_modalities]

            # Extract Reasoning
            if "**Reasoning:**" in response:
                reasoning = response.split("**Reasoning:**")[1].strip()

            # If parsing fails, default to selecting all available modalities
            if not modalities:
                modalities = available_modalities
                reasoning = "Failed to parse selection, using all available modalities"

            return {"modalities": modalities, "reasoning": reasoning}

        except Exception as e:
            print(f"[Warning] Modality selection failed: {e}")
            return {
                "modalities": available_modalities,
                "reasoning": f"Selection error: {e}, using all modalities",
            }

    def _run_independent_review(
        self,
        expert: Dict[str, Any],
        images: Dict[str, Any],
        patient_info: str,
        verbose: bool,
        question: str = "",
        options: Any = None,
    ) -> str:
        """Independent image review by a single expert (with clinical question and options context)"""
        # Format options text
        options_text = ""
        if options:
            if isinstance(options, list):
                options_dict = {chr(65 + i): opt for i, opt in enumerate(options)}
            elif isinstance(options, dict):
                options_dict = options
            else:
                options_dict = {}
            options_text = "\n".join(f"{k}. {v}" for k, v in options_dict.items())

        # Build messages (include clinical question and options)
        messages = [
            {"role": "system", "content": expert["system_prompt"]},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Clinical Question:\n{question}\n\nAnswer Options:\n{options_text}\n\nPatient Information:\n{patient_info}\n\n"},
                    {"type": "text", "text": "=== Visit 1 (Baseline) ===\n"},
                ],
            },
        ]

        # Add Visit 1 images (montage + representative)
        visit_1_data = images.get("visit_1", {})
        if isinstance(visit_1_data, dict):
            visit_1_images = visit_1_data.get("images", [])
            visit_1_types = visit_1_data.get("types", [])
        else:
            visit_1_images = visit_1_data
            visit_1_types = []

        if visit_1_images:
            selected = self._select_images(visit_1_images, visit_1_types)
            for img_data in selected:
                messages[1]["content"].append(self._build_image_message(img_data))

        # Add Visit 2 images
        visit_2_data = images.get("visit_2", {})
        if isinstance(visit_2_data, dict):
            visit_2_images = visit_2_data.get("images", [])
            visit_2_types = visit_2_data.get("types", [])
        else:
            visit_2_images = visit_2_data
            visit_2_types = []

        if visit_2_images:
            messages[1]["content"].append({"type": "text", "text": "\n\n=== Visit 2 (Follow-up) ===\n"})
            selected = self._select_images(visit_2_images, visit_2_types)
            for img_data in selected:
                messages[1]["content"].append(self._build_image_message(img_data))

        # Add analysis request
        messages[1]["content"].append(
            {
                "type": "text",
                "text": "\n\nPlease analyze these images with the clinical question and answer options in mind. Focus your findings on what is relevant to answering the question. Provide your findings in the structured format.",
            }
        )

        try:
            response = self._call_llm(messages)
            self.stats["total_llm_calls"] += 1
            return response
        except Exception as e:
            print(f"[Error] Independent review failed: {e}")
            self.stats["errors"] += 1
            return f"Error: {e}"

    def _run_panel_discussion(
        self,
        expert_outputs: Dict[str, str],
        patient_info: str,
        max_rounds: int = 2,
        verbose: bool = True,
        question: str = "",
        options: Any = None,
    ) -> List[Dict[str, Any]]:
        """Structured discussion (2-round limit, concurrent execution, with clinical question context)"""
        discussion_rounds = []

        if not expert_outputs:
            return discussion_rounds

        for round_num in range(1, max_rounds + 1):
            if verbose:
                print(f"  Round {round_num}/{max_rounds}...")

            round_messages = []

            num_experts = len(expert_outputs)
            with ThreadPoolExecutor(max_workers=max(1, num_experts)) as executor:
                futures = {}

                for expert_name, expert_output in expert_outputs.items():
                    expert_config = None
                    for modality, config in EXPERT_ROLES.items():
                        if config["name"] == expert_name:
                            expert_config = config
                            break

                    if not expert_config:
                        continue

                    future = executor.submit(
                        self._run_discussion_round,
                        expert_config,
                        expert_output,
                        expert_outputs,
                        patient_info,
                        round_num,
                        question,
                        options,
                    )
                    futures[future] = expert_name

                for future in as_completed(futures):
                    expert_name = futures[future]
                    try:
                        comment = future.result()
                        round_messages.append(
                            {
                                "speaker": expert_name,
                                "comment": comment,
                            }
                        )
                    except Exception as e:
                        print(f"  [Warning] Discussion round failed for {expert_name}: {e}")

            discussion_rounds.append({"round": round_num, "messages": round_messages})

        return discussion_rounds

    def _run_discussion_round(
        self,
        expert: Dict[str, Any],
        own_output: str,
        all_outputs: Dict[str, str],
        patient_info: str,
        round_num: int,
        question: str = "",
        options: Any = None,
    ) -> str:
        """Single discussion round (with clinical question context)"""
        other_outputs = {k: v for k, v in all_outputs.items() if k != expert["name"]}

        other_findings_lines = []
        for name, output in other_outputs.items():
            other_findings_lines.append(f"**{name}:**\n{output}")
        other_findings_text = "\n".join(other_findings_lines)

        # Format options
        options_text = ""
        if options:
            if isinstance(options, list):
                options_dict = {chr(65 + i): opt for i, opt in enumerate(options)}
            elif isinstance(options, dict):
                options_dict = options
            else:
                options_dict = {}
            options_text = "\n".join(f"{k}. {v}" for k, v in options_dict.items())

        prompt = f"""You are participating in a multi-expert medical panel discussion.

Clinical Question:
{question}

Answer Options:
{options_text}

Your previous findings:
{own_output}

Other experts' findings:
{other_findings_text}

Patient Information:
{patient_info}

Round {round_num} Task:
- Review other experts' findings in the context of the clinical question
- Identify findings relevant to the answer options
- Add any insights from cross-modal correlations that help answer the question
- Keep your comment concise (2-3 sentences)
"""

        messages = [
            {"role": "system", "content": expert["system_prompt"]},
            {"role": "user", "content": prompt},
        ]

        try:
            response = self._call_llm(messages)
            self.stats["total_llm_calls"] += 1
            return response
        except Exception as e:
            print(f"[Error] Discussion round failed: {e}")
            self.stats["errors"] += 1
            return f"Error: {e}"

    def _run_moderator_synthesis(
        self,
        expert_outputs: Dict[str, str],
        discussion_rounds: List[Dict[str, Any]],
        question: str,
        verbose: bool,
        options: Any = None,
    ) -> str:
        """Moderator synthesis (with options context)"""
        expert_summary = "\n\n".join(
            f"**{name}:**\n{output}" for name, output in expert_outputs.items()
        )

        discussion_summary = ""
        for round_data in discussion_rounds:
            discussion_summary += f"\n\n**Round {round_data['round']}:**\n"
            for msg in round_data["messages"]:
                discussion_summary += f"- {msg['speaker']}: {msg['comment']}\n"

        # Format options
        options_text = ""
        if options:
            if isinstance(options, list):
                options_dict = {chr(65 + i): opt for i, opt in enumerate(options)}
            elif isinstance(options, dict):
                options_dict = options
            else:
                options_dict = {}
            options_text = "\n".join(f"{k}. {v}" for k, v in options_dict.items())

        prompt = f"""Please synthesize the following expert panel discussion into a concise clinical insight.

Clinical Question:
{question}

Answer Options:
{options_text}

Expert Findings:
{expert_summary}

Discussion:
{discussion_summary}

Provide a structured synthesis focusing on:
1. Key findings relevant to the question and answer options
2. Temporal changes between visits
3. Clinical implications — which answer option is best supported by the evidence?
4. Confidence level

IMPORTANT: The answer options are listed above. Your synthesis must reference the specific options when discussing clinical implications.
"""

        messages = [
            {"role": "system", "content": MODERATOR_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        try:
            response = self._call_llm(messages)
            self.stats["total_llm_calls"] += 1
            return response
        except Exception as e:
            print(f"[Error] Moderator synthesis failed: {e}")
            self.stats["errors"] += 1
            return f"Error: {e}"

    def _run_final_diagnosis(
        self,
        consensus: str,
        question: str,
        options: Any,
        verbose: bool,
        patient_info: str = "",
    ) -> Dict[str, str]:
        """Final diagnosis (with patient info + anti-hallucination)"""
        if isinstance(options, list):
            options_dict = {chr(65 + i): opt for i, opt in enumerate(options)}
        elif isinstance(options, dict):
            options_dict = options
        else:
            options_dict = {}

        options_text = "\n".join(f"{k}. {v}" for k, v in options_dict.items())
        options_count = len(options_dict)

        prompt = f"""Based on the expert panel's consensus, please provide your final diagnosis.

Clinical Question:
{question}

Answer Options ({options_count} options available):
{options_text}

Patient Information:
{patient_info[:2000]}

Expert Consensus:
{consensus}

IMPORTANT: The {options_count} answer options are listed above. You MUST select exactly one option letter (A, B, C, etc.).
Do NOT claim that options were not provided — they are clearly listed above.
If the answer cannot be determined from the available data, select the "None of the above" option if available, otherwise make your best guess.

Please provide:
1. Your reasoning (2-3 sentences referencing the options)
2. Your answer (just the option letter: A, B, C, etc.)

Output format:
**Reasoning:** [your reasoning]

**Answer:** [option letter]
"""

        messages = [
            {
                "role": "system",
                "content": "You are a clinical decision-maker integrating expert opinions to make a final diagnosis.",
            },
            {"role": "user", "content": prompt},
        ]

        try:
            response = self._call_llm(messages)
            self.stats["total_llm_calls"] += 1

            # Parse answer
            answer = self._parse_answer(response, options_dict)
            reasoning = response

            return {"answer": answer, "reasoning": reasoning}
        except Exception as e:
            print(f"[Error] Final diagnosis failed: {e}")
            self.stats["errors"] += 1
            first_key = list(options_dict.keys())[0] if options_dict else "A"
            return {"answer": first_key, "reasoning": f"Error: {e}"}

    def _select_images(
        self,
        images: List[Any],
        types: List[str],
    ) -> List[Any]:
        """
        Select montage and representative images from the image list

        Args:
            images: list of image data (bytes or str)
            types: list of image types (corresponding to images)

        Returns:
            list of selected images (montage + representative, or the first 2)
        """
        if not images:
            return []

        # Use the types list to find montage and representative
        montage_idx = None
        representative_idx = None

        for idx, img_type in enumerate(types):
            if idx >= len(images):
                break
            type_lower = img_type.lower()
            if montage_idx is None and "montage" in type_lower:
                montage_idx = idx
            elif representative_idx is None and "representative" in type_lower:
                representative_idx = idx

        selected = []
        if montage_idx is not None:
            selected.append(images[montage_idx])
        if representative_idx is not None and representative_idx != montage_idx:
            selected.append(images[representative_idx])

        # If not found, take the first 2
        if not selected:
            selected = images[:2]

        return selected

    def _build_image_message(self, image_data) -> Dict[str, Any]:
        """
        Build an image message

        Args:
            image_data: can be bytes (raw image data) or str (file path)
        """
        if isinstance(image_data, bytes):
            # Already byte data, encode directly
            b64_data = base64.b64encode(image_data).decode("utf-8")
        elif isinstance(image_data, str):
            # It is a file path, read the file
            with open(image_data, "rb") as f:
                b64_data = base64.b64encode(f.read()).decode("utf-8")
        else:
            raise ValueError(f"Unsupported image_data type: {type(image_data)}")

        return {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{b64_data}",
            },
        }

    def _call_llm(self, messages: List[Dict[str, Any]], max_retries: int = 5) -> str:
        """
        Call the LLM (with retry mechanism and global concurrency control)

        Args:
            messages: list of messages
            max_retries: maximum number of retries (default 5)

        Returns:
            LLM response text

        Raises:
            Exception: still failing after retries are exhausted
        """
        import time

        for attempt in range(max_retries):
            try:
                # Use semaphore to control concurrency
                with MedPanelTool._api_semaphore:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                    )
                return response.choices[0].message.content

            except Exception as e:
                error_str = str(e).lower()

                # Check whether it is a rate limit error (429)
                if "429" in error_str or "rate limit" in error_str or "too many requests" in error_str:
                    if attempt < max_retries - 1:
                        # Exponential backoff: 2^attempt seconds (2, 4, 8, 16, 32 seconds)
                        wait_time = 2 ** (attempt + 1)
                        print(f"[MedPanelTool] Rate limited, retrying in {wait_time}s... (attempt {attempt + 1}/{max_retries})")
                        time.sleep(wait_time)
                        continue

                # Non-rate-limit error or retries exhausted, raise exception
                if attempt == max_retries - 1:
                    raise Exception(f"LLM call failed after {max_retries} attempts: {e}")

                # Other errors are also retried, but with a shorter wait time
                wait_time = 2 ** attempt
                print(f"[MedPanelTool] API error, retrying in {wait_time}s... (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait_time)

        # Theoretically unreachable here
        raise Exception("LLM call failed: unexpected error")

    def _parse_answer(self, response: str, options: Dict[str, str]) -> str:
        """Parse the answer from the response

        Args:
            options: Dict[str, str] (e.g. {"A": "Yes", "B": "No"})
        """
        response_upper = response.upper()

        # Try to extract the **ANSWER:** format
        if "**ANSWER:**" in response_upper:
            answer_part = response_upper.split("**ANSWER:**")[1].strip()
            first_line = answer_part.split("\n")[0].strip()
            for key in options.keys():
                if first_line.startswith(key.upper()):
                    return key

        # Try to match a letter answer
        for key in sorted(options.keys()):
            patterns = [
                rf'\bAnswer:\s*{key}\b',
                rf'\b{key}\.\s',
                rf'\b{key}\)',
            ]
            for pattern in patterns:
                if re.search(pattern, response_upper):
                    return key

        # Default to returning the first option
        return list(options.keys())[0] if options else "A"

    def get_tool_schema(self) -> Dict[str, Any]:
        """Return the JSON Schema (for tool calling)"""
        return {
            "type": "function",
            "function": {
                "name": self.TOOL_NAME,
                "description": "Conduct a multi-expert panel discussion on medical imaging data. The tool will analyze longitudinal MRI data (Visit 1 vs Visit 2) and provide consensus clinical insights.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                            "description": "The clinical question to answer",
                        },
                        "options": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of answer options",
                        },
                    },
                    "required": ["question", "options"],
                },
            },
        }

    @property
    def tool_name(self) -> str:
        """Tool name"""
        return "medpanel_discuss"
