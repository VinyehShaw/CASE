"""
MedPanel — Multi-expert joint diagnostic discussion tool

Wraps the multi-expert discussion flow as an distill tool, supporting trajectory recording and SFT data export.
"""

from distill.tool.common.medpanel_tool import MedPanelTool
from distill.tool.common.trajectory import PanelTrajectory

__all__ = ["MedPanelTool", "PanelTrajectory"]
