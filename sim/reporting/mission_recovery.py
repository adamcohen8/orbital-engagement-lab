"""Compatibility imports for the mission-recovery analysis service.

Mission planning and propagation live in :mod:`sim.analysis`; reporting only
formats or visualizes the resulting analysis product.
"""

from sim.analysis.mission_recovery import build_mission_recovery_summary, write_mission_recovery_trade_space_plot

__all__ = ["build_mission_recovery_summary", "write_mission_recovery_trade_space_plot"]
