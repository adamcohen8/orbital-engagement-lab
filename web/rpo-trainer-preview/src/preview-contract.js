function deepFreeze(value) {
  if (!value || typeof value !== "object" || Object.isFrozen(value)) return value;
  Object.values(value).forEach(deepFreeze);
  return Object.freeze(value);
}

export const PREVIEW_LEVEL_CONTRACTS = deepFreeze({
  tutorial: {
    title: "Level 0 - Pilot Tutorial",
    operator_title: "Level 0 - Operator Tutorial",
    max_time_s: 18000.0,
    max_delta_v_m_s: 12.0,
    goal_range_km: 0.25,
    max_goal_speed_km_s: 0.0003,
    guided_burn_delta_v_m_s: 0.25,
    guided_speed_multiplier: 10.0,
    learning_goal:
      "Learn what R, I, and C mean by creating six small target orbits, then use short pulse-and-coast translations to settle near a passive target.",
    player_brief:
      "The yellow satellite is you. R is radial, toward or away from Earth through the target. I is in-track, forward or backward along the target's orbit. C is cross-track, out of the target's orbital plane. For each guided stage, the simulation pauses until you hold the requested control. Follow the green path to build the shown 0.25 m/s burn, then the trainer resets for the next axis before the final approach.",
    pass_criteria: [
      "Complete the +I and -I guided orbit demonstrations.",
      "After +I, increase the speed multiple to 10x.",
      "Complete the +R and -R guided orbit demonstrations.",
      "Complete the +C and -C guided orbit demonstrations.",
      "Get within 250 m of the passive target below 0.3 m/s.",
      "Stay under the generous tutorial time and delta-v budgets.",
    ],
    instructor_notes: [
      "This level teaches the controls before introducing natural-motion matching, keepout constraints, or target evasion.",
      "Encourage short pulses followed by coasting rather than continuous thrust.",
      "The RI view shows in-track versus radial motion; the RC view shows cross-track versus radial motion.",
    ],
    scope:
      "Browser preview of downloadable Level 0. The full trainer adds the complete scenario-backed catalog, difficulty and progress tracking, recordings, and full debriefs.",
  },
  sandbox: {
    title: "Reduced Circular-Orbit Sandbox",
    operator_title: "Reduced Operator Sandbox",
    max_time_s: 20000.0,
    scope:
      "Reduced circular-orbit browser sandbox. The downloadable Sandbox also edits the target orbit and eccentricity and supports elliptical prediction.",
  },
  pursuit_arcade: {
    title: "Pursuit Arcade",
    web_only: true,
    scope:
      "Web-only competition mode. Pursuit Arcade is not included in the downloadable launcher; its browser-native replays are validated independently.",
  },
});
