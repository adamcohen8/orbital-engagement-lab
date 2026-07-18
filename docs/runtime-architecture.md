# Runtime Architecture

`sim.runtime_support` remains OEL's compatibility façade for runtime records,
state initialization, object factories, mission dispatch, and command helpers.
New code should locate implementation ownership through
`RUNTIME_CONSTRUCTION_FAMILIES` in `sim.runtime.architecture`:

- `models.py`: `AgentRuntime` and rate-limited controller state
- `compat.py`: plugin construction and call-signature compatibility
- `state_initialization.py`: Cartesian, COE, TLE, CR3BP, and relative states
- `actuator_factory.py`: propulsion, actuator stacks, and mass properties
- `satellite_factory.py`: satellite dynamics, estimation, control, and runtime construction
- `rocket_factory.py`: rocket stacks, guidance, ascent configuration, and runtime construction
- `knowledge_factory.py`: tracked-object knowledge and EKF configuration
- `mission_runtime.py`: deployment, mission modules, strategy, and execution
- `commands.py`: command serialization and decision-state views

`sim.single_run._SingleRunEngine` remains the stable lifecycle coordinator.
Its focused collaborators are:

- `sim.execution.runtime_profile`: runtime timing and profile payloads
- `sim.execution.object_workers`: persistent process-worker transport
- `sim.execution.object_step_coordinator`: serial/parallel planning and executor selection
- `sim.execution.single_run_history`: history growth, retention, and compaction
- `sim.reporting.run_payload_assembly`: reporting views, payload construction, and artifact dispatch

External callers should continue using `sim.execution`, `sim.api`, or the
existing compatibility modules. These implementation modules are ownership
boundaries, not a replacement public API.
