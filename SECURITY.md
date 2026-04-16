# Security

Orbital Engagement Lab is research and prototyping software. It is not
flight-qualified software and should not be used as an operational decision
system without independent mission-specific validation.

## Reporting A Vulnerability

Please report suspected security issues privately to the repository maintainers
rather than opening a public issue.

Include:

- a short description of the issue
- steps to reproduce
- affected versions or commits, if known
- any relevant logs or proof-of-concept details

## Supported Scope

Security reports are most useful when they affect:

- arbitrary code execution through scenario loading or plugin pointers
- unsafe handling of local files or paths
- dependency or packaging risks
- integration surfaces such as cFS/SIL adapters

Simulation-model correctness issues are also welcome, but they may be handled as
engineering bugs rather than security vulnerabilities unless they create a clear
security impact.
