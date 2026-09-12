# README design policy

## Purpose

Every public README must let a reader with basic Python knowledge answer four
questions quickly: what problem Veridist solves, whether it fits their work,
how to run a first example, and where to go next.

## Information order

1. Describe the user problem and the intended reader in plain language.
2. Show the published install command and clear links to the first example and
   package guide.
3. Provide a complete, executable example with its expected output.
4. Explain what that output means and state the decision-relevant assumption.
5. Route readers by task, then introduce local resume and production boundaries.
6. Put detailed contracts, architecture records, and tests behind task-focused
   documentation links.

## Language and claims

Use short sentences and define a statistical term on first use. Do not lead
with internal terms such as "candidate-bound", "public contract", or "MLE
cell". A successful calculation is not evidence that a model is adequate.

Only make scale, performance, and quality claims supported by retained evidence.
The coverage badge reports the pass/fail state of the CI gate enforcing at least
95% line and branch coverage; it is not an exact coverage percentage.

## Localization and accessibility

English, Persian, and German pages must offer the same supported workflows,
claims, installation path, executable first example, output, and next-step
links. Persian prose is wrapped in RTL markup; code and output stay outside it.

## Verification

README changes must execute the shown examples, validate local links and
anchors, preserve translation parity, and check Persian RTL fences. Reviewers
should inspect the GitHub and PyPI renderings before release. A five-minute
first-success target guides the design; it is not a measured user-study claim.
