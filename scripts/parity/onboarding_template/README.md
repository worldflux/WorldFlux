# Parity Onboarding Template

Use this template when adding a new model family to the universal parity framework.

## Required additions

1. Create a suite manifest from `suite_template.yaml`.
2. Implement a family plugin using `plugin_template.py`.
3. Add official/worldflux wrapper scripts that emit:
   - `final_return_mean`
   - `auc_return`
   - `metadata.policy_mode`
   - `metadata.policy_impl`
   - `metadata.eval_protocol_hash`
4. Register the plugin in `/Users/yoshi/Prod/worldflux/scripts/parity/suite_registry.py`.
5. Run smoke parity first, then full suite.

## Promotion gate

A suite is promotion-ready only when:

- `missing_pairs == 0`
- `validity_report.pass == true`
- `equivalence_report.global.parity_pass_final == true`
