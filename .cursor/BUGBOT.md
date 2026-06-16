# BugBot Instructions

## Changelog Enforcement

Any PR that introduces **breaking** configuration changes must update `CHANGELOG.md`. Breaking changes are those that require users to update existing configs:

- **Renamed** config fields (old name no longer accepted)
- **Removed** config fields (field deleted or moved to a different path)
- **Moved** config fields (field relocated in the config hierarchy)

Additive changes (new fields with defaults, new optional features) and default value changes do **not** require a changelog entry.

Config files live in:

- `src/prime_rl/configs/`

If breaking changes are detected without a corresponding `CHANGELOG.md` update, request that the author add an entry.
