# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html) while following the version scheme described in the [PyPA version specifiers](https://packaging.python.org/en/latest/specifications/version-specifiers/#version-specifiers) document, which is based on PEP440.

## Unreleased

### Added

-

## [0.2.0] - 2025-08-22

### Breaking Changes

- Remove `Shape` type alias. Use built-in `tuple` instead. ([40977a0](https://github.com/MilanStaffehl/numdantic/commit/40977a0633c5a6e564306853bc9ab91a769c727c))

### Added

- Support for arrays of indeterminate dimensionality with `tuple[int, ...]` ([9fb88d4](https://github.com/MilanStaffehl/numdantic/commit/9fb88d49012974f6b099a8e8ef29ec3346c6a4f1))
- Support for arrays of indeterminate dimensionality with fixed axis length with `tuple[Literal[n], ...]` ([9fb88d4](https://github.com/MilanStaffehl/numdantic/commit/9fb88d49012974f6b099a8e8ef29ec3346c6a4f1))
- Support for arrays of indeterminate dimensionality with equal axes but variable axes length with `NewType` named axes and ellipsis ([9fb88d4](https://github.com/MilanStaffehl/numdantic/commit/9fb88d49012974f6b099a8e8ef29ec3346c6a4f1))
- Support and documentation for `numpy` 2.1+ ([#65](https://github.com/MilanStaffehl/numdantic/pull/65), [#77](https://github.com/MilanStaffehl/numdantic/pull/77), [#78](https://github.com/MilanStaffehl/numdantic/pull/78))
- Support for Python 3.13 ([#70](https://github.com/MilanStaffehl/numdantic/pull/70), [#85](https://github.com/MilanStaffehl/numdantic/pull/85))
- Commit SHA or pull request number to all `CHANGELOG` entries ([9787566](https://github.com/MilanStaffehl/numdantic/commit/9787566fb3f282e213383566a6d7da844829d703))
- Section about shapes of indeterminate dimensionality to `README` ([5f2a7e4](https://github.com/MilanStaffehl/numdantic/commit/5f2a7e43fae0e918c0b31d65995f0f1a9cc8d67d))

### Changed

- Bind shape type parameter of `NDArray` to `tuple[int, ...]` instead of unparametrized type alias of `tuple` ([d6328f4](https://github.com/MilanStaffehl/numdantic/commit/d6328f49179c8a0c30862a28ab11fd4299f48453))
- Sequences are now rejected in strict mode: passing a sequence instead of an array raises a `ValidationError` ([#90](https://github.com/MilanStaffehl/numdantic/pull/90))

## [0.1.1] - 2024-07-05

### Added

- Note in `CHANGELOG` about version 0.1.0 not being available on PyPI ([69b2794](https://github.com/MilanStaffehl/numdantic/commit/69b27948b87f2b599442396a650055557dc461d2))

### Fixed

- Test failures due to repeated variable names in type checking ([3700450](https://github.com/MilanStaffehl/numdantic/commit/3700450547f719a33d97157e7172e7ac7c95e8eb))

## [0.1.0] - 2024-07-05

> [!IMPORTANT]
>
> Due to a naming conflict, this version is not available on PyPI. Use version 0.1.1 instead, which is functionally identical to this version.

### Added

- Typing support for `numpy` arrays ([a59d57d](https://github.com/MilanStaffehl/numdantic/commit/a59d57dd1a2bfc153be694b8bd3953f9bb55715d))
- Support for validation of `numpy` arrays with `pydantic` ([a59d57d](https://github.com/MilanStaffehl/numdantic/commit/a59d57dd1a2bfc153be694b8bd3953f9bb55715d))
- Helper script to determine local dtype compatibility ([7b63ac4](https://github.com/MilanStaffehl/numdantic/commit/7b63ac45766c13d660786d7b5b8e189a4b4bab3c))
- Documentation as part of the `README` ([#1](https://github.com/MilanStaffehl/numdantic/pull/1), [#13](https://github.com/MilanStaffehl/numdantic/pull/13), [#15](https://github.com/MilanStaffehl/numdantic/pull/15), [adece74](https://github.com/MilanStaffehl/numdantic/commit/adece74c8291488eb659e9a11262bfb6817cab58), [8aa5e1d](https://github.com/MilanStaffehl/numdantic/commit/8aa5e1d2439c5a06f9a19a285c4b86a004c9a35d), [#29](https://github.com/MilanStaffehl/numdantic/pull/29))
- `CHANGELOG` ([#37](https://github.com/MilanStaffehl/numdantic/pull/37))
