# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html) while following the version scheme described in the [PyPA version specifiers](https://packaging.python.org/en/latest/specifications/version-specifiers/#version-specifiers) document, which is based on PEP440.

## Unreleased

### Breaking Changes

- Remove `Shape` type alias. Use built-in `tuple` instead. ([40977a0](https://github.com/MilanStaffehl/numdantic/commit/40977a0633c5a6e564306853bc9ab91a769c727c))

### Added

- Support for arrays of indeterminate dimensionality with `tuple[int, ...]` ([9fb88d4](https://github.com/MilanStaffehl/numdantic/commit/9fb88d49012974f6b099a8e8ef29ec3346c6a4f1))
- Support for arrays of indeterminate dimensionality with fixed axis length with `tuple[Literal[n], ...]` ([9fb88d4](https://github.com/MilanStaffehl/numdantic/commit/9fb88d49012974f6b099a8e8ef29ec3346c6a4f1))
- Support for arrays of indeterminate dimensionality with equal axes but variable axes length with `NewType` named axes and ellipsis ([9fb88d4](https://github.com/MilanStaffehl/numdantic/commit/9fb88d49012974f6b099a8e8ef29ec3346c6a4f1))
- Commit SHA to all `CHANGELOG` entries ([?]())

### Changed

- Bind shape type parameter of `NDArray` to `tuple[int, ...]` instead of unparametrized type alias of `tuple` ([d6328f4](https://github.com/MilanStaffehl/numdantic/commit/d6328f49179c8a0c30862a28ab11fd4299f48453))

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
- Documentation as part of the `README` ([e500a0b](https://github.com/MilanStaffehl/numdantic/commit/e500a0bc3d75a260a40c45ca2e0e414b4654de81), [56ff3ae](https://github.com/MilanStaffehl/numdantic/commit/56ff3aec879415760fa0341b8740a9e8440a89c9), [311ca20](https://github.com/MilanStaffehl/numdantic/commit/311ca20ccd19ffaeef42729a2262835b73bdb46e), [adece74](https://github.com/MilanStaffehl/numdantic/commit/adece74c8291488eb659e9a11262bfb6817cab58), [8aa5e1d](https://github.com/MilanStaffehl/numdantic/commit/8aa5e1d2439c5a06f9a19a285c4b86a004c9a35d), [313728e](https://github.com/MilanStaffehl/numdantic/commit/313728eba0bbadd19e998262c7d781cd0cafe7a0))
- `CHANGELOG` ([e19393c](https://github.com/MilanStaffehl/numdantic/commit/e19393c4582a19964d109f0c373cf85076a120e1))
