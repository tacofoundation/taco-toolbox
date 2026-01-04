# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

[0.26.1] - 2025-01-04
Fixed

Fixed NameError: name 'Environment' is not defined when jinja2 is not installed - moved type hint to string annotation in docs.py

## [0.26.0] - 2025-01-03

### Changed
- Removed `tacoreader` dependency from tacotoolbox
- Shared utilities (constants, schemas, validation) moved to new `tacobridge` package
- tacotoolbox and tacoreader now communicate through tacobridge as a lightweight bridge

### Added
- `tacobridge` package responsible for exporting and translating tasks.