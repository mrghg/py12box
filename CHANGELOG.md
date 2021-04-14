# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]
Use this section to keep track of any intermediate changes that can then be moved to a versioned section.

## [0.2.0] - 2021-XX-XX
### Added
- Instantaneous mole fraction is now output for the end of each month.
Note that this is a breaking change, as one more return value is added to the 
core model output tuple
- Methods to change the start and end year of the simulation: 
Model.change_start_year and Model.change_end_year

### Changed
- Due to the restart output being added, an additional return value is added
to core.model

### Removed
- N/A
