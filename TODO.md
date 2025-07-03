# Attention Motifs - Comprehensive TODO List

This document contains a prioritized list of improvements for the attention-motifs repository, organized by category and priority level.

## 🔴 HIGH PRIORITY

### Code Quality & Architecture

#### Fix Critical Bugs
- **[BUG]** Fix matrix operations with integer matrices in `attn_embed/math/matrix_powers.py:80`
  - Currently breaks with integer input matrices
  - Add proper type conversion or validation
  - Update type hints to accept both integer and float matrices

#### Improve Package Structure
- **Add proper `__init__.py` exports** - All `__init__.py` files are currently empty
  - Define clear public APIs for each module
  - Enable clean imports like `from attn_embed.features import compute_features`
  - Add `__all__` declarations to control exports

#### Error Handling & Validation
- **Add comprehensive input validation** throughout the codebase
  - Use `beartype` decorators for runtime type checking
  - Validate tensor shapes and data ranges in feature computation functions
  - Add proper error messages for common failure modes

#### Security & Dependencies
- **Review and update dependencies** in `pyproject.toml`
  - Multiple outdated packages (check for security vulnerabilities)
  - Pin exact versions for reproducibility
  - Consider removing unused dependencies
  - Update `torch>=2.5.1` to latest stable version
  - Review custom dependencies (`muutils`, `pattern-lens`) for maintenance burden

### Performance & Scalability

#### Optimize Feature Computation
- **Implement GPU acceleration** for feature computation
  - Move statistical computations to PyTorch where possible
  - Parallelize feature extraction across attention heads
  - Add memory-efficient batch processing for large datasets

#### Improve I/O Performance
- **Optimize data loading and saving**
  - Use more efficient serialization formats (e.g., HDF5, Parquet)
  - Implement lazy loading for large datasets
  - Add compression for attention pattern storage

### Testing & Quality Assurance

#### Expand Test Coverage
- **Current coverage: 70%** - Target: 85%+
  - Add integration tests for the full pipeline
  - Test edge cases (empty inputs, malformed data)
  - Add property-based testing for mathematical functions
  - Test GPU/CPU compatibility

#### Add Performance Benchmarks
- **Create benchmark suite** for critical functions
  - Feature computation performance
  - Memory usage tracking
  - Regression testing for performance changes

## 🟡 MEDIUM PRIORITY

### Documentation & Usability

#### API Documentation
- **Generate comprehensive API documentation**
  - Add detailed docstrings following NumPy/Google style
  - Create API reference using existing pdoc setup
  - Document all configuration options and their effects

#### User Experience
- **Improve configuration management**
  - Add configuration validation and helpful error messages
  - Create configuration templates for common use cases
  - Add CLI help and examples

#### Examples & Tutorials
- **Create comprehensive examples**
  - Add tutorial notebooks for common workflows
  - Document best practices for different model types
  - Create troubleshooting guide

### Code Organization

#### Module Structure
- **Refactor large modules** for better maintainability
  - Split `features/features.py` into smaller, focused modules
  - Separate computation from I/O operations
  - Improve separation of concerns

#### Configuration System
- **Enhance configuration flexibility**
  - Support environment variable overrides
  - Add profile-based configurations (dev/prod)
  - Implement configuration validation schemas

### Development Workflow

#### Development Tools
- **Improve development experience**
  - Add pre-commit hooks configuration
  - Set up automated code formatting (already has ruff)
  - Add type checking in CI/CD pipeline

#### Build System
- **Enhance Makefile and build process**
  - Add development environment setup automation
  - Improve dependency management workflow
  - Add automated testing on different Python versions

## 🟢 LOW PRIORITY

### Features & Enhancements

#### Visualization Improvements
- **Enhance frontend visualizations**
  - Add more interactive features to attention pattern viewer
  - Improve mobile compatibility
  - Add data export functionality from web interface

#### Analysis Features
- **Extend analysis capabilities**
  - Add more statistical features for attention patterns
  - Implement clustering algorithms for attention heads
  - Add temporal analysis for attention evolution

#### Integration & Compatibility
- **Expand model support**
  - Add support for more transformer architectures
  - Improve handling of different tokenization schemes
  - Add compatibility with different attention mechanisms

### Maintenance & Cleanup

#### Code Cleanup
- **Remove deprecated code**
  - Clean up old/unused frontend versions
  - Remove debug print statements
  - Consolidate duplicate functionality

#### Documentation Maintenance
- **Keep documentation current**
  - Update README with latest features
  - Maintain accurate setup instructions
  - Update dependency requirements

## 📊 Existing TODOs in Codebase

### From Inline Comments
- **[HACK]** Add metadata properly in `attn_embed/features/head_analysis.py:195`
- **[TODO]** Speed up feature computation via torch computing
- **[TODO]** Add all additional features once computation is faster
- **[TODO]** Implement wasserstein or similar matching over prompts for distance matrix
- **[TODO]** Option to show attention pattern on hover in 3D embeddings
- **[TODO]** Add layer_depth column as float 0-1 for different model layer counts
- **[TODO]** Support up to 3 scalar columns via RGB color mapping

### From README TODO Section
- ✅ **Completed**: Add `activation.n_ctx` to feature set
- ✅ **Completed**: Figure out most important features
- ✅ **Completed**: Distance matrix between heads via naive point-to-point dist
- ✅ **Completed**: TSNE/UMAP over distance matrix to find clusters
- ✅ **Completed**: Fix colors matching between bottom bar and plot
- ✅ **Completed**: Default coloring by scalar column with colormap
- ✅ **Completed**: Saving selections/config in URL
- ✅ **Completed**: Throw in papers to AttentionPedia
- 🔄 **In Progress**: Speed up feature computation via torch
- 🔄 **In Progress**: Add all other features once faster
- ❌ **Pending**: Wasserstein matching over all prompts
- ❌ **Pending**: Attention pattern on hover in 3D embeddings
- ❌ **Pending**: RGB mapping for up to 3 scalar columns
- ❌ **Pending**: Layer depth normalization across models

## 🎯 Implementation Priorities

### Phase 1: Critical Fixes (1-2 weeks)
1. Fix integer matrix bug
2. Add proper `__init__.py` exports
3. Implement basic input validation
4. Update critical security dependencies

### Phase 2: Quality & Testing (2-3 weeks)
1. Expand test coverage to 85%+
2. Add comprehensive error handling
3. Implement performance benchmarks
4. Add API documentation

### Phase 3: Performance & Features (3-4 weeks)
1. GPU acceleration for feature computation
2. Optimize I/O operations
3. Enhanced configuration system
4. Additional analysis features

### Phase 4: Polish & Maintenance (1-2 weeks)
1. Code cleanup and refactoring
2. Documentation updates
3. Frontend improvements
4. Additional model support

## 📝 Notes

- **Test Coverage**: Current coverage is 70%, with several modules having low coverage (especially `figure_funcs.py` at 0%)
- **Dependencies**: Heavy reliance on custom packages (`muutils`, `pattern-lens`) may complicate maintenance
- **Configuration**: Well-designed TOML-based configuration system, but could benefit from validation
- **Architecture**: Generally good separation of concerns with clear pipeline structure
- **Type Safety**: Excellent use of `jaxtyping` for tensor shape annotations

## 🚀 Quick Wins

For immediate impact, focus on:
1. Adding `__init__.py` exports for better import experience
2. Fixing the integer matrix bug
3. Adding input validation to prevent common errors
4. Updating outdated dependencies
5. Adding docstrings to core functions

This TODO list should be regularly updated as items are completed and new requirements emerge.