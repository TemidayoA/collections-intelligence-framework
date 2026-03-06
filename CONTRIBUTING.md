# Contributing to Collections Intelligence Framework

Thank you for your interest in contributing to this project. This framework aims to transform debt collection practices from harassment-based to intelligence-based approaches.

## How to Contribute

### Reporting Issues

If you find a bug or have a feature suggestion:

1. Check existing issues to avoid duplicates
2. Open a new issue with a clear title and description
3. Include steps to reproduce (for bugs)
4. Add relevant labels

### Pull Requests

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass (`pytest tests/`)
6. Commit with clear messages
7. Push and open a pull request

### Code Style

- Follow PEP 8 guidelines
- Use type hints where appropriate
- Include docstrings for public functions and classes
- Keep functions focused and modular

### Testing

All new features should include tests:

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

### Documentation

- Update relevant documentation for new features
- Include docstrings with examples
- Update README if adding new modules

## Areas for Contribution

### High Priority

- **Additional propensity model architectures**: Neural networks, ensemble methods
- **Channel optimisation**: Multi-armed bandit approaches for channel selection
- **Real-time scoring**: Streaming implementations for live decisioning
- **Synthetic data generation**: Better test data for development

### Medium Priority

- **Visualisation**: Dashboard components for monitoring
- **Integration examples**: Connecting to common CRM/collections platforms
- **Benchmarking**: Performance comparisons with baseline approaches

### Documentation

- Case studies and examples
- Integration guides
- API documentation improvements

## Code of Conduct

- Be respectful and constructive
- Focus on the technical merits of contributions
- Help maintain a welcoming environment for all contributors

## Questions?

Open an issue with the `question` label or reach out to the maintainers.

---

By contributing, you agree that your contributions will be licensed under the MIT License.
