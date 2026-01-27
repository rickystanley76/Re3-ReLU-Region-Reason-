# Contributing to Re3: ReLU Region Reason

Thank you for your interest in contributing to the Re3 project! This document provides guidelines and instructions for contributing.

## Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Focus on what is best for the community
- Show empathy towards other community members

## How to Contribute

### Reporting Bugs

If you find a bug, please create an issue with:

1. **Clear title and description**
2. **Steps to reproduce** the bug
3. **Expected behavior** vs **actual behavior**
4. **Environment details**:
   - Python version
   - Operating system
   - Package versions (from `pip list` or `uv pip list`)
5. **Screenshots or error messages** (if applicable)

### Suggesting Features

Feature suggestions are welcome! Please create an issue with:

1. **Clear description** of the feature
2. **Use case** - why this feature would be useful
3. **Possible implementation** approach (if you have ideas)

### Pull Requests

1. **Fork the repository**
2. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Make your changes**
4. **Test your changes** thoroughly
5. **Update documentation** if needed
6. **Commit with clear messages**:
   ```bash
   git commit -m "Add feature: description of what you added"
   ```
7. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```
8. **Create a Pull Request** on GitHub

## Development Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/YOUR_USERNAME/Re3-ReLU-Region-Reason.git
   cd Re3-ReLU-Region-Reason
   ```

2. **Install dependencies**:
   ```bash
   uv pip install -r requirments.txt
   # or
   pip install -r requirments.txt
   ```

3. **Test the application**:
   ```bash
   streamlit run app.py
   ```

## Coding Standards

### Python Style

- Follow **PEP 8** style guide
- Use meaningful variable and function names
- Add docstrings to functions and classes
- Keep functions focused and small
- Add comments for complex logic

### Code Formatting

Consider using:
- **Black** for code formatting
- **flake8** or **pylint** for linting
- **mypy** for type checking (optional)

### Documentation

- Update relevant documentation files:
  - `APP_README.md` - Main application documentation
  - `QUICK_START_GUIDE.md` - Quick start guide
  - Inline code comments for complex logic

## Testing

Before submitting a PR:

1. **Test your changes** with different datasets
2. **Test edge cases** (empty data, missing columns, etc.)
3. **Ensure no errors** in the Streamlit app
4. **Check visualizations** render correctly
5. **Test with different model architectures** (1-3 layers)

## Commit Message Guidelines

Follow these guidelines for commit messages:

- Use imperative mood ("Add feature" not "Added feature")
- First line should be concise (50 chars or less)
- Add detailed description if needed (separated by blank line)
- Reference issues if applicable: "Fix #123"

**Good examples:**
```
Add support for Diabetes dataset in quick_start.py

- Added load_diabetes_dataset() function
- Configured appropriate model architecture
- Updated documentation
```

```
Fix IndexError in class name indexing (#45)

Added bounds checking to prevent IndexError when accessing
class_names with sparse class indices.
```

## Project Structure

```
.
├── app.py                    # Main Streamlit application
├── re3_core.py              # Core Re3 computation functions
├── quick_start.py           # Model training script
├── requirments.txt          # Python dependencies
├── .gitignore              # Git ignore rules
├── APP_README.md           # Application documentation
├── QUICK_START_GUIDE.md    # Quick start guide
├── CONTRIBUTING.md         # This file
└── README.md               # Project overview
```

## Areas for Contribution

We welcome contributions in these areas:

- **Bug fixes**: Fixing existing issues
- **New features**: Adding new functionality
- **Documentation**: Improving guides and docs
- **Performance**: Optimizing code
- **Testing**: Adding test cases
- **UI/UX**: Improving the Streamlit interface
- **Examples**: Adding example notebooks or use cases

## Questions?

If you have questions:

- Open an issue for discussion
- Contact: arnab.barua@mdu.se
- Check existing issues and discussions

## License

By contributing, you agree that your contributions will be licensed under the same license as the project.

Thank you for contributing to Re3! 🎉
