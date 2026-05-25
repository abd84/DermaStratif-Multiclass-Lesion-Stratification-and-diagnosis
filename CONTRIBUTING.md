# Contributing to DermaStratif

We welcome contributions! Here's how to get started:

## Development Setup

1. **Fork and Clone**
   ```bash
   git clone https://github.com/your-username/DermaStratif.git
   cd DermaStratif-Multiclass-Lesion-Stratification-and-diagnosis
   ```
 
2. **Create Virtual Environment**
   ```bash
   python3.11 -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Create Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Code Style

- Follow PEP 8 guidelines
- Use meaningful variable names
- Add docstrings to functions
- Comment complex logic

## Submitting Changes

1. Commit with clear messages:
   ```bash
   git commit -m "Add feature: description of changes"
   ```

2. Push to your branch:
   ```bash
   git push origin feature/your-feature-name
   ```

3. Create a Pull Request with:
   - Clear description of changes
   - References to related issues
   - Screenshots if UI changes

## Testing

Before submitting:
- Test the application locally
- Verify no breaking changes
- Check error handling

## Bug Reports

Include:
- Environment details (OS, Python version)
- Steps to reproduce
- Expected vs actual behavior
- Error messages/logs
