# Contributing to OncoVision AI

Thank you for wanting to contribute to OncoVision AI! Here's how to collaborate smoothly.

## Getting Started

1. **Clone the repository:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/OncoVision-AI.git
   cd OncoVision-AI
   ```

2. **Set up your environment:**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   
   pip install -r requirements.txt
   ```

3. **Create a feature branch:**
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Workflow

### Before You Start
- Create an issue describing what you want to work on
- Discuss with team members
- Assign yourself to the issue

### Making Changes
1. Make your changes in your feature branch
2. Test locally
3. Commit with clear messages:
   ```bash
   git add .
   git commit -m "feat: add your feature description"
   ```

### Commit Message Format
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation
- `style:` Code style (no logic changes)
- `refactor:` Code restructuring
- `test:` Adding tests
- `chore:` Maintenance

Example: `feat: add patient ID input validation`

### Push and Create PR
1. Push your branch:
   ```bash
   git push origin feature/your-feature-name
   ```

2. Go to GitHub and create a **Pull Request**
3. Add description of changes
4. Request review from team members
5. Address review feedback
6. Once approved, merge to main

## Code Standards

- Follow PEP 8 for Python code
- Add docstrings to functions
- Test your code before pushing
- Update README if adding new features

## Testing

Run the app before pushing:
```bash
python app.py
```

Visit `http://127.0.0.1:5000` to test

## Need Help?

- Check existing issues for similar problems
- Ask in pull request comments
- Create a new issue for bugs

Thank you for contributing! 🚀
