# Quick Collaboration Guide

## For Your Friend

### 1️⃣ Accept GitHub Invite
- You'll get an email from GitHub with an invitation link
- Click the link and accept

### 2️⃣ Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/OncoVision-AI.git
cd OncoVision-AI
```

### 3️⃣ Set Up Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate  # Windows
# or
source venv/bin/activate  # Mac/Linux
pip install -r requirements.txt
```

### 4️⃣ Create Your Feature Branch
```bash
git checkout -b feature/my-awesome-feature
```

### 5️⃣ Make Changes & Commit
```bash
# Edit files...
git add .
git commit -m "feat: add my awesome feature"
git push origin feature/my-awesome-feature
```

### 6️⃣ Create a Pull Request (PR)
- Go to GitHub repository
- Click "Pull requests" → "New pull request"
- Select your branch
- Add description
- Click "Create pull request"

### 7️⃣ Wait for Review & Merge
- Owner reviews your changes
- You make requested changes (if any)
- PR gets merged to main

---

## Common Commands

| Task | Command |
|------|---------|
| Create branch | `git checkout -b feature/name` |
| See your branches | `git branch -a` |
| Switch branch | `git checkout branch-name` |
| Update from main | `git pull origin main` |
| See changes | `git status` |
| Check commit history | `git log --oneline` |
| Undo changes | `git checkout -- file.py` |

---

## Project Structure

```
OncoVision-AI/
├── app.py                 # Flask application
├── models/               # EDCNN & U-Net models
├── utils/                # Helper functions
├── templates/            # HTML templates
├── static/               # CSS & JavaScript
└── results/              # Trained models & results
```

---

## What Can Your Friend Work On?

- 🎨 **Frontend:** Improve UI/UX in templates/
- 🔧 **Backend:** Add new API endpoints
- 📊 **Features:** New analysis functions
- 📚 **Documentation:** Improve README & docs
- 🐛 **Bugs:** Fix issues

---

## Need Help?
- Check `.github/CONTRIBUTING.md` for detailed guidelines
- Ask questions in PR comments
- Create an issue for bugs or suggestions

Happy coding! 🚀
