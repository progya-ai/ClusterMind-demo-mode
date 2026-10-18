# Public demo on Render

ClusterMind normally runs locally with small AI models. The hosted demo runs the same code in **demo mode**
(`CM_DEMO=1`), which needs no AI model and no GPU:

| Works in the demo | Turned off in the demo |
|---|---|
| Dashboard with node cards and headline results | Ask (needs a local AI model) |
| Check answers (the real checker, any text you type) | Tasks, Files, downloads, uploads |
| Guarded action (keyword parser instead of the AI model; the real typed confirmation and busy-node block) | Shared audit log |
| Recorded runs (real model outputs, with the checker's verdicts) | Confirmation delays (they need the trace file) |

Demo mode also accepts any host name, limits each visitor to 90 POST requests a minute, and keeps no log of visitors' actions.

## 1. Push the project to a new GitHub repository

1. On github.com click **+ → New repository**. Name it (for example `clustermind-demo`), choose **Public**, and leave
   "Add a README", ".gitignore" and "license" **unticked**. Click **Create repository**.
2. In the VS Code terminal, inside the project folder (the one containing `README.md`, `cmapp/` and `render.yaml`):

```powershell
git init
git add .
git status
```

   `git status` should list files such as `render.yaml`, `requirements-demo.txt`, `cmapp/`, `data/` and `runs/`.
   It must **not** list any `.gguf` model file or `.env`. (The `.gitignore` already excludes them.)

```powershell
git commit -m "ClusterMind with public demo mode"
git branch -M main
git remote add origin https://github.com/<your-username>/clustermind-demo.git
git push -u origin main
```

   Replace `<your-username>` and the repository name with yours. The first push opens a GitHub sign-in window;
   approve it. If Git says it doesn't know who you are, run
   `git config --global user.name "Your Name"` and `git config --global user.email "you@example.com"`, then repeat the commit.
3. Refresh the repository page on GitHub and check that `README.md`, `render.yaml` and `cmapp/` are at the top level.

## 2. Deploy on Render

1. Go to render.com and sign up with your GitHub account.
2. Click **New + → Blueprint**, connect GitHub if asked, and choose your repository.
3. Render reads `render.yaml`. Confirm the service **clustermind-demo** (plan **Free**) and click **Apply**.
4. Wait for the build to finish (a few minutes). The service page shows its address, like
   `https://clustermind-demo.onrender.com` (Render adds letters if the name is taken).
5. Open the address. You should see the purple dashboard with a **Public demo** banner and only four tabs.
   `<address>/healthz` should show `{"ok": true}`.

**Without the Blueprint:** choose **New + → Web Service** instead, pick the repository, and set:

| Setting | Value |
|---|---|
| Language / Runtime | Python |
| Build command | `pip install -r requirements-demo.txt` |
| Start command | `python -m cmapp web --host 0.0.0.0 --port $PORT` |
| Instance type | Free |
| Health check path | `/healthz` |
| Environment variables | `CM_DEMO` = `1` and `PYTHON_VERSION` = `3.12.10` |

## 3. Put the links in the submission form

- **Demo / Live Links:** the Render address.
- **GitHub Repository:** the repository address.

## Notes

- **Free instances sleep.** Render's free web services spin down after about 15 minutes without visitors, and the first visit afterwards
  starts the service again, which takes a moment. Open the link yourself shortly before you submit and before any live judging.
- **Updating the demo.** After any change: `git add .`, `git commit -m "message"`, `git push`. Render redeploys automatically.
- **Plans and limits change.** Check Render's current free-plan terms before relying on them.

## Troubleshooting

| What you see | What to do |
|---|---|
| Build fails with "Python version" | In Render, open the service → **Environment** and change `PYTHON_VERSION` to a 3.12.x or 3.11.x version Render accepts, or delete the variable. |
| The page shows tabs like Ask, Tasks or Files | `CM_DEMO` is missing. Add `CM_DEMO` = `1` under **Environment** and redeploy. |
| "Host not allowed" | Same cause: `CM_DEMO` is not set to `1`. |
| Dashboard numbers are empty | The `data/` or `results/` folder was not pushed. Run `git status` and push again. |
| `git push` is rejected | The new repository must be empty. Create it without a README, or add `--force` only if you are sure. |
