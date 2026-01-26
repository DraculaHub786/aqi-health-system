# Streamlit Cloud Deployment Checklist

## Pre-Deployment Checklist

### Local Testing
- [ ] Run `streamlit run streamlit_app.py` locally
- [ ] Test all features work correctly
- [ ] No console errors or warnings
- [ ] App loads within reasonable time (< 30 seconds)

### Code Preparation
- [ ] Code is committed to Git
- [ ] Repository is on GitHub (public)
- [ ] `.streamlit/config.toml` is configured
- [ ] `.gitignore` excludes unnecessary files
- [ ] No API keys/secrets in code (use `st.secrets` instead)

### Dependencies
- [ ] All imports are in `requirements.txt`
- [ ] `requirements.txt` is tested with fresh pip install
- [ ] No development-only packages in `requirements.txt`
- [ ] Versions are compatible with each other

### Configuration Files
- [ ] `streamlit_app.py` exists in root directory
- [ ] `requirements.txt` exists in root directory
- [ ] `.streamlit/config.toml` is created
- [ ] `.streamlit/secrets.toml` is in `.gitignore` (not pushed)

### Kaggle Integration (if needed)
- [ ] Kaggle API credentials obtained from Kaggle account
- [ ] `kaggle_username` and `kaggle_key` ready
- [ ] Planned to add secrets to Streamlit Cloud

---

## Deployment Steps

### Step 1: Push to GitHub
```bash
cd your-repo
git add .
git commit -m "Prepare for Streamlit Cloud deployment"
git push origin main
```
- [ ] Verify code is on GitHub
- [ ] Check repository is public

### Step 2: Create Streamlit Cloud Account
- [ ] Go to https://share.streamlit.io
- [ ] Sign up with GitHub account
- [ ] Authorize Streamlit to access your repositories

### Step 3: Deploy App
- [ ] Click "New app" button
- [ ] Select repository: `your-username/aqi-health-system`
- [ ] Select branch: `main`
- [ ] Enter main file path: `streamlit_app.py`
- [ ] Click "Deploy"

### Step 4: Add Secrets (if needed)
- [ ] Wait for app to build and deploy
- [ ] Go to app settings (gear icon)
- [ ] Click "Secrets" tab
- [ ] Add Kaggle credentials:
  ```
  kaggle_username = "your_username"
  kaggle_key = "your_api_key"
  ```
- [ ] Click "Save"
- [ ] App automatically redeploys

### Step 5: Verify Deployment
- [ ] App loads successfully
- [ ] All features work correctly
- [ ] Check logs for any errors
- [ ] Test with sample data

---

## Post-Deployment

### Share Your App
- [ ] Copy the app URL from the address bar
- [ ] Share with friends/colleagues
- [ ] Example: `https://yourname-aqi-health-system-xxxxx.streamlit.app`

### Monitor
- [ ] Check app settings periodically
- [ ] Monitor logs for errors
- [ ] Watch memory usage

### Update
- [ ] Make changes to code locally
- [ ] Push to GitHub (`git push`)
- [ ] App automatically redeploys (usually within 1-2 minutes)

---

## Troubleshooting Checklist

If deployment fails:

- [ ] Check build logs for specific errors
- [ ] Verify all dependencies are in `requirements.txt`
- [ ] Check that `streamlit_app.py` is in root directory
- [ ] Verify no secrets/API keys are exposed in code
- [ ] Test locally first: `streamlit run streamlit_app.py`
- [ ] Clear build cache and reboot app

If app runs but features don't work:

- [ ] Check Streamlit Cloud logs
- [ ] Add secrets if API credentials needed
- [ ] Verify model files are accessible
- [ ] Check data file paths are correct
- [ ] Test with reduced dataset if memory issues

---

## Quick Reference

| Task | Command/Link |
|------|--------------|
| Test Locally | `streamlit run streamlit_app.py` |
| View App | `https://share.streamlit.io` |
| Upload Code | `git push origin main` |
| Share App | Copy URL from browser address bar |
| View Logs | Settings → Logs tab |
| Add Secrets | Settings → Secrets tab |
| Reboot App | Settings → Reboot app |

---

## Success Indicators

✅ App deployed when you see:
- Build status: "Successfully built"
- App status: "Running"
- Browser can load the app without errors
- All interactive features work

🎉 Your AQI Health System is now live on the internet!
