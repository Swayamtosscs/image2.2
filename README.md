# AI Fashion Ad Studio - Deployment Guide

This project is a high-end AI Fashion Ad Generator built with Streamlit and Google Gemini.

## Local Development

1. Create a `.env` file in the root directory.
2. Add the required environment variables (see below).
3. Install dependencies: `pip install -r requirements.txt`
4. Run the app: `streamlit run Home.py`

## Environment Variables

The app requires the following variables (set in `.env` for local dev, or as container env vars for deployment):

| Variable        | Required | Description |
|-----------------|----------|-------------|
| `GEMINI_API_KEY` | ✅ Yes  | Your Google Gemini API key |
| `JWT_SECRET`    | ✅ Yes   | Long random secret string used to sign auth tokens. **Change before deploying.** |
| `AUTH_USERS`    | ✅ Yes   | JSON map of `username → password`. Example: `{"admin":"mypassword","designer":"pass2"}` |

> ⚠️ **Before deploying**, update `JWT_SECRET` to a long random string and set secure credentials in `AUTH_USERS`.

## Authentication

The app is protected by JWT-based login. Users must sign in with a valid username and password before accessing any page. Sessions are valid for **8 hours**. A **Logout** button is available in the sidebar on every page.

## Docker Deployment

The app is containerized for easy deployment.

### 1. Build the Image
```bash
docker build -t fashion-ad-app .
```

### 2. Run the Container
Provide all required environment variables via `--env-file`:
```bash
docker run -p 8501:8501 --env-file .env fashion-ad-app
```

Or pass them individually:
```bash
docker run -p 8501:8501 \
  -e GEMINI_API_KEY=your_key_here \
  -e JWT_SECRET=your_long_random_secret \
  -e AUTH_USERS='{"admin":"yourpassword"}' \
  fashion-ad-app
```

The app will be available at `http://localhost:8501`.

### Server Configuration
Ensure all three environment variables above are set in your hosting environment (AWS, GCP, Azure, etc.). If `GEMINI_API_KEY` is missing the app will show a configuration error; if `JWT_SECRET` or `AUTH_USERS` are missing, default insecure values will be used — **always set them explicitly in production**.
