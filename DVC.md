# DVC Google Drive Setup (Docker)

## Prerequisites

- Google account  
- Docker environment with Python 3.10+  

---

## Step 1: Create Google OAuth Credentials

1. Go to Google Cloud Console: https://console.cloud.google.com/  
2. Create a new project (e.g., `"DVC Storage"`).  
3. Navigate to **APIs & Services → Enable APIs and Services**.  
4. Search for **Google Drive API** and enable it.  
5. Configure OAuth consent screen:  
   - User Type: External  
   - App name: `"DVC"`  
   - Add your email as developer contact  
   - Add yourself as a test user  

6. Go to **Credentials → Create Credentials → OAuth client ID**:  
   - Application type: Desktop app  
   - Name: `"DVC Client"`  

7. Download the JSON file and note your `client_id` and `client_secret`.  

---

## Step 2: Create Google Drive Folder

1. Go to Google Drive: https://drive.google.com/  
2. Create a folder (e.g., `"dvc-storage"`).  
3. Open the folder and copy the folder ID from the URL (inline): `https://drive.google.com/drive/folders/FOLDER_ID_HERE`

---

## Step 3: Configure DVC Remote

Edit `.dvc/config` (or create it) as follows:

```ini
[core]
    remote = storage

['remote "storage"']
    url = gdrive://YOUR_FOLDER_ID
    gdrive_client_id = YOUR_CLIENT_ID
    gdrive_client_secret = YOUR_CLIENT_SECRET
```
Replace YOUR_FOLDER_ID, YOUR_CLIENT_ID, and YOUR_CLIENT_SECRET with your values.

## Step 4: Install Dependencies

```bash
pip install dvc[gdrive]==3.58.0
pip install pathspec==0.11.2
pip install pydrive2
```

## Step 5: Authenticate (Headless / Docker-Friendly)

Create the authentication script:

```bash
mkdir -p /tmp
cat > /tmp/auth_gdrive_headless.py << 'EOF'
from pydrive2.auth import GoogleAuth
import os

gauth = GoogleAuth()
gauth.settings['client_config_backend'] = 'settings'
gauth.settings['client_config'] = {
    'client_id': 'YOUR_CLIENT_ID',
    'client_secret': 'YOUR_CLIENT_SECRET',
    'auth_uri': 'https://accounts.google.com/o/oauth2/auth',
    'token_uri': 'https://oauth2.googleapis.com/token',
    'revoke_uri': 'https://accounts.google.com/o/oauth2/revoke',
    'redirect_uri': 'urn:ietf:wg:oauth:2.0:oob'
}
gauth.settings['save_credentials'] = True
gauth.settings['save_credentials_file'] = f"/root/.cache/pydrive2fs/{gauth.settings['client_config']['client_id']}/default.json"

os.makedirs(os.path.dirname(gauth.settings['save_credentials_file']), exist_ok=True)

auth_url = gauth.GetAuthUrl()
print("Go to this URL in a browser:", auth_url)
code = input("Enter verification code: ").strip()
gauth.Auth(code)
gauth.SaveCredentialsFile(gauth.settings['save_credentials_file'])

print("✓ Credentials saved to", gauth.settings['save_credentials_file'])
EOF
```
Run the script:

```bash
python3 /tmp/auth_gdrive_headless.py
```
- Copy the printed URL into your **Windows browser**  
- Approve access  
- Copy the verification code back into the terminal  

This will generate a credentials file in the correct path for DVC.
## Step 6: Use DVC

Track your data:

```bash
dvc add data
```

Push to Google Drive:
```bash
dvc push
```

Commit .dvc files:
```bash
git add data.dvc .gitignore .dvc/config
git commit -m "Add DVC tracking"
```

## Troubleshooting

- If `dvc push` fails due to `pathspec`, run:

```bash
pip install pathspec==0.11.2
```

Authentication only needs to be done once per container

Credentials are stored in /root/.cache/pydrive2fs/YOUR_CLIENT_ID/default.json

For headless Docker, do not use localhost OAuth; always use the console copy-paste method