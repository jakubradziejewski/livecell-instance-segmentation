DVC Google Drive Setup (Docker)
Prerequisites

Google account
Docker with port 8080 forwarded (add to docker-compose.yml: ports: - "8080:8080")

Step 1: Create Google OAuth Credentials

Go to https://console.cloud.google.com/
Create a new project (e.g., "DVC Storage")
Navigate to APIs & Services → Enable APIs and Services
Search for "Google Drive API" and enable it
Go to OAuth consent screen:

User Type: External
App name: "DVC"
Add your email as developer contact
Add yourself under Test users


Go to Credentials → Create Credentials → OAuth client ID:

Application type: Desktop app
Name: "DVC Client"


Download the JSON file and note your client_id and client_secret

Step 2: Create Google Drive Folder

Go to https://drive.google.com
Create a folder (e.g., "dvc-storage")
Open the folder and copy the ID from URL: https://drive.google.com/drive/folders/FOLDER_ID_HERE

Step 3: Configure DVC
```
Edit .dvc/config:
[core]
    remote = storage
['remote "storage"']
    url = gdrive://YOUR_FOLDER_ID
    gdrive_client_id = YOUR_CLIENT_ID
    gdrive_client_secret = YOUR_CLIENT_SECRET
```
Step 4: Install Dependencies
pip install dvc[gdrive]==3.58.0
pip install pathspec==0.11.2

Step 5: Authenticate
Create and run this authentication script once:
cat > /tmp/auth_gdrive.py << 'EOF'
from pydrive2.auth import GoogleAuth
import json, os

gauth = GoogleAuth()
gauth.settings['client_config_backend'] = 'settings'
gauth.settings['client_config'] = {
    'client_id': 'YOUR_CLIENT_ID',
    'client_secret': 'YOUR_CLIENT_SECRET',
    'auth_uri': 'https://accounts.google.com/o/oauth2/auth',
    'token_uri': 'https://oauth2.googleapis.com/token',
    'revoke_uri': 'https://oauth2.googleapis.com/revoke',
    'redirect_uri': 'urn:ietf:wg:oauth:2.0:oob'
}
gauth.settings['save_credentials'] = True
gauth.settings['save_credentials_file'] = f'/root/.cache/pydrive2fs/YOUR_CLIENT_ID/default.json'

auth_url = gauth.GetAuthUrl()
print("="*80)
print("Go to this URL:", auth_url)
print("="*80)
code = input("Enter code: ").strip()
gauth.Auth(code)

creds = {
    'access_token': gauth.credentials.access_token,
    'client_id': gauth.credentials.client_id,
    'client_secret': gauth.credentials.client_secret,
    'refresh_token': gauth.credentials.refresh_token,
    'token_expiry': gauth.credentials.token_expiry.isoformat() if gauth.credentials.token_expiry else None,
    'token_uri': gauth.credentials.token_uri,
    'user_agent': gauth.credentials.user_agent,
    'revoke_uri': gauth.credentials.revoke_uri,
    'scopes': list(gauth.credentials.scopes) if gauth.credentials.scopes else [],
    '_class': 'OAuth2Credentials',
    '_module': 'oauth2client.client'
}

os.makedirs(os.path.dirname(gauth.settings['save_credentials_file']), exist_ok=True)
with open(gauth.settings['save_credentials_file'], 'w') as f:
    json.dump(creds, f, indent=2)
print("\n✓ Authentication successful!")
EOF

python3 /tmp/auth_gdrive.py

Replace YOUR_CLIENT_ID and YOUR_CLIENT_SECRET with your actual values.
When prompted:

Open the URL in your browser
Authenticate with Google
Copy the authorization code Google shows you
Paste it into the terminal

Step 6: Use DVC
# Add data to DVC tracking
dvc add data

# Push to Google Drive
dvc push

# Commit .dvc files to git
git add data.dvc .gitignore .dvc/config
git commit -m "Add DVC tracking"

Troubleshooting

If dvc push fails with import error: pip install pathspec==0.11.2
Authentication only needs to be done once per container
Credentials are cached in /root/.cache/pydrive2fs/