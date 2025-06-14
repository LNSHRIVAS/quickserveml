# PyPI Publishing Setup Guide

This guide explains how to set up PyPI publishing for QuickServeML so that users can install your package with `pip install quickserveml`.

## Current Status

**v1.0.2**: The GitHub Actions workflow is now configured to handle missing PyPI tokens gracefully. The release will be created on GitHub, but PyPI publishing will be skipped if no token is configured.

## Option 1: Skip PyPI Publishing (Current Setup)

If you don't want to publish to PyPI right now, the current setup is perfect:
- ✅ GitHub releases are created automatically
- ✅ Users can install from GitHub releases
- ✅ No PyPI account needed
- ✅ No additional configuration required

## Option 2: Enable PyPI Publishing

If you want to enable PyPI publishing so users can install with `pip install quickserveml`, follow these steps:

### Step 1: Create PyPI Account

1. Go to https://pypi.org/account/register/
2. Create a new account
3. Verify your email address

### Step 2: Generate API Token

1. Go to https://pypi.org/manage/account/token/
2. Click "Add API token"
3. Give it a name like "QuickServeML GitHub Actions"
4. Set scope to "Entire account (all projects)"
5. Copy the token (you won't see it again!)

### Step 3: Add Token to GitHub Secrets

1. Go to your GitHub repository: https://github.com/LNSHRIVAS/quickserveml
2. Click "Settings" tab
3. Click "Secrets and variables" → "Actions"
4. Click "New repository secret"
5. Name: `PYPI_API_TOKEN`
6. Value: Paste the token you copied from PyPI
7. Click "Add secret"

### Step 4: Test PyPI Publishing

1. Create a new release:
   ```bash
   python scripts/release.py 1.0.3
   ```
2. Check the GitHub Actions workflow
3. If successful, your package will be available on PyPI

## Package Installation Options

### From GitHub (Current)
```bash
pip install git+https://github.com/LNSHRIVAS/quickserveml.git
```

### From PyPI (After Setup)
```bash
pip install quickserveml
```

## Troubleshooting

### Common Issues

**"Package name already exists"**
- The package name "quickserveml" might already be taken
- Check https://pypi.org/project/quickserveml/
- If taken, update the package name in `pyproject.toml`

**"Invalid token"**
- Make sure the token is copied correctly
- Check that the token has the right permissions
- Verify the secret name is exactly `PYPI_API_TOKEN`

**"Trusted publisher error"**
- This is now fixed! The workflow uses traditional token authentication
- No trusted publisher setup needed

## Benefits of PyPI Publishing

### For Users
- **Easy Installation**: `pip install quickserveml`
- **Version Management**: `pip install quickserveml==1.0.2`
- **Dependency Resolution**: Automatic dependency management

### For You
- **Wider Distribution**: Available to all Python users
- **Professional Appearance**: Listed on PyPI
- **Version Tracking**: Clear version history

## Package Metadata

Your package is already configured with:
- **Name**: `quickserveml`
- **Description**: "One-command ONNX model serving, benchmarking, and visual inspection"
- **Author**: Lakshminarayan Shrivas
- **Email**: lakshminarayanshrivas7@gmail.com
- **License**: MIT
- **Python Version**: 3.7+

## Next Steps

1. **Decide**: Do you want to publish to PyPI now?
2. **If Yes**: Follow the setup steps above
3. **If No**: Continue with GitHub-only releases
4. **Future**: You can always enable PyPI publishing later

## Security Notes

- **Never commit tokens** to your repository
- **Use repository secrets** for sensitive data
- **Rotate tokens** periodically
- **Limit token scope** if possible

---

**Current Recommendation**: The GitHub-only approach is working well and is sufficient for most use cases. You can always add PyPI publishing later when you're ready for wider distribution. 