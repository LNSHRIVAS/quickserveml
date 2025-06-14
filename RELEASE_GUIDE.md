# QuickServeML Release Guide

This guide explains how to create releases for QuickServeML using different methods.

## Release Methods

### Method 1: Automated Release (Recommended)

The easiest way to create a release is using our automated release script:

```bash
# Make sure you're on the main branch and have committed all changes
git checkout main
git add .
git commit -m "Final changes for v1.0.0"

# Run the release script
python scripts/release.py 1.0.0
```

This will:
1. ✅ Update version numbers in `pyproject.toml` and `__init__.py`
2. ✅ Commit the version changes
3. ✅ Create and push a git tag (`v1.0.0`)
4. ✅ Push to GitHub
5. ✅ Trigger the automated GitHub Actions workflow

### Method 2: Manual GitHub Release

If you prefer to create releases manually through GitHub:

1. **Update version numbers**:
   ```bash
   # Edit pyproject.toml and change version to "1.0.0"
   # Edit quickserveml/quickserveml/__init__.py and change __version__ to "1.0.0"
   ```

2. **Commit and push changes**:
   ```bash
   git add .
   git commit -m "Bump version to 1.0.0"
   git push origin main
   ```

3. **Create a tag**:
   ```bash
   git tag -a v1.0.0 -m "Release 1.0.0"
   git push origin v1.0.0
   ```

4. **Create GitHub Release**:
   - Go to https://github.com/LNSHRIVAS/quickserveml/releases
   - Click "Create a new release"
   - Choose the `v1.0.0` tag
   - Copy content from `RELEASE_NOTES.md` as the release description
   - Upload built packages (optional, GitHub Actions will do this)

### Method 3: Using GitHub CLI

If you have GitHub CLI installed:

```bash
# Update versions and commit
python scripts/release.py 1.0.0

# Create release using gh CLI
gh release create v1.0.0 \
  --title "QuickServeML v1.0.0" \
  --notes-file RELEASE_NOTES.md \
  --draft=false \
  --prerelease=false
```

## Pre-Release Checklist

Before creating a release, ensure:

### ✅ Code Quality
- [ ] All tests pass: `python test_api_endpoints.py`
- [ ] Code is clean and professional (no emojis in code)
- [ ] All features are working correctly
- [ ] Documentation is up to date

### ✅ Version Numbers
- [ ] Version updated in `pyproject.toml`
- [ ] Version updated in `quickserveml/quickserveml/__init__.py`
- [ ] Version matches across all files

### ✅ Documentation
- [ ] `README.md` is current and accurate
- [ ] `RELEASE_NOTES.md` is complete
- [ ] `PROJECT_SUMMARY.md` reflects current state
- [ ] API documentation is generated correctly

### ✅ Git Status
- [ ] All changes are committed
- [ ] You're on the main branch
- [ ] Working directory is clean
- [ ] Remote is up to date

## Release Workflow

### 1. Automated Workflow (GitHub Actions)

When you push a tag like `v1.0.0`, the GitHub Actions workflow will:

1. **Build the package**:
   - Creates wheel (`.whl`) and source (`.tar.gz`) distributions
   - Runs on Ubuntu with Python 3.10

2. **Create GitHub Release**:
   - Uses content from `RELEASE_NOTES.md`
   - Attaches built packages
   - Sets as public release (not draft/prerelease)

3. **Upload to PyPI** (if configured):
   - Requires `PYPI_API_TOKEN` secret in GitHub
   - Skips if package already exists

### 2. Manual Steps (if needed)

If you need to build manually:

```bash
# Install build tools
pip install build twine

# Build package
python -m build

# Check package
twine check dist/*

# Upload to PyPI (if you have credentials)
twine upload dist/*
```

## Version Numbering

Follow semantic versioning (SemVer):

- **MAJOR.MINOR.PATCH** (e.g., 1.0.0)
- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

Examples:
- `1.0.0` - Initial release
- `1.0.1` - Bug fix release
- `1.1.0` - New features added
- `2.0.0` - Breaking changes

## PyPI Publishing

To publish to PyPI:

1. **Create PyPI account**: https://pypi.org/account/register/
2. **Generate API token**: https://pypi.org/manage/account/token/
3. **Add to GitHub Secrets**:
   - Go to repository Settings → Secrets and variables → Actions
   - Add `PYPI_API_TOKEN` with your PyPI token

The automated workflow will handle the rest!

## Troubleshooting

### Common Issues

**Tag already exists**:
```bash
git tag -d v1.0.0
git push origin :refs/tags/v1.0.0
# Then create new tag
```

**GitHub Actions failed**:
- Check the Actions tab in GitHub
- Verify all secrets are set correctly
- Ensure `RELEASE_NOTES.md` exists and is valid

**PyPI upload failed**:
- Verify `PYPI_API_TOKEN` secret is set
- Check if package name is available on PyPI
- Ensure version number is unique

### Getting Help

- **GitHub Issues**: Report problems with releases
- **Actions Logs**: Check detailed logs in GitHub Actions
- **PyPI Support**: Contact PyPI for package issues

## Quick Commands

### Create Release
```bash
python scripts/release.py 1.0.0
```

### Check Current Version
```bash
python -c "import quickserveml; print(quickserveml.__version__)"
```

### Build Package Locally
```bash
python -m build
```

### Test Package
```bash
pip install dist/quickserveml-1.0.0-py3-none-any.whl
quickserveml --help
```

---

**Happy Releasing!** 🚀

For questions or issues, please open a GitHub issue or contact the maintainers. 