@echo off
echo QuickServeML Release Script
echo ===========================

if "%1"=="" (
    echo Usage: release.bat ^<version^>
    echo Example: release.bat 1.0.0
    exit /b 1
)

echo Creating release for version %1...
python scripts/release.py %1

if %ERRORLEVEL% EQU 0 (
    echo.
    echo Release created successfully!
    echo Check GitHub for the automated release process.
) else (
    echo.
    echo Release failed. Please check the error messages above.
) 