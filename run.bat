@echo off
setlocal enabledelayedexpansion

set "PROJECT_DIR=%~dp0"
set "PARENT_DIR=%PROJECT_DIR%..\"
for %%I in ("%PARENT_DIR%") do set "PARENT_DIR=%%~fI"
set "PIXI_EXE=%PARENT_DIR%\bin\pixi.exe"
set "CUSTOM_PIXI=0"
set "PASS_ARGS="
set "BACKEND=auto"

:parse_args
if "%~1"=="" goto args_done
if /I "%~1"=="--pixi-path" (
    if "%~2"=="" exit /b 1
    for %%I in ("%~2") do set "PIXI_EXE=%%~fI"
    set "CUSTOM_PIXI=1"
    shift
    shift
    goto parse_args
)
if /I "%~1"=="--backend" (
    if "%~2"=="" exit /b 1
    set "BACKEND=%~2"
    shift
    shift
    goto parse_args
)
set "ARG1=%~1"
if /I "!ARG1:~0,12!"=="--pixi-path=" (
    set "PIXI_VALUE=!ARG1:~12!"
    for %%I in ("!PIXI_VALUE!") do set "PIXI_EXE=%%~fI"
    set "CUSTOM_PIXI=1"
    shift
    goto parse_args
)
if /I "!ARG1:~0,10!"=="--backend=" (
    set "BACKEND=!ARG1:~10!"
    shift
    goto parse_args
)
set "PASS_ARGS=!PASS_ARGS! %1"
shift
goto parse_args

:args_done
if /I "%BACKEND%"=="auto" (
    where nvidia-smi >nul 2>nul
    if errorlevel 1 (
        set "BACKEND=cpu"
    ) else (
        nvidia-smi -L >nul 2>nul
        if errorlevel 1 (
            set "BACKEND=cpu"
        ) else (
            set "BACKEND=cuda"
        )
    )
)
if /I "%BACKEND%"=="cpu" (
    set "PIXI_ENV=cpu"
) else if /I "%BACKEND%"=="cuda" (
    set "PIXI_ENV=default"
) else (
    echo Unsupported RVC backend: %BACKEND%
    exit /b 1
)

if "%CUSTOM_PIXI%"=="1" if not exist "%PIXI_EXE%" exit /b 1
if "%CUSTOM_PIXI%"=="0" if not exist "%PIXI_EXE%" (
    if not exist "%PARENT_DIR%\bin" mkdir "%PARENT_DIR%\bin"
    powershell -Command "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri 'https://github.com/prefix-dev/pixi/releases/download/v0.68.1/pixi-x86_64-pc-windows-msvc.exe' -OutFile '%PIXI_EXE%'"
    if errorlevel 1 exit /b 1
)

set "PIXI_CACHE_DIR=%PARENT_DIR%\.pixi-cache"
set "PIP_CACHE_DIR=%PARENT_DIR%\.pip-cache"
set "TMP=%PARENT_DIR%\.tmp"
set "TEMP=%PARENT_DIR%\.tmp"
set "PIXI_FROZEN=true"
if not exist "%PIXI_CACHE_DIR%" mkdir "%PIXI_CACHE_DIR%"
if not exist "%PIP_CACHE_DIR%" mkdir "%PIP_CACHE_DIR%"
if not exist "%TMP%" mkdir "%TMP%"

cd /d "%PROJECT_DIR%"
"%PIXI_EXE%" install --frozen --environment "%PIXI_ENV%"
if errorlevel 1 exit /b 1
"%PIXI_EXE%" run --environment "%PIXI_ENV%" python run.py --backend "%BACKEND%" !PASS_ARGS!
