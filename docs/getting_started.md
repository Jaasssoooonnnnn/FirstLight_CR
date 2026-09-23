# Install FirstLight CR

[Home](../README.md) · [Training](training.md)

This guide installs the offline engine from a fresh checkout. You need to supply the original Null's Royale APK; the repository does not include it or the game's downloaded update.

## Give this task to a coding agent

```text
Install FirstLight CR using docs/getting_started.md. Use an Android 12 MuMu instance with root and ARM64 support. Set up the local Python environment and .env. Install the original Null's Royale APK in MuMu and let it update until the normal lobby opens. After closing the game, run check_original_game.ps1; build and install the offline APK only if that check passes. If an input or tool is missing, tell me exactly what you need. Stop when install_offline_engine.ps1 succeeds.
```

## Requirements

| Component | Requirement |
| --- | --- |
| Desktop | Windows and Python 3.12 with Tkinter |
| MuMu | Android 12 instance with root and ARM64 application support |
| Game | Null's Royale APK 15.535.13, arm64-v8a; in-game content update 15.535.86 |
| Build tools | JDK 17, Android NDK, and Android SDK build-tools; NDK 27.3 and build-tools 37.0 have been used |

The build tools are needed to create the offline APK. They are not needed to run an APK already built for this version, but keep its matching `.build.json` file if you install one.

## 1. Set up Python

Run from the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
```

## 2. Select MuMu and fill `.env`

The regular MuMu Player installer is sufficient. Create an Android 12 instance, or use an existing one if you are willing to replace its Null's Royale installation. Enable root and ARM64 support.

Fill these values in the local `.env` file with paths and details from your machine:

| Setting | Value |
| --- | --- |
| `CR_PYTHON`, `CR_PYTHONW` | Absolute paths to `.venv/Scripts/python.exe` and `pythonw.exe` |
| `CR_MUMU_MANAGER` | Absolute path to `MuMuManager.exe` |
| `CR_ADB` | Absolute path to Android SDK `adb.exe` |
| `CR_VM_INDEX`, `CR_VM_NAME` | Index and exact name of the selected MuMu instance |
| `CR_ADB_SERIAL` | `127.0.0.1:<adb_port>` for that instance |
| `CR_JAVA`, `CR_KEYTOOL` | JDK executables |
| `CR_BUILD_TOOLS`, `CR_NDK_ROOT` | SDK build-tools directory and NDK root |

Find the instance name and ADB port with MuMu Manager:

```powershell
$manager = 'C:\path\to\MuMuManager.exe' # replace with your installed path
$vmIndex = 4                        # replace with your instance index
& $manager info --vmindex $vmIndex | ConvertFrom-Json |
  Select-Object index, name, adb_host_ip, adb_port
```

The scripts use the index, name, and ADB address to select the same instance. Leave unrelated `.env` settings blank.

## 3. Update the original game, then build

1. Put the unmodified original APK at `user-provided/nulls-royale.apk`, or pass its path with `-InputApk`.
2. Install it in the selected MuMu instance. Open the game, let it download its update naturally, and wait until you can enter the normal game lobby. Then close the game.
3. Run the commands below in order. `check_original_game.ps1` checks the local and installed APKs, Android 12, and the downloaded resources before the probe is built. A mismatch stops the process.

```powershell
./check_original_game.ps1 -InputApk ./user-provided/nulls-royale.apk
./build_offline_engine.ps1 `
  -InputApk ./user-provided/nulls-royale.apk `
  -OutputApk ./build/cr-ai-offline.apk
./install_offline_engine.ps1 -Apk ./build/cr-ai-offline.apk
```

The check isolates that MuMu instance from external network access after the game has updated. The build compiles the probe, patches and signs the APK, and may download Apktool. The installer checks the resources again before replacing the game.

After installation, open the interface with `./launch_interface.cmd` from the repository root.
