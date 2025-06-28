# Triton Inference Server Troubleshooting Guide

## Common Issues and Solutions

### Docker Issues

#### Incomprehensible Characters in the Terminal

If you see incomprehensible characters (like `╨Ч╨░╨┐╤Г╤Б╨║`) in your terminal when running the script, it's likely a character encoding issue. Use the English version of the scripts provided:

- `run_triton.bat` (English version)
- `run_triton.ps1` (English version)

#### Docker API Error

If you get an error like:

```
docker: request returned Internal Server Error for API route and version http://%2F%2F.%2Fpipe%2FdockerDesktopLinuxEngine/_ping
```

This indicates issues with Docker Desktop. Try these solutions:

1. Restart Docker Desktop
2. Check if Docker Desktop is running
3. Reinstall Docker Desktop if needed
4. Run Docker commands in an administrative PowerShell or Command Prompt

### Triton Server Issues

#### SIGTERM/SIGINT Errors

If you see messages like `got 3 SIGTERM/SIGINTs, forcefully exiting`, try these solutions:

1. **Run without GPU**:
   ```bash
   docker run --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 -v %cd%/model_repository:/models nvcr.io/nvidia/tritonserver:23.04-py3 tritonserver --model-repository=/models --strict-model-config=false
   ```

2. **Verify NVIDIA Docker integration**:
   ```bash
   docker info | findstr "Runtimes"
   ```
   It should list `nvidia` as one of the runtimes.

3. **Check GPU Status**:
   ```bash
   nvidia-smi
   ```
   If this command fails, you may need to update your NVIDIA drivers.

#### Missing Model Files

Ensure your model structure is correct:

```
model_repository/
└── resnet50/
    ├── config.pbtxt
    └── 1/
        └── model.onnx
```

If `model.onnx` is missing, generate it using:

```bash
python model_converter.py --model resnet50 --output model_repository/resnet50/1/model.onnx --create_config
```

#### Testing Connectivity

To test if Triton server is running and accessible, use the included testing script:

```bash
python triton_test.py
```

## Specific Docker Troubleshooting

### Docker Desktop

1. **Reset Docker Desktop**:
   - Open Docker Desktop
   - Go to Settings > Troubleshoot
   - Click "Reset to factory defaults"

2. **Check Docker Service**:
   ```powershell
   # In PowerShell as Administrator
   Get-Service com.docker*
   Restart-Service com.docker.service
   ```

3. **WSL Issues** (if using WSL backend):
   ```powershell
   # In PowerShell as Administrator
   wsl --shutdown
   wsl --update
   ```

## Path Problems

If you're experiencing path-related issues with Docker volumes, try these alternate volume mounting syntax:

### In PowerShell:

```powershell
docker run --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 -v "${PWD}/model_repository":/models nvcr.io/nvidia/tritonserver:23.04-py3 tritonserver --model-repository=/models --strict-model-config=false
```

### In CMD:

```cmd
docker run --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 -v "%cd%/model_repository":/models nvcr.io/nvidia/tritonserver:23.04-py3 tritonserver --model-repository=/models --strict-model-config=false
```

For paths with spaces, use double quotes and escape properly.
