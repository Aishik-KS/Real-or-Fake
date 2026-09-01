# AIGC Image Detector

This project checks images and returns a confidence score showing how likely
each image is to be AI-generated.

Docker is not required.

## Project files

```text
.streamlit/
└── config.toml

Model/
└── best_model.pt

Test Images/
└── Images to evaluate

RunModel.py
WebApp.py
README.md
```

## Requirements

- Python 3.10 or newer
- Approximately 5 GB of free disk space
- 16 GB RAM recommended

## Setup in VS Code

1. Open VS Code.
2. Select **File → Open Folder**.
3. Open the folder containing `RunModel.py`.
4. Select **Terminal → New Terminal**.
5. Check that Python is installed:

   ```powershell
   python --version
   ```

6. Install the required packages:

   ```powershell
   python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
   python -m pip install transformers pillow numpy opencv-python-headless streamlit
   ```

## Running the Web Frontend

You can run the interactive web interface locally using Streamlit.

### 1. Prerequisites & Dependencies

Ensure you have Python 3.10+ installed and the required dependencies set up:

```bash
# Install PyTorch (CPU version)
python -m pip install torch --index-url [https://download.pytorch.org/whl/cpu](https://download.pytorch.org/whl/cpu)

# Install frontend and image processing packages
python -m pip install streamlit transformers pillow numpy opencv-python-headless

## Run from the command line

Place the model checkpoint here:

```text
Model/best_model.pt
```

Place the images you want to check inside `Test Images`, then run:

```powershell
python RunModel.py --image_dir "Test Images" --output predictions.json
```

You can also use the shorter command:

```powershell
python RunModel.py
```

The model may take a few minutes to run on CPU.

## Output

The results are saved in `predictions.json`:

```json
[
  {
    "image_path": "Test Images/example.png",
    "pred": 0.982451
  }
]
```

Score meaning:

- Near `1.0`: more likely AI-generated
- Near `0.0`: more likely real
- Near `0.5`: uncertain

## Choose an image folder

The images do not have to be copied into `Test Images`. That folder is only
provided as the default example.

You can choose either of these methods.

### Option 1: Use a different folder

Keep the images in any folder and provide its path using `--image_dir`:

```powershell
python RunModel.py --image_dir "C:\path\to\images" --output results.json
```

### Option 2: Use `Test Images` via CLI

Copy the images into `Test Images` and run:

```powershell
python RunModel.py
```

Using a separate folder is recommended because it keeps the supplied test
images and your own images separate.

Supported image formats are `.jpg`, `.jpeg`, `.png`, `.bmp`, and `.webp`.

## Common errors

### `python` is not recognized

Install Python from <https://www.python.org/downloads/> and enable
**Add Python to PATH** during installation. On Windows, you can also try `py`
instead of `python`.

### Missing Python package

Run the installation commands in the **Setup in VS Code** section again.

### Model checkpoint not found

Make sure the checkpoint is located at `Model/best_model.pt`.

## GitHub model upload note

`best_model.pt` is approximately 2.94 GB and cannot be uploaded as a normal
GitHub file. GitHub Team or Enterprise users can store it with Git LFS. On
GitHub Free or Pro, upload the model to external storage and provide a download
link. After downloading, place it at:

```text
Model/best_model.pt
```
