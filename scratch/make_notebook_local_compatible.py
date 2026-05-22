import json

with open('train_board_segmentation_colab.ipynb', 'r') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        src = "".join(cell.get('source', []))
        
        # Update upload cell
        if 'from google.colab import files' in src and 'uploaded = files.upload()' in src:
            cell['source'] = [
                "import sys\n",
                "import os\n",
                "IN_COLAB = 'google.colab' in sys.modules\n",
                "if IN_COLAB:\n",
                "    from google.colab import files\n",
                "    print('Upload colab_training_data.zip')\n",
                "    uploaded = files.upload()\n",
                "    zip_name = list(uploaded.keys())[0]\n",
                "    print(f'Uploaded: {zip_name}')\n",
                "else:\n",
                "    print('Running locally. No need to upload zip.')\n"
            ]
            
        # Update unzip cell
        elif 'with zipfile.ZipFile(zip_name, \'r\') as z:' in src:
            cell['source'] = [
                "import zipfile\n",
                "if IN_COLAB:\n",
                "    with zipfile.ZipFile(zip_name, 'r') as z:\n",
                "        z.extractall('.')\n",
                "else:\n",
                "    print('Running locally. Using local annotation_frames directory.')\n"
            ]
            
        # Update download cell
        elif 'files.download(\'board_segmentation_model.pth\')' in src:
            cell['source'] = [
                "if IN_COLAB:\n",
                "    from google.colab import files\n",
                "    files.download('board_segmentation_model.pth')\n",
                "    print('Downloaded! Replace src/calibration/board_segmentation_model.pth in your repo.')\n",
                "else:\n",
                "    import shutil\n",
                "    shutil.copy('board_segmentation_model.pth', 'src/calibration/board_segmentation_model.pth')\n",
                "    print('Saved model locally to src/calibration/board_segmentation_model.pth')\n"
            ]

with open('train_board_segmentation_colab.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)
    f.write("\n")

print("Notebook updated for local compatibility.")
