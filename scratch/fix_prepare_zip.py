#!/usr/bin/env python3

with open('prepare_colab_upload.py', 'r') as f:
    lines = f.readlines()

new_lines = []
skip_mode = False
for line in lines:
    if "for script_path in [WARP_SCRIPT, TRAIN_WARPED_SCRIPT, WARPED_DATASET]:" in line:
        skip_mode = True
        new_lines.append("    # Package all Python source code in src/ and scripts/\n")
        new_lines.append("    script_files = []\n")
        new_lines.append("    import glob\n")
        new_lines.append("    script_files.extend(glob.glob('src/**/*.py', recursive=True))\n")
        new_lines.append("    script_files.extend(glob.glob('scripts/**/*.py', recursive=True))\n")
        new_lines.append("    for script_path in script_files:\n")
        new_lines.append("        if os.path.exists(script_path):\n")
        new_lines.append("            zf.write(script_path, script_path)\n")
        new_lines.append("            sz = os.path.getsize(script_path)\n")
        new_lines.append("            total += sz\n")
        new_lines.append("            print(f'  + {script_path}  ({sz/1024:.0f} KB)')\n")
        continue

    if skip_mode:
        if "Optional HockeyRink keypoint" in line:
            skip_mode = False
            new_lines.append(line)
    else:
        new_lines.append(line)

with open('prepare_colab_upload.py', 'w') as f:
    f.writelines(new_lines)

print("Updated prepare_colab_upload.py to include all src/ and scripts/ files.")
