from huggingface_hub import HfApi
from remotezip import RemoteZip
from pathlib import Path
import re
import numpy as np

# configuration
REPO_ID = "l-lt/LaSOT"
NUM_FRAMES = 100
NUM_CATEGORIES = 50
OUTPUT_DIR = Path("./data/raw")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# return list of 100 frames
def get_sequence_frames(sequence, files):
    all_frames = [f for f in files if sequence in f and f.lower().endswith((".jpg", ".jpeg", ".png"))]
    return all_frames[:NUM_FRAMES]

# save the image to data path
def write_img_data(frame, pth, zf):
    f_path = Path(frame).name
    image_bytes = zf.read(frame)
    out_pth = (pth / Path(frame).name)
    out_pth.write_bytes(image_bytes)
    return out_pth

# main 
def main():
    # load hf api
    api = HfApi()

    # get zip files
    repo_files = api.list_repo_files(repo_id=REPO_ID, repo_type="dataset")
    all_zips = [file for i, file in enumerate(repo_files[2:]) if ".zip" in file]

    # subset of all categories in lasot dataset
    zip_files = all_zips[: NUM_CATEGORIES]

    # iterate over categories
    for zip in zip_files:
        # category name
        category = zip.split(".")[0]
        
        # load dataset from url
        url = ("https://huggingface.co/datasets/l-lt/LaSOT/" f"resolve/main/{zip}")
        with RemoteZip(url) as zf:
            files = zf.namelist()
            
            # video sequences in dataset directory
            sequences = sorted({
                match.group(1) for file in files
                if (match := re.match(r"^(.+)/img/\d+\.jpg$", file))
            })

            # iterate over sequences 
            for sequence in sequences:
                # path data
                output_sequence = (OUTPUT_DIR / category / sequence)
                output_sequence.mkdir(parents=True, exist_ok=True)
                
                # load frames
                frames = get_sequence_frames(sequence, files)

                # save frames to data path
                for frame in frames: out_path = write_img_data(frame, output_sequence, zf)
    
    # successfully created subset 
    return 0


if __name__ == "__main__":
    main()
