import wget
import sys
import os
import tqdm

jpg_dir = sys.argv[1]

for map_listing in tqdm.tqdm(os.listdir(jpg_dir)):
    dir_name = map_listing.split('.')[0]
    print(dir_name)
    os.makedirs(f"./{dir_name}", exist_ok=True) #saves locally
    print(map_listing)
    with open(os.path.join(jpg_dir,map_listing), "r") as hjpg:
        for line in tqdm.tqdm(hjpg.readlines()):
            print(line)
            wget.download(line, out=os.path.join(dir_name, line.split('/')[-1]))

