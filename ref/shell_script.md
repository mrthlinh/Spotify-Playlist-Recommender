1. Move files between Local and Remote Location
  - Copy all from Local Location to Remote Location (Upload)
```
scp -r /path/from/destination username@hostname:/path/to/destination
```
  - o copy all from Remote Location to Local Location (Download)
scp txt171930@ares.utdallas.edu:~/SpotifyRecSys/data/df_data/df_tracks_challenge_incomplete.hdf .

"." means current directory, the above command copy file to your current folder in local machine

2. Check all process running
```
top -H
```
Ctrl + C to exit

3. Make python script running after closing terminal
```
Using No-Hang-Up
nohup python pythonScript.py
```

3. Remove directory

```
rm -rf directoryName
```

4. Check more info of files
```
du -h
ls -l
```

5. Add new bash script to execute from anywhere
```
Put file into ~/bin
chmod +x filename
Make sure you add ~/bin to .bashrc  export PATH="/home/bking/bin:$PATH"

```
