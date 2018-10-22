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

Writing console output to a different file
nohup ./script.sh > my-output 2>&1 &
```

4. Remove directory

```
rm -rf directoryName
```

5. Check more info of files

```
du -h
ls -l
```

6. Add new bash script to execute from anywhere

```
Put file into ~/bin
chmod +x filename
Make sure you add ~/bin to .bashrc  export PATH="/home/bking/bin:$PATH"

```

7. Connect existing kernel

```
- In server, open terminal "jupyter kernel"
- You should see something like

[KernelApp] Starting kernel 'python3'
[KernelApp] Connection file: /run/user/605339/jupyter/kernel-6a6ad673-f5ea-4d2f-86ba-806ff3f7a223.json
[KernelApp] To connect a client: --existing kernel-6a6ad673-f5ea-4d2f-86ba-806ff3f7a223.json

- Copy the "kernel-666.json" to your local folder
- In Spyder, open Console -> Connect to an existing kernel
- Connection Info: browse to the copy "kernel.json"
- hostname: txt171930@ares.utdallas.edu
- ssh key: browse to ssh key of your computer id_rsa.pem
- pass:
```

scp -r data/df_data/df_small txt171930@ares.utdallas.edu:~/SpotifyRecSys/data/df_data/df_small
