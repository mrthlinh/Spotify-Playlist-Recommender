1. Move files between Local and Remote Location
  - Copy all from Local Location to Remote Location (Upload)
```
scp -r /path/from/destination username@hostname:/path/to/destination
```
  - o copy all from Remote Location to Local Location (Download)

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
