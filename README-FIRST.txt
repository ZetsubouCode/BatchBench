===============================================
BatchBench — Local Batch Image & Dataset Toolkit
===============================================
A tiny Flask site you can run on your own PC (localhost) with simple menus that wrap common batch utilities:
  • WebP → PNG converter
  • Batch photo adjust using presets (warmth/tint/brightness/etc.)
  • Dataset tag editor (insert/delete/replace/move/dedup for .txt beside images)
  • Append suffix to all files in a folder
  • Reorder paired (image + .txt) names across numbered subfolders

This is designed for Windows beginners.

----------------------
0) You’ll need Python
----------------------
• Download Python 3.11 or newer from https://www.python.org/downloads/windows/
• When installing, CHECK “Add python.exe to PATH”.

(Optional) Verify installer integrity with MD5 on Windows:
  1) Put the installer (e.g., python-3.11.9-amd64.exe) in your Downloads folder.
  2) Open Command Prompt and run:
     certutil -hashfile "%USERPROFILE%\Downloads\python-3.11.9-amd64.exe" MD5

  You will see an MD5 hash. Compare it with the checksum on python.org (if provided).

----------------------------------------------
1) Unzip this folder somewhere simple, like C:\BatchBench
----------------------------------------------
The folder should contain:
  app.py
  requirements.txt
  .env.example
  setup.bat
  run.bat
  md5sum.py
  /templates  /static  /tools

---------------------------------------------------
2) (One-time) Install everything with setup.bat
---------------------------------------------------
• Double-click setup.bat
  - It creates a virtual environment (.venv)
  - Installs required Python packages
  - Prepares a .env file (you can edit later)

If setup.bat fails because Python is not found, install Python then run setup.bat again.

-----------------------------------
3) Start the site with run.bat
-----------------------------------
• Double-click run.bat
• Your browser should open to: http://127.0.0.1:5000/
• If the browser does not open automatically, open it yourself and paste the address.

-----------------------------------------------
4) MD5 checksums for your project files (local)
-----------------------------------------------
You can generate MD5 for all project files to check integrity after copying:
  - Double-click: md5sum.py  (or run: .venv\Scripts\python md5sum.py)
  - It prints MD5 for each file and also creates checksums.md

-----------------------------------
Tips
-----------------------------------
• If a tool says “folder not found,” copy & paste the full Windows path, e.g.:
    D:\_Training DATA\MySet
  You can also drag a folder into the text box to paste its path.
• Back up important data before bulk operations.
