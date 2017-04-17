@echo off
title ROI generator
mode con cols=80 lines=3
color 7E
echo %ProgramFiles%

python "%PROGRAMDATA%\MedikEye\Roi_Gene\roi_gui.py"
::python "C:\Users\sylvain\Documents\boulot\startup\radiology\roittool\modulepython\roi_gui.py"

::pause > nul
exit