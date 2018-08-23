@echo off

mkdir positive_test negative_test

set i=0

:loop

set /a R=%RANDOM%*5001/32768

copy positives\%R%.png positive_test\ > nul 2>&1
copy negatives\%R%.png negative_test\ > nul 2>&1

set /a i+=1

if %i% leq 999 goto loop
