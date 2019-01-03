#!/bin/bash

python3 -m code.download_hpa >download_hpa.log 2>&1 & disown
