#!/bin/bash

#python3 -m code.resnet_fastai -D official_hpav18 -a resnet -s 512 -b 16 -d 50 -t 0.1 -r 1e-2 -S random -l bce -e 15 -E 25 -i 0 >trial1.log 2>&1 & disown
#python3 -m code.resnet_fastai -D official_hpav18 -a resnet -s 32 -b 256 -d 50 -t 0.1 -r 1e-2 -S random -l bce -e 5 -E 15 -i 0 >trial2.log 2>&1 & disown
#python3 -m code.resnet_fastai -D official -a resnet -s 512 -b 32 -d 50 -t 0.1 -r 1e-2 -S random -l bce -e 5 -E 15 -f 5 -i 0 >trial3.log 2>&1 & disown
#python3 -m code.resnet_fastai -D official_hpav18 -a resnet -s 512 -b 32 -d 50 -t 0.1 -r 1e-2 -S random -l bce -e 5 -E 15 -i 0 >trial4.log 2>&1 & disown
#python3 -m code.resnet_fastai -D official_hpav18 -a resnet -s 512 -b 32 -d 50 -t 0.1 -r 1e-2 -S random -l bce -e 3 -E 15 -i 0 >trial5.log 2>&1 & disown
#python3 -m code.resnet_fastai -D official_hpav18 -a resnet -s 512 -b 32 -d 50 -t 0.1 -r 1e-2 -S random -l bce -e 3 -E 15 -i 0 -m model/stage-1-resnet50-512-official_hpav18-bce-random-drop0.5-th0.1-bs32-lr0.01-ep3_15-0.pth >trial6.log 2>&1 & disown
#python3 -m code.resnet_fastai -D official -a resnet -s 512 -b 32 -d 50 -t 0.1 -r 1e-2 -S random -l bce -e 5 -E 15 -i 0 -f 5 >trial7.log 2>&1 & disown
#python3 -m code.resnet_fastai -D official -a resnet -s 512 -b 32 -d 50 -t 0.1 -r 1e-2 -S random -l bce -e 0 -E 0 -i 0 -m stage-2-resnet50-512-official-bce-random-drop0.5-th0.1-bs32-lr0.01-ep5_15 -f 5 >trial8.log 2>&1 & disown
#python3 -m code.resnet_fastai -D official -a resnet -s 512 -b 32 -d 50 -t 0.1 -r 1e-2 -S random -l bce -e 0 -E 15 -i 0 -m stage-2-resnet50-512-official-bce-random-drop0.5-th0.1-bs32-lr0.01-ep5_15 -f 5     >trial9.log 2>&1 & disown
#python3 -m code.resnet_fastai -D official -a resnet -s 512 -b 32 -d 50 -t 0.1 -r 1e-2 -S random -l bce -e 0 -E 15 -i 0 -m stage-2-resnet50-512-official-bce-random-drop0.5-th0.1-bs32-lr0.01-ep5_15 -f 5 >trial10.log 2>&1 & disown
#python3 -m code.resnet_fastai -D official -a resnet -s 512 -b 32 -d 50 -t 0.1 -r 5e-3 -S random -l bce -e 15 -E 0 -i 0 -m stage-1-resnet50-512-official-bce-random-drop0.5-th0.1-bs32-lr0.01-ep5_15 -f 5     >trial11.log 2>&1 & disown
#python3 -m code.resnet_fastai -D official -a resnet -s 512 -b 32 -d 50 -t 0.1 -r 2e-3 -S random -l bce -e 0 -E 15 -i 0 -m stage-2-resnet50-512-official-bce-random-drop0.5-th0.1-bs32-lr0.01-ep5_15 -f     5 >trial12.log 2>&1 & disown
#python3 -m code.resnet_fastai -D official -a resnet -s 512 -b 32 -d 50 -t 0.1 -S random -l bce -e 15 -E 25 -i 0 >trial13.log 2>&1 & disown
#python3 -m code.resnet_fastai -D official -a resnet -s 512 -b 32 -d 50 -t 0.1 -S random -l bce -e 15 -E 25 -i 0 >trial14.log 2>&1 & disown
#python3 -m code.resnet_fastai -D official -a resnet -s 512 -b 32 -d 50 -t 0.1 -S random -l bce -e 15 -E 25 -i 0 >trial15.log 2>&1 & disown
#python3 -m code.resnet_fastai -D official -a resnet -s 512 -b 32 -d 50 -t 0.1 -S random -l bce -e 15 -E 25 -i 0 >trial16.log 2>&1 & disown
#python3 -m code.resnet_fastai -D official -a resnet -s 512 -b 32 -d 50 -t 0.1 -S random -l bce -e 3 -E 25 -i 0 >trial17.log 2>&1 & disown
#python3 -m code.resnet_fastai -D official -a resnet -s 512 -b 32 -d 50 -t 0.1 -S random -l bce -e 0 -E 25 -i 0 -m stage-1-resnet50-512-official-bce-random-drop0.5-th0.1-bs32-lr0-ep3_25 >trial18.log 2>&1 & disown
#python3 -m code.resnet_fastai -D official -a resnet -s 512 -b 32 -d 50 -t 0.1 -S random -l bce -e 0 -E 25 -i 0 -r 2e-5 -m stage-1-resnet50-512-official-bce-random-drop0.5-th0.1-bs32-lr0-ep3_25 >trial19.log 2>&1 & disown
#python3 -m code.resnet_fastai -D official -a resnet -s 512 -b 32 -d 50 -t 0.1 -S random -l bce -e 3 -E 25 -i 0 -f 5 >trial20.log 2>&1 & disown
#python3 -m code.resnet_fastai -D official -a resnet -s 512 -b 32 -d 50 -t 0.1 -S random -l bce -e 0 -E 0 -r 2e-5 -m stage-1-resnet50-512-official-bce-random-drop0.5-th0.1-bs32-lr0-ep3_25 -i 0 >trial21.log 2>&1 & disown
#python3 -m code.resnet_fastai -D official -a resnet -s 512 -b 32 -d 50 -t 0.1 -S random -l bce -e 0 -E 0 -r 2e-5 -m stage-2-resnet50-512-official-bce-random-drop0.5-th0.1-bs32-lr0-ep3_25 -i 0 >trial22.log 2>&1 & disown
#python3 -m code.resnet_fastai -D official_hpav18 -a resnet -s 512 -b 32 -d 50 -t 0.2 -S random -l bce -e 3 -E 25 -i 0 >trial23.log 2>&1 & disown
#python3 -m code.resnet_fastai -D official_hpav18 -a resnet -s 512 -b 32 -d 50 -t 0.1 -S random -l bce -e 0 -E 25 -r 1e-6 -m stage-1-resnet50-512-official_hpav18-bce-random-drop0.5-th0.2-bs32-lr0-ep3_25 -i 0 >trial24.log 2>&1 & disown
#python3 -m code.resnet_fastai -D official_hpav18 -a resnet -s 512 -b 32 -d 50 -t 0.1 -S random -l bce -e 3 -E 25 -i 0 >trial25.log 2>&1 & disown
#python3 -m code.resnet_fastai -D official_hpav18 -a resnet -s 512 -b 32 -d 50 -t 0.1 -S random -l bce -e 2 -E 0 -m stage-1-resnet50-512-official_hpav18-bce-random-drop0.5-th0.1-bs32-lr0-ep3_25 -i 0 -r 0.01 >trial26.log 2>&1 & disown
#python3 -m code.resnet_fastai -D official_hpav18 -a resnet -s 512 -b 32 -d 50 -t 0.2 -S random -l bce -e 0 -E 25 -r 2e-6 -m stage-1-resnet50-512-official_hpav18-bce-random-drop0.5-th0.2-bs32-lr0-ep3_25 -i 0 >trial27.log 2>&1 & disown
#python3 -m code.resnet_fastai -D official_hpav18 -a resnet -s 512 -b 32 -d 50 -t 0.1 -S random -l bce -e 0 -E 25 -m stage-1-resnet50-512-official_hpav18-bce-random-drop0.5-th0.1-bs32-lr0-ep3_25 -i 0 -r 1e-6 >trial28.log 2>&1 & disown
#python3 -m code.resnet_fastai -D official_hpav18 -a resnet -s 512 -b 32 -d 50 -t 0.1 -S random -l bce -e 0 -E 15 -m stage-2-resnet50-512-official_hpav18-bce-random-drop0.5-th0.1-bs32-lr0-ep3_25 -i 0 -r 1e-6 >trial29.log 2>&1 & disown
#python3 -m code.resnet_fastai -D official_hpav18 -a resnet -s 512 -b 32 -d 50 -t 0.1 -S random -l bce -e 0 -E 15 -m stage-2-resnet50-512-official_hpav18-bce-random-drop0.5-th0.1-bs32-lr0-ep3_25 -i 0 -r 1e-5 >trial30.log 2>&1 & disown
#python3 -m code.resnet_fastai -D official -a resnet -s 512 -b 32 -d 50 -t 0.1 -S random -l bce -e 0 -E 15 -m stage-2-resnet50-512-official-bce-random-drop0.5-th0.1-bs32-lr0-ep3_25 -i 0 >trial31.log 2>&1 & disown
#python3 -m code.resnet_fastai -D official -a resnet -s 512 -b 32 -d 50 -t 0.1 -S random -l bce -e 0 -E 15 -r 7e-6 -m stage-2-resnet50-512-official-bce-random-drop0.5-th0.1-bs32-lr0-ep3_25 -i 0 >trial32.log 2>&1 & disown
#python3 -m code.resnet_fastai -D official -a resnet -s 512 -b 32 -d 50 -t 0.1 -S random -l bce -e 0 -E 25 -r 2e-5 -m stage-2-resnet50-512-official-bce-random-drop0.5-th0.1-bs32-lr0-ep3_25 -i 0 >trial33.log 2>&1 & disown
#python3 -m code.resnet_fastai -D official -a resnet -s 512 -b 32 -d 50 -t 0.1 -S weighted -l bce -e 3 -E 25 -i 0 >trial34.log 2>&1 & disown
#python3 -m code.resnet_fastai -D official -a resnet -s 512 -b 32 -d 50 -t 0.1 -S weighted -l bce -e 0 -E 25 -r 1e-6 -m stage-2-resnet50-512-official-bce-weighted-drop0.5-th0.1-bs32-lr0-ep3_25 -i 0 >trial35.log 2>&1 & disown
#python3 -m code.resnet_fastai -D official -a resnet -s 512 -b 32 -d 50 -t 0.1 -S weighted -l bce -e 0 -E 25 -r 1e-5 -m stage-2-resnet50-512-official-bce-weighted-drop0.5-th0.1-bs32-lr0-ep3_25 -i 0 >trial36.log 2>&1 & disown
#python3 -m code.resnet_fastai -D official -a resnet -s 512 -b 32 -d 50 -t 0.1 -S random -l bce -e 3 -E 30 -i 0 >trial37.log 2>&1 & disown
#python3 -m code.resnet_fastai -D official -a resnet -s 512 -b 32 -d 50 -t 0.1 -S random -l bce -e 3 -E 50 -i 0 -f 2 >trial38.log 2>&1 & disown
#python3 -m code.resnet_fastai -D official -a resnet -s 256 -b 64 -d 50 -t 0.1 -S random -l f1 -e 3 -E 30 -r 0.01 -i 0 >trial39.log 2>&1 & disown
#python3 -m code.resnet_fastai -D official_hpav18 -a resnet -s 512 -b 32 -d 50 -t 0.1 -S random -l bce -e 3 -E 30 -i 0 >trial40.log 2>&1 & disown
#python3 -m code.resnet_fastai -D official_hpav18 -a resnet -s 512 -b 32 -d 50 -t 0.1 -S random -l bce -e 3 -E 30 -i 0 >trial41.log 2>&1 & disown
#python3 -m code.resnet_fastai -D official_hpav18 -a resnet -s 512 -b 32 -d 50 -t 0.1 -S random -l bce -e 0 -E 30 -i 0 -r 7e-6 -m stage-1-resnet50-512-official_hpav18-bce-random-drop0.5-th0.1-bs32-lr0-ep3_30 >trial42.log 2>&1 & disown
#python3 -m code.resnet_fastai -D official_hpav18 -a resnet -s 512 -b 32 -d 50 -t 0.1 -S random -l bce -e 3 -E 30 -i 0 -f 2 >trial43.log 2>&1 & disown
#python3 -m code.resnet_fastai -D official_hpav18 -a resnet -s 512 -b 32 -d 50 -t 0.1 -S random -l bce -e 3 -E 30 -i 0 -f 3 >trial44.log 2>&1 & disown
#python3 -m code.resnet_fastai -D official_hpav18 -a resnet -s 512 -b 32 -d 50 -t 0.1 -S random -l bce -e 3 -E 30 -i 0 -f 3 >trial45.log 2>&1 & disown
#python3 -m code.resnet_fastai -D official_hpav18 -a resnet -s 512 -b 32 -d 50 -t 0.1 -S random -l bce -e 3 -E 30 -i 0 >trial46.log 2>&1 & disown
#python3 -m code.resnet_fastai -D official_hpav18 -a resnet -s 512 -b 32 -d 50 -t 0.1 -S random -l bce -e 3 -E 30 -i 0 >trial47.log 2>&1 & disown
#python3 -m code.resnet_fastai -D official_hpav18 -a resnet -s 512 -b 32 -d 50 -t 0.1 -S random -l bce -e 3 -E 30 -i 0 -f 4 >trial48.log 2>&1 & disown
#python3 -m code.resnet_fastai -D official_hpav18 -a resnet -s 512 -b 32 -d 50 -t 0.1 -S random -l bce -e 3 -E 30 -i 0 -f 5 >trial49.log 2>&1 & disown
python3 -m code.resnet_fastai -D official_hpav18 -a resnet -s 256 -b 64 -d 50 -t 0.1 -S random -l focal -e 3 -E 30 -i 0 -f 6 >trial50.log 2>&1 & disown
