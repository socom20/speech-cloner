#! /bin/bash

for file_name in *.wav
do
    echo "Converting $file_name  >>>  ${file_name/.*/.mp3}"
    ffmpeg -i $file_name -acodec libmp3lame -ac 1 -ar 16000 -ab 128k -f mp3 ${file_name/.*/.mp3}
done

# read -rsp $'Press enter to continue...\n'

