#!/usr/bin/env bash

container_name=l2s_container

# docker needs a full path for mounts, so set them here
if [[ $OSTYPE == darwin* ]]; then
  project_root=/Users/josh/l2s
elif [[ $OSTYPE == linux* ]]; then
  project_root=/home/josh/l2s
else
  project_root=/mnt/d/projects/nice90sguy/l2s
fi
# The only REQUIRED mount is /cache, which maintains potentially
# user-sensitive information (face landmark dataset built from webcam and model checkpoint)
docker run --rm -d -it --name "$container_name" -p 5020:5000 \
--mount type=bind,source=$project_root/cache,target=/cache \
l2s:app

docker attach "$container_name"
