#!/usr/bin/env bash

container_name=l2s_container

if [[ $OSTYPE == darwin* ]]; then
  project_root=/Users/josh/l2s
else
  project_root=/mnt/d/projects/nice90sguy/l2s
fi

docker run --rm -d -it --name "$container_name" -p 5020:5000 \
--mount type=bind,source=$project_root/cache,target=/cache \
--mount type=bind,source=$project_root/dist/static,target=/static \
--mount type=bind,source=$project_root/dist/pkg,target=/pkg \
l2s:app

docker attach "$container_name"
