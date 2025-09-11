
@echo off

call vite build
docker build -t l2s:app -f docker\app.Dockerfile .
set container_name=l2s_container
set project_root=/d/projects/l2s
rem "C:\Program Files\Docker\Docker\Docker Desktop.exe"
docker run --rm -d -it  --name %container_name% -p 5020:5000 --mount type=bind,source=%project_root%/cache,target=/cache l2s:app
docker attach %container_name%