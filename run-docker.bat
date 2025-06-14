@echo off
cd %~dp0
set container_name=%1
if "%container_name%"=="" set container_name=l2s_container
set project_root=/d/projects/nice90sguy/l2s
rem "C:\Program Files\Docker\Docker\Docker Desktop.exe"
docker run --rm -d -it  --name %container_name% -p 5020:5000 --mount type=bind,source=%project_root%/cache,target=/cache l2s:app
docker attach %container_name%