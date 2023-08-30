@echo off
cd %~dp0
set container_name=%1
if "%container_name%"=="" set container_name=l2s_container

rem "C:\Program Files\Docker\Docker\Docker Desktop.exe"
docker run --rm -d -it --name %container_name% -p 5020:5000 --mount type=bind,source=/d/projects/nice90sguy/l2s/cache,target=/cache --mount type=bind,source=/d/projects/nice90sguy/l2s/dist/static,target=/static --mount type=bind,source=/d/projects/nice90sguy/l2s/dist/pkg,target=/pkg l2s:app
docker attach %container_name%