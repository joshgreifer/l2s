cd %~dp0
set build_type=%1
if "%build_type%"=="" set build_type=dev
npm run build-%build_type%

