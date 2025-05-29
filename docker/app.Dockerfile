FROM l2s:base
# Run this from root of project
# e.g. docker build -t l2s:app -f docker\app.Dockerfile .

# To launch the app, expose port 5000, mounting a host directory for the cache.
# e.g docker run --rm -d -it --name l2s_container -p 5020:5000 --mount type=bind,source=/d/home/someuser/l2s_cache,target=/cache l2s:app

#COPY docker/scripts/install_python_deps.sh ./
#RUN /bin/bash install_python_deps.sh

EXPOSE 5000
COPY dist ./
ENV PYTHONHASHSEED=857
ENV USE_TORCH=1
CMD [ "flask", "run", "--host=0.0.0.0" ]