# Build a docker container which builds the app (using npm build)

FROM l2s:base
# Run this from root of project
# e.g. docker build -t l2s:build -f docker\build.Dockerfile .

# To launch the app, expose port 5000, mounting a host directory for the cache.
# e.g docker run --rm -d -it --name l2s_build -p 5020:5000 l2s:build

#COPY docker/scripts/install_python_deps.sh ./
#RUN /bin/bash install_python_deps.sh
RUN apt-get install nodejs npm  -y

COPY pkg l2s/pkg
COPY src l2s/src
COPY static l2s/static
COPY tsconfig.json l2s/
COPY webpack.config.js l2s/
COPY package.json l2s/
COPY app.py l2s/

WORKDIR l2s
RUN npm install
EXPOSE 5000
ENV PYTHONHASHSEED=857
ENV USE_TORCH=1
# CMD [ "python", "app.py" ]