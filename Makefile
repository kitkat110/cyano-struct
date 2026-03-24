PORT ?= 8050
NAME ?= "username/project:tag"

filter:
        docker ps --filter "expose=${PORT}"  --format "table {{.Names}}\t{{.Image}}\t{{.Ports}}\t{{.Status}}"

build:
        docker build -t ${NAME} .

run: 
        docker run -d -p 8050:8050 ${NAME}

stop:
        docker stop $(shell docker ps -q)

all: stop build run