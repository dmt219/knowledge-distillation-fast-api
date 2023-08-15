# Deploy GPT2 model to triton server

## 1. Prepare packages
- Download the checkpoint (*.pt) and move it to ./docker/tritonserver/student_distill.pt \
  If not, the ckpt will automatic download while build docker image
- Download tritonserver images:
```
docker pull nvcr.io/nvidia/tritonserver:23.07-py3
docker pull nvcr.io/nvidia/tritonserver:23.07-py3-sdk
```
## 2. Prepare Triton server
- 2.1 Build image and start tritonserver container
```bash
PWD=$(pwd) docker compose -f ./docker/tritonserver/docker-compose.yml up -d
```
- 2.1 Stop tritonserver container
```bash
PWD=$(pwd) docker compose -f ./docker/tritonserver/docker-compose.yml down
```


## 3. Prepare Client server
- 3.1 Build image and start FastAPI container
```bash
PWD=$(pwd) docker compose -f ./docker/fastapi/docker-compose.yml up -d
```
- 3.2 Stop FastAPI container
```bash
PWD=$(pwd) docker compose -f ./docker/fastapi/docker-compose.yml down
```

## 4. Test inference
```bash
curl -X 'POST' \
  'http://localhost:5000/generate?prompt=Say somethings' \
  -H 'accept: application/json' \
  -d ''
```

