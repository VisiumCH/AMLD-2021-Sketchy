FROM node:lts-alpine

RUN npm install -g http-server

WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .

RUN npm run build
CMD http-server --proxy http://localhost:8080? ./build
# CMD npm run start -- --port $PORT --hostname 0.0.0.0 --disable-host-check