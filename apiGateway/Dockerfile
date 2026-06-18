FROM node:22-alpine

WORKDIR /apiGateway

COPY package*.json ./
RUN npm ci

COPY . .
RUN npm run build

EXPOSE 8080 9898

CMD ["npm", "start"]
