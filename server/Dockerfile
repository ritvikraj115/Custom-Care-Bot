FROM node:22-alpine

WORKDIR /opt/app

COPY package.json package-lock.json ./
RUN npm ci --omit=dev

COPY . .

ENV NODE_ENV=production \
    PORT=5000

EXPOSE 5000

CMD ["node", "server.js"]
