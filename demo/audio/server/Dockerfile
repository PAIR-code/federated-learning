FROM node:10.6.0
ARG auth_user
ARG auth_pass
RUN mkdir -p /app/demo/audio/server && mkdir -p /app/src/server
ADD . /app/demo/audio/server
ADD ./.yalc/federated-learning-server /app/src/server
WORKDIR /app/demo/audio/server
RUN yarn && yarn add @tensorflow/tfjs-node
RUN openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes -subj "/CN=www.example.com"
ENV SSL_KEY /app/demo/audio/server/key.pem
ENV SSL_CERT /app/demo/audio/server/cert.pem
ENV PORT 443
ENV BASIC_AUTH_USER ${auth_user}
ENV BASIC_AUTH_PASS ${auth_pass}
CMD ["yarn", "dev"]
EXPOSE 443
