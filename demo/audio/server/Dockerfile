FROM node:10.6.0
ARG auth_user
ARG auth_pass
ARG ssl_key
ARG ssl_cert
ARG root_dir=/app
RUN mkdir -p ${root_dir}/demo/audio/server && mkdir -p ${root_dir}/src/server
ADD . ${root_dir}/demo/audio/server
ADD ./.yalc/federated-learning-server ${root_dir}/src/server
WORKDIR ${root_dir}/demo/audio/server
ENV PORT 443
ENV SSL_KEY ${root_dir}/${ssl_key}
ENV SSL_CERT ${root_dir}/${ssl_cert}
ENV BASIC_AUTH_USER ${auth_user}
ENV BASIC_AUTH_PASS ${auth_pass}
RUN yarn
CMD ["yarn", "dev"]
EXPOSE 443
