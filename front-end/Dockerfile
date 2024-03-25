# Install node image in container
<<<<<<< HEAD
FROM node:11.10.1
=======
FROM node:16
>>>>>>> 0f3f6fb (Initial commit)

# Create and set the working directory
RUN mkdir -p /app/client
WORKDIR /app/client

# Copy the dependency files to the container
COPY package*.json /app/client/

# Install dependencies
RUN npm install

# Copy the server files to the container
COPY . /app/client/

# Run the client
CMD [ "npm", "start" ]
