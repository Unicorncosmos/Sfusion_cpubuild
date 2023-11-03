
# Start with a base image that has the necessary dependencies installed.
FROM python:3.9-slim-buster

# Copy the requirements.txt file to the container.
COPY requirements.txt .

# Install the Python packages listed in requirements.txt using pip.
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code to the container.
COPY . .

# Set the command to run the application.
CMD [ "python", "builder.py" ]
