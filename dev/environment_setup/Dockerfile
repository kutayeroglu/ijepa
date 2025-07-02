# Use the official Miniconda3 base image
FROM continuumio/miniconda3

# Set the working directory inside the container
WORKDIR /app

# Copy the environment.yml file into the container
COPY environment-cpu.yml .

# Create the Conda environment from the environment.yml file
RUN conda env create -f environment-cpu.yml

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "ijepa_py310", "/bin/bash", "-c"]

# Copy the rest of the application code into the container
COPY . .

# Define the command to run your application
CMD ["bash"]