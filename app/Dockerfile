FROM python:3.12-slim
WORKDIR /app

# Install required OS packages
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \

# copy and install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy all project files into a workdir
COPY api/ ./api/
COPY models/ ./models/

# Define environment variable
ENV FLASK_APP=api/flask_app.py
ENV FLASK_ENV=development

# Make port 5001 available to the world outside this container
EXPOSE 5001

# Run app.py when the container launches
CMD ["flask", "--app", "api/flask_app.py", "run", "--port=5001", "--host=0.0.0.0"]
# CMD ["python", "api/flask_app.py"]
