# Use Python 3.9 image
FROM registry.access.redhat.com/ubi8/python-311

# Set environment variables
ENV PYTHONUNBUFFERED=1

USER 1001

# Set working directory
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY MLServerLLM.py .
COPY logging.conf .
COPY settings.json .
COPY model-settings.json .
COPY test_request.py .

ENV CTX_SIZE=512
ENV MLSERVER_MODEL_URI=/tmp/models

EXPOSE 8087
EXPOSE 8088
EXPOSE 8089
# Command to run the application
CMD ["mlserver", "start", "."]
