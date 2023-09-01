# Use Python 3.9 image
FROM registry.access.redhat.com/ubi8/python-311

# Set environment variables
ENV PYTHONUNBUFFERED=1

USER 1001

# Set working directory
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY llm_serving.py .
# Expose the port the app runs on


ENV STORAGE_URI=pvc://none
ENV MODEL_MNT=/mnt/models
ENV CTX_SIZE=512

EXPOSE 8080
# Command to run the application
CMD ["python3", "llm_serving.py"]
