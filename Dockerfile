# 1) Start from a tiny official Python image
FROM python:3.11-slim

# 2) Make Python behave well in containers
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# 3) Set the working directory inside the image
WORKDIR /app



# 4) Install Python dependencies first (better caching)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# 5) Copy in your application code and model artifacts
COPY src /app/src
COPY app.py /app/app.py
COPY model /app/model

# 6) Tell Docker which port the app uses
EXPOSE 8000

# 7) Start FastAPI with uvicorn when the container runs
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
