# 1. Use an official lightweight Python runtime
FROM python:3.10-slim

# 2. Set the working directory in the container (app)
WORKDIR /app

# 3. Copy dependencies first for better caching (improved caching)
COPY requirements.txt .

# 4. Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 5. Download SpaCy model
RUN python -m spacy download en_core_web_sm

# 6. Copy the rest of the application code
COPY . .

# 7. Expose the port Streamlit runs on
EXPOSE 8501

# 8. Command to run the application
CMD ["streamlit", "run", "app/main.py", "--server.address=0.0.0.0"]