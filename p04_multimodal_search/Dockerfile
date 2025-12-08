FROM python:3.10-slim

WORKDIR /app
COPY ../requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . ./p04_multimodal_search

EXPOSE 8080
CMD ["streamlit", "run", "p04_multimodal_search/app/Main.py", "--server.port", "8080", "--server.address", "0.0.0.0"]
