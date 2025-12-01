#!/bin/bash

# Combined launcher for API + UI
# This script starts both the FastAPI backend and Gradio frontend

echo "Starting Sentiment Analysis System..."
echo ""

# Start FastAPI in the background
echo "Starting API server on http://localhost:8000..."
uvicorn app.main:app --host 0.0.0.0 --port 8000 &
API_PID=$!

# Wait for API to be ready
sleep 3

# Start Gradio UI
echo "Starting Web UI on http://localhost:7860..."
python ui/gradio_app.py &
UI_PID=$!

echo ""
echo "System is ready!"
echo "   API Docs: http://localhost:8000/docs"
echo "   Web UI:   http://localhost:7860"
echo ""
echo "Press Ctrl+C to stop both servers"

# Wait for both processes
trap "kill $API_PID $UI_PID; exit" INT
wait
