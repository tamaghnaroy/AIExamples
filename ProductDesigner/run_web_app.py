#!/usr/bin/env python3
"""
Startup script for the Deep Planning LangGraph Web Application.
This script provides an easy way to run the web interface.
"""

import os
import sys
from dotenv import load_dotenv

def check_environment():
    """Check if the required environment variables are set."""
    load_dotenv()
    
    required_vars = ['OPENAI_API_KEY']
    missing_vars = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print("❌ Missing required environment variables:")
        for var in missing_vars:
            print(f"   - {var}")
        print("\nPlease copy .env.example to .env and fill in the required values.")
        return False
    
    return True

def main():
    """Main entry point for the web application."""
    print("Starting Deep Planning LangGraph Web Application...")
    
    # Check environment
    if not check_environment():
        sys.exit(1)
    
    print("Environment check passed")
    print("Starting web server...")
    print("Open your browser to: http://localhost:5000")
    print("Press Ctrl+C to stop the server")
    print("-" * 50)
    
    # Import and run the web application
    try:
        from web_app import app, socketio
        socketio.run(app, debug=True, host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
    except Exception as e:
        print(f"Error starting web application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
