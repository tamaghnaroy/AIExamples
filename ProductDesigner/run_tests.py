#!/usr/bin/env python3
"""
Test runner for the Deep Planning LangGraph system.

This script provides an easy way to run all tests with proper configuration
and reporting.
"""

import os
import sys
import subprocess
from pathlib import Path


def setup_python_path():
    """Add the project directory to Python path for imports."""
    project_dir = Path(__file__).parent
    if str(project_dir) not in sys.path:
        sys.path.insert(0, str(project_dir))


def check_pytest_cov():
    """Check if pytest-cov is available."""
    try:
        import pytest_cov
        return True
    except ImportError:
        return False


def main():
    """Run all tests with pytest."""
    # Ensure we're in the project directory and set up Python path
    project_dir = Path(__file__).parent
    os.chdir(project_dir)
    setup_python_path()
    
    # Set up environment variables for testing
    os.environ.setdefault('OPENAI_API_KEY', 'test-key-for-mocking')
    os.environ.setdefault('MODEL_NAME', 'gpt-4')
    os.environ.setdefault('TEMPERATURE', '0.7')
    os.environ['PYTHONPATH'] = str(project_dir)
    
    # Check if pytest-cov is available
    has_coverage = check_pytest_cov()
    
    # Base pytest arguments
    pytest_args = [
        'python', '-m', 'pytest',
        'tests/',  # Test directory
        '-v',      # Verbose output
        '--tb=short',  # Short traceback format
        '--durations=10',  # Show 10 slowest tests
    ]
    
    # Add coverage arguments only if pytest-cov is available
    if has_coverage:
        pytest_args.extend([
            '--cov=.',  # Coverage for all modules
            '--cov-report=term-missing',  # Show missing lines
            '--cov-report=html',  # Generate HTML coverage report (uses .coveragerc config)
        ])
    
    print("Running Deep Planning LangGraph Tests")
    print("=" * 50)
    print(f"Project Directory: {project_dir}")
    print(f"Python Version: {sys.version}")
    
    # Check if pytest is installed
    try:
        import pytest
        print(f"Pytest version: {pytest.__version__}")
    except ImportError:
        print("Pytest not found. Installing...")
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'pytest'], check=True)
        print("Pytest installed successfully")
    
    # Check coverage availability
    if has_coverage:
        print("Coverage reporting enabled")
    else:
        print("Coverage reporting disabled (pytest-cov not installed)")
        print("   Install with: pip install pytest-cov")
    
    print()
    print("Running tests...")
    print("-" * 30)
    
    # Run the tests
    try:
        result = subprocess.run(pytest_args, check=False, env=os.environ.copy())
        
        print()
        print("=" * 50)
        
        if result.returncode == 0:
            print("All tests passed!")
            if has_coverage:
                print()
                print("Coverage report generated in 'htmlcov/' directory")
                print("   Open 'htmlcov/index.html' in your browser to view detailed coverage")
        else:
            print("Some tests failed")
            print(f"   Exit code: {result.returncode}")
        
        print()
        return result.returncode
        
    except KeyboardInterrupt:
        print("\nTests interrupted by user")
        return 1
    except Exception as e:
        print(f"Error running tests: {e}")
        return 1


def install_test_dependencies():
    """Install missing test dependencies."""
    print("Installing test dependencies...")
    dependencies = ['pytest', 'pytest-cov', 'pytest-mock', 'langchain-community']
    
    for dep in dependencies:
        try:
            # Check if dependency is already installed
            if dep == 'langchain-community':
                import langchain_community
                print(f"{dep} already installed")
            else:
                __import__(dep.replace('-', '_'))
                print(f"{dep} already installed")
        except ImportError:
            print(f"Installing {dep}...")
            subprocess.run([sys.executable, '-m', 'pip', 'install', dep], check=True)
            print(f"{dep} installed successfully")


def run_specific_test(test_file=None, test_function=None):
    """Run a specific test file or function."""
    project_dir = Path(__file__).parent
    os.chdir(project_dir)
    setup_python_path()
    
    # Set up environment variables
    os.environ.setdefault('OPENAI_API_KEY', 'test-key-for-mocking')
    os.environ['PYTHONPATH'] = str(project_dir)
    
    pytest_args = ['python', '-m', 'pytest', '-v']
    
    if test_file:
        if test_function:
            pytest_args.append(f"tests/{test_file}::{test_function}")
        else:
            pytest_args.append(f"tests/{test_file}")
    
    print(f"Running specific test: {' '.join(pytest_args[3:])}")
    print("-" * 30)
    
    result = subprocess.run(pytest_args, check=False, env=os.environ.copy())
    return result.returncode


if __name__ == "__main__":
    # Parse command line arguments for specific test runs
    if len(sys.argv) > 1:
        if sys.argv[1] == "--install-deps":
            install_test_dependencies()
            sys.exit(0)
        elif len(sys.argv) == 2:
            # Run specific test file
            exit_code = run_specific_test(test_file=sys.argv[1])
        elif len(sys.argv) == 3:
            # Run specific test function in file
            exit_code = run_specific_test(test_file=sys.argv[1], test_function=sys.argv[2])
        else:
            print("Usage:")
            print("  python run_tests.py                    # Run all tests")
            print("  python run_tests.py --install-deps     # Install test dependencies")
            print("  python run_tests.py test_base_agent.py # Run specific test file")
            print("  python run_tests.py test_base_agent.py TestBaseAgent::test_initialization")
            exit_code = 1
    else:
        # Run all tests
        exit_code = main()
    
    sys.exit(exit_code)
