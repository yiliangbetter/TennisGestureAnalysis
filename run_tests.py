#!/usr/bin/env python3
"""
Test runner for Tennis Gesture Analysis database tests.

Usage:
    python run_tests.py              # Run all tests
    python run_tests.py database     # Run only database tests
    python run_tests.py processor    # Run only video processor tests
    python run_tests.py --verbose    # Run with verbose output
"""

import unittest
import sys
import os


def run_tests(test_suite_name: str = None, verbose: bool = False):
    """Run tests and return results"""

    # Set up test discovery
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    if test_suite_name is None or test_suite_name == 'database':
        # Add database tests
        db_tests = loader.discover(
            start_dir=os.path.dirname(__file__),
            pattern='test_database.py'
        )
        suite.addTests(db_tests)

    if test_suite_name is None or test_suite_name == 'processor':
        # Add video processor tests
        processor_tests = loader.discover(
            start_dir=os.path.dirname(__file__),
            pattern='test_video_processor.py'
        )
        suite.addTests(processor_tests)

    # Create test runner
    if verbose:
        runner = unittest.TextTestRunner(verbosity=2)
    else:
        runner = unittest.TextTestRunner(verbosity=1)

    # Run tests
    result = runner.run(suite)

    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")

    if result.failures:
        print("\nFailed tests:")
        for test, traceback in result.failures:
            print(f"  - {test}")

    if result.errors:
        print("\nTests with errors:")
        for test, traceback in result.errors:
            print(f"  - {test}")

    # Return exit code
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    # Parse command line arguments
    test_suite = None
    verbose = False

    for arg in sys.argv[1:]:
        if arg in ('--verbose', '-v'):
            verbose = True
        elif arg in ('database', 'db'):
            test_suite = 'database'
        elif arg in ('processor', 'video', 'video_processor'):
            test_suite = 'processor'
        elif arg in ('--help', '-h'):
            print(__doc__)
            sys.exit(0)

    # Run tests
    exit_code = run_tests(test_suite, verbose)
    sys.exit(exit_code)
