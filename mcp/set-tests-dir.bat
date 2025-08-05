@echo off
if "%~1"=="" (
    set TESTS_DIR=%~dp0tests\generated_tests
) else (
    set TESTS_DIR=%~1
)
echo Tests directory set to: %TESTS_DIR%