# StakTrak Tests

This directory contains tests for the StakTrak application.

## Test File Storage

By default, test files are stored in the `tests/generated_tests` directory. This directory is ignored by git, so your local test files won't be committed.

## Setting a Custom Test Directory

First, navigate to the mcp directory:

```
cd mcp
```

### Windows (Git Bash) - Working Commands

```
# Set custom tests directory (specific example)
./set-tests-dir.sh /d/je1/stakgraph/custom_tests

# OR directly export the environment variable
export TESTS_DIR="/d/je1/stakgraph/custom_tests"

# Start the application
NO_DB=1 npm run dev
```

### Generic Commands for Any Environment

```
# Set custom tests directory (generic example)
./set-tests-dir.sh /path/to/custom_tests

# OR directly export the environment variable
export TESTS_DIR="/path/to/custom_tests"

# Start the application
NO_DB=1 npm run dev
```

**Note:** Use `/d/` format instead of `D:\` for Windows drives in Git Bash.

### Windows (Command Prompt)

```
set TESTS_DIR=D:\path\to\custom_tests
```

### Windows (PowerShell)

```
$env:TESTS_DIR="D:\path\to\custom_tests"
```

### Linux/Mac

```
export TESTS_DIR="/path/to/custom_tests"
```

**Note:** After changing the test directory, restart the application for changes to take effect.

## Running Tests from Custom Directory

When using a custom tests directory, the system will:

1. **Automatically find your test files** - You can store test files anywhere
2. **Use the default configuration** - No need to create your own config files
3. **Handle all path translations** - Tests run correctly regardless of location

For the best experience:

- Make sure your test files end with `.spec.js` or `.spec.ts`
- Avoid using dependencies that aren't installed in the main project

## Resetting Test Directory

To reset or unset the custom test directory:

### Windows (Git Bash)

```
unset TESTS_DIR
```

### Windows (Command Prompt)

```
set TESTS_DIR=
```

### Windows (PowerShell)

```
Remove-Item Env:\TESTS_DIR
```

### Linux/Mac

```
unset TESTS_DIR
```
