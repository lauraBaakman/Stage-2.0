#!/bin/bash 

# Remove python executable
rm -f _kde.cpython-35m-darwin.so

# Run the unittests
cd tests
make test && ./test_main.out

exit $?