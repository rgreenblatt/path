#!/usr/bin/env bash

cmake -H. -Bdebug -DCMAKE_BUILD_TYPE=Debug -DCMAKE_EXPORT_COMPILE_COMMANDS=YES
ln -s debug/compile_commands.json .
