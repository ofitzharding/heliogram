#!/bin/bash
set -e
cd "$(dirname "$0")"
swiftc -O SmearTest.swift -o smeartest \
  -framework Cocoa -framework Metal -framework MetalKit -framework QuartzCore
echo "built ./smeartest"
