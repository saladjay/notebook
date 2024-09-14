#!/usr/bin/env python3
import example_array_pb2
import sys




# Main procedure:  Reads the entire address book from a file,
#   adds one person based on user input, then writes it back out to the same
#   file.
if len(sys.argv) != 2:
  print("Usage:", sys.argv[0], "FLOAT_ARRAY_FILE")
  sys.exit(-1)

float_array = example_array_pb2.ExampleArray

# Read the existing address book.
# try:
#   with open(sys.argv[1], "rb") as f:
#     float_array.ParseFromString(f.read())
# except IOError:
#   print(sys.argv[1] + ": Could not open file.  Creating a new one.")

# Add an address.
float_array.name = "float_array"
print(dir(float_array))
print(dir(float_array.values))
# float_array.values.extend([1.1, 2.2, 3.3, 4.4, 5.5])

# Write the new address book back to disk.
with open(sys.argv[1], "wb") as f:
  f.write(float_array.SerializeToString())