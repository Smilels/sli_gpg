# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.5

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /homeL/demo/overlays/sources/sli_gpg

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /homeL/demo/overlays/sources/sli_gpg/build

# Include any dependencies generated for this target.
include CMakeFiles/sli_gpg_eigen_utils.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/sli_gpg_eigen_utils.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/sli_gpg_eigen_utils.dir/flags.make

CMakeFiles/sli_gpg_eigen_utils.dir/src/sli_gpg/eigen_utils.cpp.o: CMakeFiles/sli_gpg_eigen_utils.dir/flags.make
CMakeFiles/sli_gpg_eigen_utils.dir/src/sli_gpg/eigen_utils.cpp.o: ../src/sli_gpg/eigen_utils.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/homeL/demo/overlays/sources/sli_gpg/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/sli_gpg_eigen_utils.dir/src/sli_gpg/eigen_utils.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/sli_gpg_eigen_utils.dir/src/sli_gpg/eigen_utils.cpp.o -c /homeL/demo/overlays/sources/sli_gpg/src/sli_gpg/eigen_utils.cpp

CMakeFiles/sli_gpg_eigen_utils.dir/src/sli_gpg/eigen_utils.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/sli_gpg_eigen_utils.dir/src/sli_gpg/eigen_utils.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /homeL/demo/overlays/sources/sli_gpg/src/sli_gpg/eigen_utils.cpp > CMakeFiles/sli_gpg_eigen_utils.dir/src/sli_gpg/eigen_utils.cpp.i

CMakeFiles/sli_gpg_eigen_utils.dir/src/sli_gpg/eigen_utils.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/sli_gpg_eigen_utils.dir/src/sli_gpg/eigen_utils.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /homeL/demo/overlays/sources/sli_gpg/src/sli_gpg/eigen_utils.cpp -o CMakeFiles/sli_gpg_eigen_utils.dir/src/sli_gpg/eigen_utils.cpp.s

CMakeFiles/sli_gpg_eigen_utils.dir/src/sli_gpg/eigen_utils.cpp.o.requires:

.PHONY : CMakeFiles/sli_gpg_eigen_utils.dir/src/sli_gpg/eigen_utils.cpp.o.requires

CMakeFiles/sli_gpg_eigen_utils.dir/src/sli_gpg/eigen_utils.cpp.o.provides: CMakeFiles/sli_gpg_eigen_utils.dir/src/sli_gpg/eigen_utils.cpp.o.requires
	$(MAKE) -f CMakeFiles/sli_gpg_eigen_utils.dir/build.make CMakeFiles/sli_gpg_eigen_utils.dir/src/sli_gpg/eigen_utils.cpp.o.provides.build
.PHONY : CMakeFiles/sli_gpg_eigen_utils.dir/src/sli_gpg/eigen_utils.cpp.o.provides

CMakeFiles/sli_gpg_eigen_utils.dir/src/sli_gpg/eigen_utils.cpp.o.provides.build: CMakeFiles/sli_gpg_eigen_utils.dir/src/sli_gpg/eigen_utils.cpp.o


# Object files for target sli_gpg_eigen_utils
sli_gpg_eigen_utils_OBJECTS = \
"CMakeFiles/sli_gpg_eigen_utils.dir/src/sli_gpg/eigen_utils.cpp.o"

# External object files for target sli_gpg_eigen_utils
sli_gpg_eigen_utils_EXTERNAL_OBJECTS =

libsli_gpg_eigen_utils.a: CMakeFiles/sli_gpg_eigen_utils.dir/src/sli_gpg/eigen_utils.cpp.o
libsli_gpg_eigen_utils.a: CMakeFiles/sli_gpg_eigen_utils.dir/build.make
libsli_gpg_eigen_utils.a: CMakeFiles/sli_gpg_eigen_utils.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/homeL/demo/overlays/sources/sli_gpg/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX static library libsli_gpg_eigen_utils.a"
	$(CMAKE_COMMAND) -P CMakeFiles/sli_gpg_eigen_utils.dir/cmake_clean_target.cmake
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/sli_gpg_eigen_utils.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/sli_gpg_eigen_utils.dir/build: libsli_gpg_eigen_utils.a

.PHONY : CMakeFiles/sli_gpg_eigen_utils.dir/build

CMakeFiles/sli_gpg_eigen_utils.dir/requires: CMakeFiles/sli_gpg_eigen_utils.dir/src/sli_gpg/eigen_utils.cpp.o.requires

.PHONY : CMakeFiles/sli_gpg_eigen_utils.dir/requires

CMakeFiles/sli_gpg_eigen_utils.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/sli_gpg_eigen_utils.dir/cmake_clean.cmake
.PHONY : CMakeFiles/sli_gpg_eigen_utils.dir/clean

CMakeFiles/sli_gpg_eigen_utils.dir/depend:
	cd /homeL/demo/overlays/sources/sli_gpg/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /homeL/demo/overlays/sources/sli_gpg /homeL/demo/overlays/sources/sli_gpg /homeL/demo/overlays/sources/sli_gpg/build /homeL/demo/overlays/sources/sli_gpg/build /homeL/demo/overlays/sources/sli_gpg/build/CMakeFiles/sli_gpg_eigen_utils.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/sli_gpg_eigen_utils.dir/depend

