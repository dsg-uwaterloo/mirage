# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.29

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /opt/homebrew/Cellar/cmake/3.29.0/bin/cmake

# The command to remove a file.
RM = /opt/homebrew/Cellar/cmake/3.29.0/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/sairaj/faiss

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/sairaj/faiss/build

# Include any dependencies generated for this target.
include c_api/CMakeFiles/faiss_c.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include c_api/CMakeFiles/faiss_c.dir/compiler_depend.make

# Include the progress variables for this target.
include c_api/CMakeFiles/faiss_c.dir/progress.make

# Include the compile flags for this target's objects.
include c_api/CMakeFiles/faiss_c.dir/flags.make

c_api/CMakeFiles/faiss_c.dir/AutoTune_c.cpp.o: c_api/CMakeFiles/faiss_c.dir/flags.make
c_api/CMakeFiles/faiss_c.dir/AutoTune_c.cpp.o: /Users/sairaj/faiss/c_api/AutoTune_c.cpp
c_api/CMakeFiles/faiss_c.dir/AutoTune_c.cpp.o: c_api/CMakeFiles/faiss_c.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/sairaj/faiss/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object c_api/CMakeFiles/faiss_c.dir/AutoTune_c.cpp.o"
	cd /Users/sairaj/faiss/build/c_api && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT c_api/CMakeFiles/faiss_c.dir/AutoTune_c.cpp.o -MF CMakeFiles/faiss_c.dir/AutoTune_c.cpp.o.d -o CMakeFiles/faiss_c.dir/AutoTune_c.cpp.o -c /Users/sairaj/faiss/c_api/AutoTune_c.cpp

c_api/CMakeFiles/faiss_c.dir/AutoTune_c.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/faiss_c.dir/AutoTune_c.cpp.i"
	cd /Users/sairaj/faiss/build/c_api && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/sairaj/faiss/c_api/AutoTune_c.cpp > CMakeFiles/faiss_c.dir/AutoTune_c.cpp.i

c_api/CMakeFiles/faiss_c.dir/AutoTune_c.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/faiss_c.dir/AutoTune_c.cpp.s"
	cd /Users/sairaj/faiss/build/c_api && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/sairaj/faiss/c_api/AutoTune_c.cpp -o CMakeFiles/faiss_c.dir/AutoTune_c.cpp.s

c_api/CMakeFiles/faiss_c.dir/Clustering_c.cpp.o: c_api/CMakeFiles/faiss_c.dir/flags.make
c_api/CMakeFiles/faiss_c.dir/Clustering_c.cpp.o: /Users/sairaj/faiss/c_api/Clustering_c.cpp
c_api/CMakeFiles/faiss_c.dir/Clustering_c.cpp.o: c_api/CMakeFiles/faiss_c.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/sairaj/faiss/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object c_api/CMakeFiles/faiss_c.dir/Clustering_c.cpp.o"
	cd /Users/sairaj/faiss/build/c_api && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT c_api/CMakeFiles/faiss_c.dir/Clustering_c.cpp.o -MF CMakeFiles/faiss_c.dir/Clustering_c.cpp.o.d -o CMakeFiles/faiss_c.dir/Clustering_c.cpp.o -c /Users/sairaj/faiss/c_api/Clustering_c.cpp

c_api/CMakeFiles/faiss_c.dir/Clustering_c.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/faiss_c.dir/Clustering_c.cpp.i"
	cd /Users/sairaj/faiss/build/c_api && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/sairaj/faiss/c_api/Clustering_c.cpp > CMakeFiles/faiss_c.dir/Clustering_c.cpp.i

c_api/CMakeFiles/faiss_c.dir/Clustering_c.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/faiss_c.dir/Clustering_c.cpp.s"
	cd /Users/sairaj/faiss/build/c_api && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/sairaj/faiss/c_api/Clustering_c.cpp -o CMakeFiles/faiss_c.dir/Clustering_c.cpp.s

c_api/CMakeFiles/faiss_c.dir/IndexFlat_c.cpp.o: c_api/CMakeFiles/faiss_c.dir/flags.make
c_api/CMakeFiles/faiss_c.dir/IndexFlat_c.cpp.o: /Users/sairaj/faiss/c_api/IndexFlat_c.cpp
c_api/CMakeFiles/faiss_c.dir/IndexFlat_c.cpp.o: c_api/CMakeFiles/faiss_c.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/sairaj/faiss/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object c_api/CMakeFiles/faiss_c.dir/IndexFlat_c.cpp.o"
	cd /Users/sairaj/faiss/build/c_api && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT c_api/CMakeFiles/faiss_c.dir/IndexFlat_c.cpp.o -MF CMakeFiles/faiss_c.dir/IndexFlat_c.cpp.o.d -o CMakeFiles/faiss_c.dir/IndexFlat_c.cpp.o -c /Users/sairaj/faiss/c_api/IndexFlat_c.cpp

c_api/CMakeFiles/faiss_c.dir/IndexFlat_c.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/faiss_c.dir/IndexFlat_c.cpp.i"
	cd /Users/sairaj/faiss/build/c_api && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/sairaj/faiss/c_api/IndexFlat_c.cpp > CMakeFiles/faiss_c.dir/IndexFlat_c.cpp.i

c_api/CMakeFiles/faiss_c.dir/IndexFlat_c.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/faiss_c.dir/IndexFlat_c.cpp.s"
	cd /Users/sairaj/faiss/build/c_api && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/sairaj/faiss/c_api/IndexFlat_c.cpp -o CMakeFiles/faiss_c.dir/IndexFlat_c.cpp.s

c_api/CMakeFiles/faiss_c.dir/IndexIVFFlat_c.cpp.o: c_api/CMakeFiles/faiss_c.dir/flags.make
c_api/CMakeFiles/faiss_c.dir/IndexIVFFlat_c.cpp.o: /Users/sairaj/faiss/c_api/IndexIVFFlat_c.cpp
c_api/CMakeFiles/faiss_c.dir/IndexIVFFlat_c.cpp.o: c_api/CMakeFiles/faiss_c.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/sairaj/faiss/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object c_api/CMakeFiles/faiss_c.dir/IndexIVFFlat_c.cpp.o"
	cd /Users/sairaj/faiss/build/c_api && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT c_api/CMakeFiles/faiss_c.dir/IndexIVFFlat_c.cpp.o -MF CMakeFiles/faiss_c.dir/IndexIVFFlat_c.cpp.o.d -o CMakeFiles/faiss_c.dir/IndexIVFFlat_c.cpp.o -c /Users/sairaj/faiss/c_api/IndexIVFFlat_c.cpp

c_api/CMakeFiles/faiss_c.dir/IndexIVFFlat_c.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/faiss_c.dir/IndexIVFFlat_c.cpp.i"
	cd /Users/sairaj/faiss/build/c_api && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/sairaj/faiss/c_api/IndexIVFFlat_c.cpp > CMakeFiles/faiss_c.dir/IndexIVFFlat_c.cpp.i

c_api/CMakeFiles/faiss_c.dir/IndexIVFFlat_c.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/faiss_c.dir/IndexIVFFlat_c.cpp.s"
	cd /Users/sairaj/faiss/build/c_api && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/sairaj/faiss/c_api/IndexIVFFlat_c.cpp -o CMakeFiles/faiss_c.dir/IndexIVFFlat_c.cpp.s

c_api/CMakeFiles/faiss_c.dir/IndexIVF_c.cpp.o: c_api/CMakeFiles/faiss_c.dir/flags.make
c_api/CMakeFiles/faiss_c.dir/IndexIVF_c.cpp.o: /Users/sairaj/faiss/c_api/IndexIVF_c.cpp
c_api/CMakeFiles/faiss_c.dir/IndexIVF_c.cpp.o: c_api/CMakeFiles/faiss_c.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/sairaj/faiss/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object c_api/CMakeFiles/faiss_c.dir/IndexIVF_c.cpp.o"
	cd /Users/sairaj/faiss/build/c_api && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT c_api/CMakeFiles/faiss_c.dir/IndexIVF_c.cpp.o -MF CMakeFiles/faiss_c.dir/IndexIVF_c.cpp.o.d -o CMakeFiles/faiss_c.dir/IndexIVF_c.cpp.o -c /Users/sairaj/faiss/c_api/IndexIVF_c.cpp

c_api/CMakeFiles/faiss_c.dir/IndexIVF_c.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/faiss_c.dir/IndexIVF_c.cpp.i"
	cd /Users/sairaj/faiss/build/c_api && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/sairaj/faiss/c_api/IndexIVF_c.cpp > CMakeFiles/faiss_c.dir/IndexIVF_c.cpp.i

c_api/CMakeFiles/faiss_c.dir/IndexIVF_c.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/faiss_c.dir/IndexIVF_c.cpp.s"
	cd /Users/sairaj/faiss/build/c_api && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/sairaj/faiss/c_api/IndexIVF_c.cpp -o CMakeFiles/faiss_c.dir/IndexIVF_c.cpp.s

c_api/CMakeFiles/faiss_c.dir/IndexLSH_c.cpp.o: c_api/CMakeFiles/faiss_c.dir/flags.make
c_api/CMakeFiles/faiss_c.dir/IndexLSH_c.cpp.o: /Users/sairaj/faiss/c_api/IndexLSH_c.cpp
c_api/CMakeFiles/faiss_c.dir/IndexLSH_c.cpp.o: c_api/CMakeFiles/faiss_c.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/sairaj/faiss/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CXX object c_api/CMakeFiles/faiss_c.dir/IndexLSH_c.cpp.o"
	cd /Users/sairaj/faiss/build/c_api && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT c_api/CMakeFiles/faiss_c.dir/IndexLSH_c.cpp.o -MF CMakeFiles/faiss_c.dir/IndexLSH_c.cpp.o.d -o CMakeFiles/faiss_c.dir/IndexLSH_c.cpp.o -c /Users/sairaj/faiss/c_api/IndexLSH_c.cpp

c_api/CMakeFiles/faiss_c.dir/IndexLSH_c.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/faiss_c.dir/IndexLSH_c.cpp.i"
	cd /Users/sairaj/faiss/build/c_api && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/sairaj/faiss/c_api/IndexLSH_c.cpp > CMakeFiles/faiss_c.dir/IndexLSH_c.cpp.i

c_api/CMakeFiles/faiss_c.dir/IndexLSH_c.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/faiss_c.dir/IndexLSH_c.cpp.s"
	cd /Users/sairaj/faiss/build/c_api && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/sairaj/faiss/c_api/IndexLSH_c.cpp -o CMakeFiles/faiss_c.dir/IndexLSH_c.cpp.s

c_api/CMakeFiles/faiss_c.dir/IndexPreTransform_c.cpp.o: c_api/CMakeFiles/faiss_c.dir/flags.make
c_api/CMakeFiles/faiss_c.dir/IndexPreTransform_c.cpp.o: /Users/sairaj/faiss/c_api/IndexPreTransform_c.cpp
c_api/CMakeFiles/faiss_c.dir/IndexPreTransform_c.cpp.o: c_api/CMakeFiles/faiss_c.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/sairaj/faiss/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Building CXX object c_api/CMakeFiles/faiss_c.dir/IndexPreTransform_c.cpp.o"
	cd /Users/sairaj/faiss/build/c_api && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT c_api/CMakeFiles/faiss_c.dir/IndexPreTransform_c.cpp.o -MF CMakeFiles/faiss_c.dir/IndexPreTransform_c.cpp.o.d -o CMakeFiles/faiss_c.dir/IndexPreTransform_c.cpp.o -c /Users/sairaj/faiss/c_api/IndexPreTransform_c.cpp

c_api/CMakeFiles/faiss_c.dir/IndexPreTransform_c.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/faiss_c.dir/IndexPreTransform_c.cpp.i"
	cd /Users/sairaj/faiss/build/c_api && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/sairaj/faiss/c_api/IndexPreTransform_c.cpp > CMakeFiles/faiss_c.dir/IndexPreTransform_c.cpp.i

c_api/CMakeFiles/faiss_c.dir/IndexPreTransform_c.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/faiss_c.dir/IndexPreTransform_c.cpp.s"
	cd /Users/sairaj/faiss/build/c_api && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/sairaj/faiss/c_api/IndexPreTransform_c.cpp -o CMakeFiles/faiss_c.dir/IndexPreTransform_c.cpp.s

c_api/CMakeFiles/faiss_c.dir/VectorTransform_c.cpp.o: c_api/CMakeFiles/faiss_c.dir/flags.make
c_api/CMakeFiles/faiss_c.dir/VectorTransform_c.cpp.o: /Users/sairaj/faiss/c_api/VectorTransform_c.cpp
c_api/CMakeFiles/faiss_c.dir/VectorTransform_c.cpp.o: c_api/CMakeFiles/faiss_c.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/sairaj/faiss/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "Building CXX object c_api/CMakeFiles/faiss_c.dir/VectorTransform_c.cpp.o"
	cd /Users/sairaj/faiss/build/c_api && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT c_api/CMakeFiles/faiss_c.dir/VectorTransform_c.cpp.o -MF CMakeFiles/faiss_c.dir/VectorTransform_c.cpp.o.d -o CMakeFiles/faiss_c.dir/VectorTransform_c.cpp.o -c /Users/sairaj/faiss/c_api/VectorTransform_c.cpp

c_api/CMakeFiles/faiss_c.dir/VectorTransform_c.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/faiss_c.dir/VectorTransform_c.cpp.i"
	cd /Users/sairaj/faiss/build/c_api && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/sairaj/faiss/c_api/VectorTransform_c.cpp > CMakeFiles/faiss_c.dir/VectorTransform_c.cpp.i

c_api/CMakeFiles/faiss_c.dir/VectorTransform_c.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/faiss_c.dir/VectorTransform_c.cpp.s"
	cd /Users/sairaj/faiss/build/c_api && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/sairaj/faiss/c_api/VectorTransform_c.cpp -o CMakeFiles/faiss_c.dir/VectorTransform_c.cpp.s

c_api/CMakeFiles/faiss_c.dir/IndexShards_c.cpp.o: c_api/CMakeFiles/faiss_c.dir/flags.make
c_api/CMakeFiles/faiss_c.dir/IndexShards_c.cpp.o: /Users/sairaj/faiss/c_api/IndexShards_c.cpp
c_api/CMakeFiles/faiss_c.dir/IndexShards_c.cpp.o: c_api/CMakeFiles/faiss_c.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/sairaj/faiss/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_9) "Building CXX object c_api/CMakeFiles/faiss_c.dir/IndexShards_c.cpp.o"
	cd /Users/sairaj/faiss/build/c_api && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT c_api/CMakeFiles/faiss_c.dir/IndexShards_c.cpp.o -MF CMakeFiles/faiss_c.dir/IndexShards_c.cpp.o.d -o CMakeFiles/faiss_c.dir/IndexShards_c.cpp.o -c /Users/sairaj/faiss/c_api/IndexShards_c.cpp

c_api/CMakeFiles/faiss_c.dir/IndexShards_c.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/faiss_c.dir/IndexShards_c.cpp.i"
	cd /Users/sairaj/faiss/build/c_api && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/sairaj/faiss/c_api/IndexShards_c.cpp > CMakeFiles/faiss_c.dir/IndexShards_c.cpp.i

c_api/CMakeFiles/faiss_c.dir/IndexShards_c.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/faiss_c.dir/IndexShards_c.cpp.s"
	cd /Users/sairaj/faiss/build/c_api && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/sairaj/faiss/c_api/IndexShards_c.cpp -o CMakeFiles/faiss_c.dir/IndexShards_c.cpp.s

c_api/CMakeFiles/faiss_c.dir/IndexReplicas_c.cpp.o: c_api/CMakeFiles/faiss_c.dir/flags.make
c_api/CMakeFiles/faiss_c.dir/IndexReplicas_c.cpp.o: /Users/sairaj/faiss/c_api/IndexReplicas_c.cpp
c_api/CMakeFiles/faiss_c.dir/IndexReplicas_c.cpp.o: c_api/CMakeFiles/faiss_c.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/sairaj/faiss/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_10) "Building CXX object c_api/CMakeFiles/faiss_c.dir/IndexReplicas_c.cpp.o"
	cd /Users/sairaj/faiss/build/c_api && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT c_api/CMakeFiles/faiss_c.dir/IndexReplicas_c.cpp.o -MF CMakeFiles/faiss_c.dir/IndexReplicas_c.cpp.o.d -o CMakeFiles/faiss_c.dir/IndexReplicas_c.cpp.o -c /Users/sairaj/faiss/c_api/IndexReplicas_c.cpp

c_api/CMakeFiles/faiss_c.dir/IndexReplicas_c.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/faiss_c.dir/IndexReplicas_c.cpp.i"
	cd /Users/sairaj/faiss/build/c_api && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/sairaj/faiss/c_api/IndexReplicas_c.cpp > CMakeFiles/faiss_c.dir/IndexReplicas_c.cpp.i

c_api/CMakeFiles/faiss_c.dir/IndexReplicas_c.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/faiss_c.dir/IndexReplicas_c.cpp.s"
	cd /Users/sairaj/faiss/build/c_api && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/sairaj/faiss/c_api/IndexReplicas_c.cpp -o CMakeFiles/faiss_c.dir/IndexReplicas_c.cpp.s

c_api/CMakeFiles/faiss_c.dir/Index_c.cpp.o: c_api/CMakeFiles/faiss_c.dir/flags.make
c_api/CMakeFiles/faiss_c.dir/Index_c.cpp.o: /Users/sairaj/faiss/c_api/Index_c.cpp
c_api/CMakeFiles/faiss_c.dir/Index_c.cpp.o: c_api/CMakeFiles/faiss_c.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/sairaj/faiss/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_11) "Building CXX object c_api/CMakeFiles/faiss_c.dir/Index_c.cpp.o"
	cd /Users/sairaj/faiss/build/c_api && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT c_api/CMakeFiles/faiss_c.dir/Index_c.cpp.o -MF CMakeFiles/faiss_c.dir/Index_c.cpp.o.d -o CMakeFiles/faiss_c.dir/Index_c.cpp.o -c /Users/sairaj/faiss/c_api/Index_c.cpp

c_api/CMakeFiles/faiss_c.dir/Index_c.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/faiss_c.dir/Index_c.cpp.i"
	cd /Users/sairaj/faiss/build/c_api && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/sairaj/faiss/c_api/Index_c.cpp > CMakeFiles/faiss_c.dir/Index_c.cpp.i

c_api/CMakeFiles/faiss_c.dir/Index_c.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/faiss_c.dir/Index_c.cpp.s"
	cd /Users/sairaj/faiss/build/c_api && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/sairaj/faiss/c_api/Index_c.cpp -o CMakeFiles/faiss_c.dir/Index_c.cpp.s

c_api/CMakeFiles/faiss_c.dir/IndexBinary_c.cpp.o: c_api/CMakeFiles/faiss_c.dir/flags.make
c_api/CMakeFiles/faiss_c.dir/IndexBinary_c.cpp.o: /Users/sairaj/faiss/c_api/IndexBinary_c.cpp
c_api/CMakeFiles/faiss_c.dir/IndexBinary_c.cpp.o: c_api/CMakeFiles/faiss_c.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/sairaj/faiss/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_12) "Building CXX object c_api/CMakeFiles/faiss_c.dir/IndexBinary_c.cpp.o"
	cd /Users/sairaj/faiss/build/c_api && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT c_api/CMakeFiles/faiss_c.dir/IndexBinary_c.cpp.o -MF CMakeFiles/faiss_c.dir/IndexBinary_c.cpp.o.d -o CMakeFiles/faiss_c.dir/IndexBinary_c.cpp.o -c /Users/sairaj/faiss/c_api/IndexBinary_c.cpp

c_api/CMakeFiles/faiss_c.dir/IndexBinary_c.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/faiss_c.dir/IndexBinary_c.cpp.i"
	cd /Users/sairaj/faiss/build/c_api && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/sairaj/faiss/c_api/IndexBinary_c.cpp > CMakeFiles/faiss_c.dir/IndexBinary_c.cpp.i

c_api/CMakeFiles/faiss_c.dir/IndexBinary_c.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/faiss_c.dir/IndexBinary_c.cpp.s"
	cd /Users/sairaj/faiss/build/c_api && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/sairaj/faiss/c_api/IndexBinary_c.cpp -o CMakeFiles/faiss_c.dir/IndexBinary_c.cpp.s

c_api/CMakeFiles/faiss_c.dir/IndexScalarQuantizer_c.cpp.o: c_api/CMakeFiles/faiss_c.dir/flags.make
c_api/CMakeFiles/faiss_c.dir/IndexScalarQuantizer_c.cpp.o: /Users/sairaj/faiss/c_api/IndexScalarQuantizer_c.cpp
c_api/CMakeFiles/faiss_c.dir/IndexScalarQuantizer_c.cpp.o: c_api/CMakeFiles/faiss_c.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/sairaj/faiss/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_13) "Building CXX object c_api/CMakeFiles/faiss_c.dir/IndexScalarQuantizer_c.cpp.o"
	cd /Users/sairaj/faiss/build/c_api && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT c_api/CMakeFiles/faiss_c.dir/IndexScalarQuantizer_c.cpp.o -MF CMakeFiles/faiss_c.dir/IndexScalarQuantizer_c.cpp.o.d -o CMakeFiles/faiss_c.dir/IndexScalarQuantizer_c.cpp.o -c /Users/sairaj/faiss/c_api/IndexScalarQuantizer_c.cpp

c_api/CMakeFiles/faiss_c.dir/IndexScalarQuantizer_c.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/faiss_c.dir/IndexScalarQuantizer_c.cpp.i"
	cd /Users/sairaj/faiss/build/c_api && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/sairaj/faiss/c_api/IndexScalarQuantizer_c.cpp > CMakeFiles/faiss_c.dir/IndexScalarQuantizer_c.cpp.i

c_api/CMakeFiles/faiss_c.dir/IndexScalarQuantizer_c.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/faiss_c.dir/IndexScalarQuantizer_c.cpp.s"
	cd /Users/sairaj/faiss/build/c_api && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/sairaj/faiss/c_api/IndexScalarQuantizer_c.cpp -o CMakeFiles/faiss_c.dir/IndexScalarQuantizer_c.cpp.s

c_api/CMakeFiles/faiss_c.dir/MetaIndexes_c.cpp.o: c_api/CMakeFiles/faiss_c.dir/flags.make
c_api/CMakeFiles/faiss_c.dir/MetaIndexes_c.cpp.o: /Users/sairaj/faiss/c_api/MetaIndexes_c.cpp
c_api/CMakeFiles/faiss_c.dir/MetaIndexes_c.cpp.o: c_api/CMakeFiles/faiss_c.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/sairaj/faiss/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_14) "Building CXX object c_api/CMakeFiles/faiss_c.dir/MetaIndexes_c.cpp.o"
	cd /Users/sairaj/faiss/build/c_api && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT c_api/CMakeFiles/faiss_c.dir/MetaIndexes_c.cpp.o -MF CMakeFiles/faiss_c.dir/MetaIndexes_c.cpp.o.d -o CMakeFiles/faiss_c.dir/MetaIndexes_c.cpp.o -c /Users/sairaj/faiss/c_api/MetaIndexes_c.cpp

c_api/CMakeFiles/faiss_c.dir/MetaIndexes_c.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/faiss_c.dir/MetaIndexes_c.cpp.i"
	cd /Users/sairaj/faiss/build/c_api && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/sairaj/faiss/c_api/MetaIndexes_c.cpp > CMakeFiles/faiss_c.dir/MetaIndexes_c.cpp.i

c_api/CMakeFiles/faiss_c.dir/MetaIndexes_c.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/faiss_c.dir/MetaIndexes_c.cpp.s"
	cd /Users/sairaj/faiss/build/c_api && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/sairaj/faiss/c_api/MetaIndexes_c.cpp -o CMakeFiles/faiss_c.dir/MetaIndexes_c.cpp.s

c_api/CMakeFiles/faiss_c.dir/clone_index_c.cpp.o: c_api/CMakeFiles/faiss_c.dir/flags.make
c_api/CMakeFiles/faiss_c.dir/clone_index_c.cpp.o: /Users/sairaj/faiss/c_api/clone_index_c.cpp
c_api/CMakeFiles/faiss_c.dir/clone_index_c.cpp.o: c_api/CMakeFiles/faiss_c.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/sairaj/faiss/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_15) "Building CXX object c_api/CMakeFiles/faiss_c.dir/clone_index_c.cpp.o"
	cd /Users/sairaj/faiss/build/c_api && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT c_api/CMakeFiles/faiss_c.dir/clone_index_c.cpp.o -MF CMakeFiles/faiss_c.dir/clone_index_c.cpp.o.d -o CMakeFiles/faiss_c.dir/clone_index_c.cpp.o -c /Users/sairaj/faiss/c_api/clone_index_c.cpp

c_api/CMakeFiles/faiss_c.dir/clone_index_c.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/faiss_c.dir/clone_index_c.cpp.i"
	cd /Users/sairaj/faiss/build/c_api && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/sairaj/faiss/c_api/clone_index_c.cpp > CMakeFiles/faiss_c.dir/clone_index_c.cpp.i

c_api/CMakeFiles/faiss_c.dir/clone_index_c.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/faiss_c.dir/clone_index_c.cpp.s"
	cd /Users/sairaj/faiss/build/c_api && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/sairaj/faiss/c_api/clone_index_c.cpp -o CMakeFiles/faiss_c.dir/clone_index_c.cpp.s

c_api/CMakeFiles/faiss_c.dir/error_impl.cpp.o: c_api/CMakeFiles/faiss_c.dir/flags.make
c_api/CMakeFiles/faiss_c.dir/error_impl.cpp.o: /Users/sairaj/faiss/c_api/error_impl.cpp
c_api/CMakeFiles/faiss_c.dir/error_impl.cpp.o: c_api/CMakeFiles/faiss_c.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/sairaj/faiss/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_16) "Building CXX object c_api/CMakeFiles/faiss_c.dir/error_impl.cpp.o"
	cd /Users/sairaj/faiss/build/c_api && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT c_api/CMakeFiles/faiss_c.dir/error_impl.cpp.o -MF CMakeFiles/faiss_c.dir/error_impl.cpp.o.d -o CMakeFiles/faiss_c.dir/error_impl.cpp.o -c /Users/sairaj/faiss/c_api/error_impl.cpp

c_api/CMakeFiles/faiss_c.dir/error_impl.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/faiss_c.dir/error_impl.cpp.i"
	cd /Users/sairaj/faiss/build/c_api && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/sairaj/faiss/c_api/error_impl.cpp > CMakeFiles/faiss_c.dir/error_impl.cpp.i

c_api/CMakeFiles/faiss_c.dir/error_impl.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/faiss_c.dir/error_impl.cpp.s"
	cd /Users/sairaj/faiss/build/c_api && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/sairaj/faiss/c_api/error_impl.cpp -o CMakeFiles/faiss_c.dir/error_impl.cpp.s

c_api/CMakeFiles/faiss_c.dir/index_factory_c.cpp.o: c_api/CMakeFiles/faiss_c.dir/flags.make
c_api/CMakeFiles/faiss_c.dir/index_factory_c.cpp.o: /Users/sairaj/faiss/c_api/index_factory_c.cpp
c_api/CMakeFiles/faiss_c.dir/index_factory_c.cpp.o: c_api/CMakeFiles/faiss_c.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/sairaj/faiss/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_17) "Building CXX object c_api/CMakeFiles/faiss_c.dir/index_factory_c.cpp.o"
	cd /Users/sairaj/faiss/build/c_api && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT c_api/CMakeFiles/faiss_c.dir/index_factory_c.cpp.o -MF CMakeFiles/faiss_c.dir/index_factory_c.cpp.o.d -o CMakeFiles/faiss_c.dir/index_factory_c.cpp.o -c /Users/sairaj/faiss/c_api/index_factory_c.cpp

c_api/CMakeFiles/faiss_c.dir/index_factory_c.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/faiss_c.dir/index_factory_c.cpp.i"
	cd /Users/sairaj/faiss/build/c_api && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/sairaj/faiss/c_api/index_factory_c.cpp > CMakeFiles/faiss_c.dir/index_factory_c.cpp.i

c_api/CMakeFiles/faiss_c.dir/index_factory_c.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/faiss_c.dir/index_factory_c.cpp.s"
	cd /Users/sairaj/faiss/build/c_api && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/sairaj/faiss/c_api/index_factory_c.cpp -o CMakeFiles/faiss_c.dir/index_factory_c.cpp.s

c_api/CMakeFiles/faiss_c.dir/index_io_c.cpp.o: c_api/CMakeFiles/faiss_c.dir/flags.make
c_api/CMakeFiles/faiss_c.dir/index_io_c.cpp.o: /Users/sairaj/faiss/c_api/index_io_c.cpp
c_api/CMakeFiles/faiss_c.dir/index_io_c.cpp.o: c_api/CMakeFiles/faiss_c.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/sairaj/faiss/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_18) "Building CXX object c_api/CMakeFiles/faiss_c.dir/index_io_c.cpp.o"
	cd /Users/sairaj/faiss/build/c_api && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT c_api/CMakeFiles/faiss_c.dir/index_io_c.cpp.o -MF CMakeFiles/faiss_c.dir/index_io_c.cpp.o.d -o CMakeFiles/faiss_c.dir/index_io_c.cpp.o -c /Users/sairaj/faiss/c_api/index_io_c.cpp

c_api/CMakeFiles/faiss_c.dir/index_io_c.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/faiss_c.dir/index_io_c.cpp.i"
	cd /Users/sairaj/faiss/build/c_api && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/sairaj/faiss/c_api/index_io_c.cpp > CMakeFiles/faiss_c.dir/index_io_c.cpp.i

c_api/CMakeFiles/faiss_c.dir/index_io_c.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/faiss_c.dir/index_io_c.cpp.s"
	cd /Users/sairaj/faiss/build/c_api && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/sairaj/faiss/c_api/index_io_c.cpp -o CMakeFiles/faiss_c.dir/index_io_c.cpp.s

c_api/CMakeFiles/faiss_c.dir/impl/AuxIndexStructures_c.cpp.o: c_api/CMakeFiles/faiss_c.dir/flags.make
c_api/CMakeFiles/faiss_c.dir/impl/AuxIndexStructures_c.cpp.o: /Users/sairaj/faiss/c_api/impl/AuxIndexStructures_c.cpp
c_api/CMakeFiles/faiss_c.dir/impl/AuxIndexStructures_c.cpp.o: c_api/CMakeFiles/faiss_c.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/sairaj/faiss/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_19) "Building CXX object c_api/CMakeFiles/faiss_c.dir/impl/AuxIndexStructures_c.cpp.o"
	cd /Users/sairaj/faiss/build/c_api && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT c_api/CMakeFiles/faiss_c.dir/impl/AuxIndexStructures_c.cpp.o -MF CMakeFiles/faiss_c.dir/impl/AuxIndexStructures_c.cpp.o.d -o CMakeFiles/faiss_c.dir/impl/AuxIndexStructures_c.cpp.o -c /Users/sairaj/faiss/c_api/impl/AuxIndexStructures_c.cpp

c_api/CMakeFiles/faiss_c.dir/impl/AuxIndexStructures_c.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/faiss_c.dir/impl/AuxIndexStructures_c.cpp.i"
	cd /Users/sairaj/faiss/build/c_api && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/sairaj/faiss/c_api/impl/AuxIndexStructures_c.cpp > CMakeFiles/faiss_c.dir/impl/AuxIndexStructures_c.cpp.i

c_api/CMakeFiles/faiss_c.dir/impl/AuxIndexStructures_c.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/faiss_c.dir/impl/AuxIndexStructures_c.cpp.s"
	cd /Users/sairaj/faiss/build/c_api && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/sairaj/faiss/c_api/impl/AuxIndexStructures_c.cpp -o CMakeFiles/faiss_c.dir/impl/AuxIndexStructures_c.cpp.s

c_api/CMakeFiles/faiss_c.dir/utils/distances_c.cpp.o: c_api/CMakeFiles/faiss_c.dir/flags.make
c_api/CMakeFiles/faiss_c.dir/utils/distances_c.cpp.o: /Users/sairaj/faiss/c_api/utils/distances_c.cpp
c_api/CMakeFiles/faiss_c.dir/utils/distances_c.cpp.o: c_api/CMakeFiles/faiss_c.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/sairaj/faiss/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_20) "Building CXX object c_api/CMakeFiles/faiss_c.dir/utils/distances_c.cpp.o"
	cd /Users/sairaj/faiss/build/c_api && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT c_api/CMakeFiles/faiss_c.dir/utils/distances_c.cpp.o -MF CMakeFiles/faiss_c.dir/utils/distances_c.cpp.o.d -o CMakeFiles/faiss_c.dir/utils/distances_c.cpp.o -c /Users/sairaj/faiss/c_api/utils/distances_c.cpp

c_api/CMakeFiles/faiss_c.dir/utils/distances_c.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/faiss_c.dir/utils/distances_c.cpp.i"
	cd /Users/sairaj/faiss/build/c_api && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/sairaj/faiss/c_api/utils/distances_c.cpp > CMakeFiles/faiss_c.dir/utils/distances_c.cpp.i

c_api/CMakeFiles/faiss_c.dir/utils/distances_c.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/faiss_c.dir/utils/distances_c.cpp.s"
	cd /Users/sairaj/faiss/build/c_api && /Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/sairaj/faiss/c_api/utils/distances_c.cpp -o CMakeFiles/faiss_c.dir/utils/distances_c.cpp.s

# Object files for target faiss_c
faiss_c_OBJECTS = \
"CMakeFiles/faiss_c.dir/AutoTune_c.cpp.o" \
"CMakeFiles/faiss_c.dir/Clustering_c.cpp.o" \
"CMakeFiles/faiss_c.dir/IndexFlat_c.cpp.o" \
"CMakeFiles/faiss_c.dir/IndexIVFFlat_c.cpp.o" \
"CMakeFiles/faiss_c.dir/IndexIVF_c.cpp.o" \
"CMakeFiles/faiss_c.dir/IndexLSH_c.cpp.o" \
"CMakeFiles/faiss_c.dir/IndexPreTransform_c.cpp.o" \
"CMakeFiles/faiss_c.dir/VectorTransform_c.cpp.o" \
"CMakeFiles/faiss_c.dir/IndexShards_c.cpp.o" \
"CMakeFiles/faiss_c.dir/IndexReplicas_c.cpp.o" \
"CMakeFiles/faiss_c.dir/Index_c.cpp.o" \
"CMakeFiles/faiss_c.dir/IndexBinary_c.cpp.o" \
"CMakeFiles/faiss_c.dir/IndexScalarQuantizer_c.cpp.o" \
"CMakeFiles/faiss_c.dir/MetaIndexes_c.cpp.o" \
"CMakeFiles/faiss_c.dir/clone_index_c.cpp.o" \
"CMakeFiles/faiss_c.dir/error_impl.cpp.o" \
"CMakeFiles/faiss_c.dir/index_factory_c.cpp.o" \
"CMakeFiles/faiss_c.dir/index_io_c.cpp.o" \
"CMakeFiles/faiss_c.dir/impl/AuxIndexStructures_c.cpp.o" \
"CMakeFiles/faiss_c.dir/utils/distances_c.cpp.o"

# External object files for target faiss_c
faiss_c_EXTERNAL_OBJECTS =

c_api/libfaiss_c.dylib: c_api/CMakeFiles/faiss_c.dir/AutoTune_c.cpp.o
c_api/libfaiss_c.dylib: c_api/CMakeFiles/faiss_c.dir/Clustering_c.cpp.o
c_api/libfaiss_c.dylib: c_api/CMakeFiles/faiss_c.dir/IndexFlat_c.cpp.o
c_api/libfaiss_c.dylib: c_api/CMakeFiles/faiss_c.dir/IndexIVFFlat_c.cpp.o
c_api/libfaiss_c.dylib: c_api/CMakeFiles/faiss_c.dir/IndexIVF_c.cpp.o
c_api/libfaiss_c.dylib: c_api/CMakeFiles/faiss_c.dir/IndexLSH_c.cpp.o
c_api/libfaiss_c.dylib: c_api/CMakeFiles/faiss_c.dir/IndexPreTransform_c.cpp.o
c_api/libfaiss_c.dylib: c_api/CMakeFiles/faiss_c.dir/VectorTransform_c.cpp.o
c_api/libfaiss_c.dylib: c_api/CMakeFiles/faiss_c.dir/IndexShards_c.cpp.o
c_api/libfaiss_c.dylib: c_api/CMakeFiles/faiss_c.dir/IndexReplicas_c.cpp.o
c_api/libfaiss_c.dylib: c_api/CMakeFiles/faiss_c.dir/Index_c.cpp.o
c_api/libfaiss_c.dylib: c_api/CMakeFiles/faiss_c.dir/IndexBinary_c.cpp.o
c_api/libfaiss_c.dylib: c_api/CMakeFiles/faiss_c.dir/IndexScalarQuantizer_c.cpp.o
c_api/libfaiss_c.dylib: c_api/CMakeFiles/faiss_c.dir/MetaIndexes_c.cpp.o
c_api/libfaiss_c.dylib: c_api/CMakeFiles/faiss_c.dir/clone_index_c.cpp.o
c_api/libfaiss_c.dylib: c_api/CMakeFiles/faiss_c.dir/error_impl.cpp.o
c_api/libfaiss_c.dylib: c_api/CMakeFiles/faiss_c.dir/index_factory_c.cpp.o
c_api/libfaiss_c.dylib: c_api/CMakeFiles/faiss_c.dir/index_io_c.cpp.o
c_api/libfaiss_c.dylib: c_api/CMakeFiles/faiss_c.dir/impl/AuxIndexStructures_c.cpp.o
c_api/libfaiss_c.dylib: c_api/CMakeFiles/faiss_c.dir/utils/distances_c.cpp.o
c_api/libfaiss_c.dylib: c_api/CMakeFiles/faiss_c.dir/build.make
c_api/libfaiss_c.dylib: faiss/libfaiss.dylib
c_api/libfaiss_c.dylib: c_api/CMakeFiles/faiss_c.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/Users/sairaj/faiss/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_21) "Linking CXX shared library libfaiss_c.dylib"
	cd /Users/sairaj/faiss/build/c_api && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/faiss_c.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
c_api/CMakeFiles/faiss_c.dir/build: c_api/libfaiss_c.dylib
.PHONY : c_api/CMakeFiles/faiss_c.dir/build

c_api/CMakeFiles/faiss_c.dir/clean:
	cd /Users/sairaj/faiss/build/c_api && $(CMAKE_COMMAND) -P CMakeFiles/faiss_c.dir/cmake_clean.cmake
.PHONY : c_api/CMakeFiles/faiss_c.dir/clean

c_api/CMakeFiles/faiss_c.dir/depend:
	cd /Users/sairaj/faiss/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/sairaj/faiss /Users/sairaj/faiss/c_api /Users/sairaj/faiss/build /Users/sairaj/faiss/build/c_api /Users/sairaj/faiss/build/c_api/CMakeFiles/faiss_c.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : c_api/CMakeFiles/faiss_c.dir/depend

