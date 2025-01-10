#----------------------------------------------------------------
# Generated CMake target import file for configuration "Debug".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "faiss" for configuration "Debug"
set_property(TARGET faiss APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(faiss PROPERTIES
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/lib/libfaiss.dylib"
  IMPORTED_SONAME_DEBUG "@rpath/libfaiss.dylib"
  )

list(APPEND _cmake_import_check_targets faiss )
list(APPEND _cmake_import_check_files_for_faiss "${_IMPORT_PREFIX}/lib/libfaiss.dylib" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
