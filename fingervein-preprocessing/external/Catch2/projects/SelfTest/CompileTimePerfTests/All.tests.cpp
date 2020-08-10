// include set of usage tests into one file for compiler performance test purposes
// This whole file can now be included multiple times in 10.tests.cpp, and *that*
// file included multiple times (in 100.tests.cpp)

// Note that the intention is only for these files to be compiled. They will
// fail at runtime due to the re-user of test case names

#include "external/Catch2/projects/SelfTest/UsageTests/Approx.tests.cpp"
#include "external/Catch2/projects/SelfTest/UsageTests/BDD.tests.cpp"
#include "external/Catch2/projects/SelfTest/UsageTests/Class.tests.cpp"
#include "external/Catch2/projects/SelfTest/UsageTests/Compilation.tests.cpp"
#include "external/Catch2/projects/SelfTest/UsageTests/Condition.tests.cpp"
#include "external/Catch2/projects/SelfTest/UsageTests/Exception.tests.cpp"
#include "external/Catch2/projects/SelfTest/UsageTests/Matchers.tests.cpp"
#include "external/Catch2/projects/SelfTest/UsageTests/Misc.tests.cpp"
