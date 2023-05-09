TARGET_EXEC := gpu_arch_test
DEBUG_EXEC := gpu_arch_test_debug

BUILD_DIR = ./bin
SRC_DIRS := ./src
DBG_DIRS := ./debug

# Compiler Flags
CC     	:= gcc
CFLAGS 	:= -O3
AR		:= ar
ARFLAGS := -rcs
LD    	:= nvcc
NVCC   	:= nvcc
# Targeting Compute 5.X to 9.0, disable as required
NVFLAGS := -O3 -gencode arch=compute_50,code=sm_50 \
			  -gencode arch=compute_52,code=sm_52 \
			  -gencode arch=compute_60,code=sm_60 \
			  -gencode arch=compute_61,code=sm_61 \
			  -gencode arch=compute_70,code=sm_70 \
			  -gencode arch=compute_75,code=sm_75 \
			  -gencode arch=compute_80,code=sm_80 \
			  -gencode arch=compute_86,code=sm_86 \
			  -gencode arch=compute_89,code=sm_89 \
			  -gencode arch=compute_90,code=sm_90 \
			  -rdc=true

# Find all the C and CU files we want to compile
# Note the single quotes around the * expressions. The shell will incorrectly expand these otherwise, but we want to send the * directly to the find command.
SRCS := $(shell find $(SRC_DIRS) -name '*.cu' -or -name '*.c')
NULL := 
DBG_SRCS := $(shell find $(DBG_DIRS) -name '*.cu' -or -name '*.c') $(patsubst ./src/main.cu,$(NULL),$(SRCS))

# Prepends BUILD_DIR and appends .o to every src file
# As an example, ./your_dir/hello.cpp turns into ./build/./your_dir/hello.cpp.o
OBJS := $(SRCS:%=$(BUILD_DIR)/%.o)
DBG_OBJS := $(DBG_SRCS:%=$(BUILD_DIR)/%.o)

# String substitution (suffix version without %).
# As an example, ./build/hello.cpp.o turns into ./build/hello.cpp.d
DEPS := $(OBJS:.o=.d)
DBG_DEPS := $(DBG_OBJS:.o=.d)

# Every folder in ./src will need to be passed to GCC so that it can find header files
INC_DIRS := $(shell find $(SRC_DIRS) -type d)
DBG_INC_DIRS := $(shell find $(DBG_DIRS) -type d)
# Add a prefix to INC_DIRS. So moduleA would become -ImoduleA. GCC understands this -I flag
INC_FLAGS := $(addprefix -I,$(INC_DIRS))
DBG_INC_FLAGS := $(addprefix -I,$(DBG_INC_DIRS))

# The -MMD and -MP flags together generate Makefiles for us!
# These files will have .d instead of .o as the output.
DEPFLAGS := $(INC_FLAGS) -MMD -MP
DBG_DEPFLAGS := $(DBG_INC_FLAGS) -MMD -MP


all:
	echo "doing nothing"

.PHONY: clean clean_debug release debug
clean:
	rm -r $(BUILD_DIR)
clean_debug:
	rm -r ./debug

release: $(OBJS)
	$(LD) $(OBJS) -o $@ $(NVFLAGS) $(DEP_FLAGS) 

debug: $(DBG_OBJS)
	$(LD) -o $(BUILD_DIR)/$(DEBUG_EXEC) $(DBG_OBJS) ./src/main.cu $(NVFLAGS) -g -G -O0 -DDEBUG $(DBG_DEPFLAGS) 


# Generic build patterns

# Build step for C source
$(BUILD_DIR)/%.c.o: %.c
	mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@

# Build step for Cuda source
$(BUILD_DIR)/%.cu.o: %.cu
	mkdir -p $(dir $@)
	$(NVCC) $(NVFLAGS) -c $< -o $@ -diag-suppress 2464


# Include the .d makefiles. The - at the front suppresses the errors of missing
# Makefiles. Initially, all the .d files will be missing, and we don't want those
# errors to show up.
-include $(DEPS)
-include $(DBG_DEPS)