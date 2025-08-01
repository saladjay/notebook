# 创建共享库
add_library(intercept_lib SHARED intercept.c)

# 链接 dl 库（用于 dlsym）
target_link_libraries(intercept_lib dl)