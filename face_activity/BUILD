licenses(["notice"])

package(default_visibility = ["//visibility:private"])

cc_library(name = "face_movement_calculator",
    srcs        = ["face_movement_calculator.cc"],
    visibility  = ["//visibility:public"],
    deps        = [
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/port:status",
        "//mediapipe/framework/port:opencv_core",
        "//mediapipe/framework/formats:landmark_cc_proto",
        "//mediapipe/calculators/core:begin_loop_calculator",
    ],
    alwayslink = 1,
)

cc_library(name = "face_activity_calculator",
    srcs        = ["face_activity_calculator.cc"],
    visibility  = ["//visibility:public"],
    deps        = [
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/port:status",
        "//mediapipe/framework/port:opencv_core",
        "//mediapipe/framework/formats:landmark_cc_proto",
        "//mediapipe/calculators/core:begin_loop_calculator",
    ],
    alwayslink = 1,
)
