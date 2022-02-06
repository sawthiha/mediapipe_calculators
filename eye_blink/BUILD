licenses(["notice"])

package(default_visibility = ["//visibility:private"])

cc_library(name = "eye_blink_calculator",
    srcs        = ["eye_blink_calculator.cc"],
    visibility  = ["//visibility:public"],
    deps        = [
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/port:status",
        "//mediapipe/framework/formats:landmark_cc_proto",
        "//mediapipe/util:render_data_cc_proto",
        "//mediapipe/framework/port:opencv_imgproc",
        "//mediapipe/calculators/core:begin_loop_calculator",
    ],
    alwayslink = 1,
)

cc_library(name = "eye_blink_to_render_data_calculator",
    srcs        = ["eye_blink_to_render_data_calculator.cc"],
    visibility  = ["//visibility:public"],
    deps        = [
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/port:status",
        "//mediapipe/framework/formats:landmark_cc_proto",
        "//mediapipe/util:render_data_cc_proto",
    ],
    alwayslink = 1,
)
