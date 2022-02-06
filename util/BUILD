licenses(["notice"])

package(default_visibility = ["//visibility:private"])

cc_library(name = "landmark_standardization",
    srcs        = ["landmark_standardization.cc"],
    visibility  = ["//visibility:public"],
    deps        = [
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/port:status",
        "//mediapipe/framework/port:opencv_imgproc",
        "//mediapipe/framework/formats:landmark_cc_proto",
        "//mediapipe/framework/formats:detection_cc_proto",
    ],
    alwayslink = 1,
)
