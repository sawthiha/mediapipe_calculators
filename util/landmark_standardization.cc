#include <vector>

#include "absl/memory/memory.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/detection.pb.h"
#include "mediapipe/framework/port/opencv_core_inc.h"

namespace mediapipe
{
    /**
     * @brief Detect face alignments from Standardized Landmarks
     * 
     * INPUTS:
     *      IMAGE - Reference Image, serves as tick signal
     *      0 - Standardized Landmarks (NormalizedLandmarkList)
     * OUTPUTS:
     *      0 - Standardized Landmarks (NormalizedLandmarkList)
     * 
     * Example:
     * 
     * node {
     *   calculator: "LandmarkStandardizationCalculator"
     *   input_stream: "face_landmarks"
     *   output_stream: "face_std_landmarks"
     * }
     * 
     */
    class LandmarkStandardizationCalculator: public CalculatorBase
    {
    public:
        LandmarkStandardizationCalculator() = default;
        ~LandmarkStandardizationCalculator() override = default;

        static absl::Status GetContract(CalculatorContract* cc);

        absl::Status Open(CalculatorContext* cc) override;
        absl::Status Process(CalculatorContext* cc) override;
        absl::Status Close(CalculatorContext* cc) override;
    };

    // Register the calculator to be used in the graph
    REGISTER_CALCULATOR(LandmarkStandardizationCalculator);

    absl::Status LandmarkStandardizationCalculator::GetContract(CalculatorContract* cc)
    {
        cc->Inputs().Index(0).Set<NormalizedLandmarkList>();
        cc->Outputs().Index(0).Set<NormalizedLandmarkList>();
        return absl::OkStatus();
    }

    absl::Status LandmarkStandardizationCalculator::Open(CalculatorContext* cc)
    {
        return absl::OkStatus();
    }

    absl::Status LandmarkStandardizationCalculator::Process(CalculatorContext* cc)
    {
        auto landmarks = cc->Inputs().Index(0).Get<NormalizedLandmarkList>();
        cv::Mat mat(landmarks.landmark_size(), 3, CV_64FC1);
        cv::Mat norm_mat(landmarks.landmark_size(), 3, CV_64FC1);
        cv::Mat mean_mat, std_mat;

        for (int i = 0; i < landmarks.landmark_size(); ++i) {
            mat.at<double>(i, 0) = landmarks.landmark(i).x();
            mat.at<double>(i, 1) = landmarks.landmark(i).y();
            mat.at<double>(i, 2) = landmarks.landmark(i).z();
        }

        for (int i = 0; i < mat.cols; i++) {
            cv::meanStdDev(mat.col(i), mean_mat, std_mat);
            norm_mat.col(i) = (mat.col(i) - mean_mat.at<double>(0, 0)) / std_mat.at<double>(0, 0);
        }

        NormalizedLandmarkList norm_landmarks;
        for (int i = 0; i < landmarks.landmark_size(); ++i) {
            NormalizedLandmark* landmark = norm_landmarks.add_landmark();
            landmark->set_x(norm_mat.at<double>(i, 0));
            landmark->set_y(norm_mat.at<double>(i, 1));
            landmark->set_z(norm_mat.at<double>(i, 2));
        }

        Packet packet = MakePacket<decltype(norm_landmarks)>(norm_landmarks).At(cc->InputTimestamp());
        cc->Outputs().Index(0).AddPacket(packet);

        return absl::OkStatus();
    } // Process()

    absl::Status LandmarkStandardizationCalculator::Close(CalculatorContext* cc)
    { return absl::OkStatus(); }

} // namespace mediapipe
