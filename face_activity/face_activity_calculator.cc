#include <vector>
#include <map>
#include <optional>

#include "absl/memory/memory.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/port/opencv_core_inc.h"
#include "mediapipe/framework/formats/landmark.pb.h"

namespace mediapipe
{
    /**
     * @brief Detect facial activity changes
     * 
     * INPUTS:
     *      0 - Standardized Landmarks (NormalizedLandmarkList)
     * OUTPUTS:
     *      0 - Facial Activity Delta (double)
     * 
     * Example:
     * 
     * node {
     *   calculator: "FaceActivityCalculator"
     *   input_stream: "face_std_landmarks"
     *   output_stream: "face_activities"
     * }
     * 
     */
    class FaceActivityCalculator: public CalculatorBase
    {
    private:
        cv::Mat m_prev_landmark_mat;

    public:
        FaceActivityCalculator() = default;
        ~FaceActivityCalculator() override = default;

        static absl::Status GetContract(CalculatorContract* cc);

        absl::Status Open(CalculatorContext* cc) override;
        absl::Status Process(CalculatorContext* cc) override;
        absl::Status Close(CalculatorContext* cc) override;
    };

    // Register the calculator to be used in the graph
    REGISTER_CALCULATOR(FaceActivityCalculator);

    absl::Status FaceActivityCalculator::GetContract(CalculatorContract* cc)
    {
        cc->Inputs().Index(0).Set<NormalizedLandmarkList>();
        cc->Outputs().Index(0).Set<double>();
        return absl::OkStatus();
    }

    absl::Status FaceActivityCalculator::Open(CalculatorContext* cc)
    { return absl::OkStatus(); }

    absl::Status FaceActivityCalculator::Process(CalculatorContext* cc)
    {
        auto landmarks = cc->Inputs().Index(0).Get<NormalizedLandmarkList>();
        cv::Mat cur_landmark_mat(landmarks.landmark_size(), 3, CV_64FC1);

        for (int i = 0; i < landmarks.landmark_size(); ++i) {
            cur_landmark_mat.at<double>(i, 0) = landmarks.landmark(i).x();
            cur_landmark_mat.at<double>(i, 1) = landmarks.landmark(i).y();
            cur_landmark_mat.at<double>(i, 2) = landmarks.landmark(i).z();
        }
        
        // Initialize prev_landmark_mat
        if (m_prev_landmark_mat.size().area() == 0)
        {
            m_prev_landmark_mat = cur_landmark_mat;
        }
        
        auto delta = cv::norm(cur_landmark_mat - m_prev_landmark_mat, cv::NORM_L2);
        m_prev_landmark_mat = cur_landmark_mat;
            
        Packet packet = MakePacket<double>(delta).At(cc->InputTimestamp());
        cc->Outputs().Index(0).AddPacket(packet);

        return absl::OkStatus();
    } // Process()

    absl::Status FaceActivityCalculator::Close(CalculatorContext* cc)
    { return absl::OkStatus(); }

} // namespace mediapipe
